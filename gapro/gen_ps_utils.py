import numpy as np
import scipy.ndimage
import torch

from tqdm import tqdm

import torch_scatter
import torch.nn.functional as F

from gaussian_process_utils import fit_gp, fit_regression_model, fit_gp_spp

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    N, C = xyz.shape
    # print("DEBUG", N)
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)
    distance = torch.ones(N).to(device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[ farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def batch_giou_cross(boxes1, boxes2):
    # boxes1: N, 6
    # boxes2: M, 6
    # out: N, M
    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]
    intersection = torch.prod(torch.clamp((torch.min(boxes1[..., 3:], boxes2[..., 3:]) - torch.max(boxes1[..., :3], boxes2[..., :3])), min=0.0), -1)  # N


    boxes1_volumes = torch.prod(torch.clamp((boxes1[..., 3:] - boxes1[..., :3]), min=0.0), -1)
    boxes2_volumes = torch.prod(torch.clamp((boxes2[..., 3:] - boxes2[..., :3]), min=0.0), -1)

    union = boxes1_volumes + boxes2_volumes - intersection
    iou = intersection / (union + 1e-6)

    volumes_bound = torch.prod(torch.clamp((torch.max(boxes1[..., 3:], boxes2[..., 3:]) - torch.min(boxes1[..., :3], boxes2[..., :3])), min=0.0), -1)  # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou

def get_cropped_instance_label(instance_label, valid_idxs=None):
    if valid_idxs is not None:
        instance_label = instance_label[valid_idxs]
    j = 0
    while j < instance_label.max():
        if (instance_label == j).sum() == 0:
            instance_label[instance_label == instance_label.max()] = j
        j += 1
    return instance_label

def is_box1_in_box2(box1, box2, offset=0.05):
    return torch.all((box1[:3] + offset) >= box2[:3]) & torch.all((box1[3:] - offset) <= box2[3:])


def is_within_bb_torch(points, bb_min, bb_max):
    return torch.all(points >= bb_min, dim=-1) & torch.all(points <= bb_max, dim=-1)

def is_within_bb_np(points, bb_min, bb_max):
    return np.all(points >= bb_min, axis=-1) & np.all(points <= bb_max, axis=-1)

def major_voting(inst_masks, sem_pred):
    r_inds, c_inds = torch.nonzero(inst_masks, as_tuple=True)
    sem_ = sem_pred[c_inds]

    sem_onehot = F.one_hot(sem_.long(), num_classes=20) # N, n_classes

    count_sem = torch_scatter.scatter_add(sem_onehot, index=r_inds, dim=0) # Q, n_classes

    label = torch.argmax(count_sem, dim=1) # Q
    return label


def spp_align_label(spp, label, n_classes=-1, bb_occupancy_spp=None, prob_label=None):
    if n_classes == -1:
        n_classes = torch.max(label)+1

    # breakpoint()
    onehot_label = F.one_hot(label.long(), num_classes=n_classes).permute(1,0)

    n_label, n_points = onehot_label.shape[:2]
    _, spp_ids = torch.unique(spp, return_inverse=True)

    count_label_spp = torch_scatter.scatter(
        onehot_label, spp_ids.expand(n_label, n_points), dim=-1, reduce="sum"
    )  # n_labels, n_spp
    # spp_mask = (mean_spp_inst > 0.5)


    count_label_spp_sum = (count_label_spp).sum(dim=0)

    if bb_occupancy_spp is not None:
        count_label_spp[1:, :] = count_label_spp[1:, :] * (bb_occupancy_spp == 1).float()

    label_spp = torch.argmax(count_label_spp, dim=0) # n_spp
    

    refined_label = label_spp[spp_ids]

    if prob_label is None:
        return refined_label
    
    prob_label_spp = torch_scatter.scatter(prob_label, spp_ids, reduce="mean")

    refined_prob_label = prob_label_spp[spp_ids]
    return refined_label, refined_prob_label

    
def spp_major_voting(spp, label, prob_label, bb_occupancy, n_classes):
    n_points = len(label)
    onehot_label = F.one_hot(label.long(), num_classes=n_classes) # npoint, nlabel
    onehot_prob_label = torch.zeros((n_points, n_classes), dtype=torch.float, device=label.device)
    onehot_prob_label[torch.arange(n_points, dtype=torch.long, device=label.device), label.long()] = prob_label
    # onehot_prob_label = torch.scatter(, dim, onehot_label, prob_label)

    _, spp_ids = torch.unique(spp, return_inverse=True)

    bb_occupancy_spp = torch_scatter.scatter(
        bb_occupancy.float(), spp[:, None].expand(n_points, n_classes-1), dim=0, reduce="mean"
    ) # n_spp, n_label

    count_label_spp = torch_scatter.scatter(
        onehot_label, spp_ids[:, None].expand(n_points, n_classes), dim=0, reduce="sum"
    )  # n_spp, n_labels

    prob_label_spp = torch_scatter.scatter(
        onehot_prob_label, spp_ids[:, None].expand(n_points, n_classes), dim=0, reduce="sum"
    )  # n_spp, n_labels
    prob_label_spp = prob_label_spp / (count_label_spp + 1e-4)

    assert not torch.any(prob_label_spp > 1)
    
    count_label_spp_sum = count_label_spp.sum(dim=1) # spp

    count_label_spp[:, 1:] = count_label_spp[:, 1:] * (bb_occupancy_spp == 1).float()
    label_spp = torch.argmax(count_label_spp, dim=1) # n_spp

    prob_spp = (prob_label_spp * (count_label_spp / count_label_spp_sum[:, None])).sum(dim=1)
    # prob_spp = torch_scatter.scatter(prob_label, spp_ids, reduce="mean")
    refined_label = label_spp[spp_ids]
    refined_prob_label = prob_spp[spp_ids]
    # refined_prob_label = prob_label
    return refined_label, refined_prob_label

def spp_major_voting_onehot(spp, onehot_label, onehot_prob_label, bb_occupancy, n_classes):
    n_points = len(spp)

    _, spp_ids = torch.unique(spp, return_inverse=True)

    bb_occupancy_spp = torch_scatter.scatter(
        bb_occupancy.float(), spp[:, None].expand(n_points, n_classes), dim=0, reduce="mean"
    ) # n_spp, n_label


    mean_spp_inst = torch_scatter.scatter(
        onehot_label.float(), spp_ids.expand(n_classes, n_points), dim=1, reduce="mean"
    )  # n_inst, n_spp

    prob_label_spp = torch_scatter.scatter(
        onehot_prob_label.float(), spp_ids.expand(n_classes, n_points), dim=1, reduce="mean"
    )  # n_spp, n_labels

    spp_mask = (mean_spp_inst >= 0.2) * (bb_occupancy_spp == 1).float().permute(1,0)
    # spp_mask = (mean_spp_inst >= 0.5)

    refined_label = torch.gather(spp_mask, dim=1, index=spp_ids.expand(n_classes, n_points))
    refined_prob_label = torch.gather(prob_label_spp, dim=1, index=spp_ids.expand(n_classes, n_points))


    return refined_label, refined_prob_label


def getInstanceInfo(xyz, instance_label, semantic_label, dataset_name="scannetv2"):

    instance_cls = []
    instance_box = []
    instance_box_volume = []
    instance_num = int(instance_label.max()) + 1

    # pt_offset_vertices_label = np.ones((xyz.shape[0], 3 * 2), dtype=np.float32) * -100.0
    # center_label = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
    corners_label = np.ones((xyz.shape[0], 3 * 2), dtype=np.float32) * -100.0
    # breakpoint()
    for i_ in range(instance_num):
        inst_idx_i = np.where(instance_label == i_)
        
        if len(inst_idx_i[0]) == 0: continue

        cls_idx = inst_idx_i[0][0]
        sem_label = semantic_label[cls_idx]

        # if sem_label == -100: continue

        xyz_i = xyz[inst_idx_i]

        min_xyz_i = xyz_i.min(0)
        max_xyz_i = xyz_i.max(0)
        corners_label[inst_idx_i, :3] = min_xyz_i - xyz_i
        corners_label[inst_idx_i, 3:] = max_xyz_i - xyz_i


        # if sem_label == -100: continue
        instance_box.append(np.concatenate([min_xyz_i, max_xyz_i], axis=0))
        instance_cls.append(sem_label)
        instance_box_volume.append(np.prod(np.clip((max_xyz_i - min_xyz_i), a_min=0.0, a_max=None)))

    if len(instance_cls) == 0:
        return None

    instance_cls = np.array(instance_cls)
    instance_box = np.stack(instance_box, axis=0)  # N, 6
    instance_box_volume = np.array(instance_box_volume)

    if dataset_name == 'scannetv2':
        instance_cls[instance_cls!=-100] -= 2

    return instance_num, instance_cls, instance_box, instance_box_volume, corners_label

def gen_pseudo_label_box2mask(coords_float, spp, instance_cls, instance_box, instance_box_volume, instance_classes=18, dataset_name="scannetv2"):
    n_points_ = len(coords_float)
    n_boxes = len(instance_box)


    bb_occupancy = is_within_bb_torch(coords_float[:, None, :], instance_box[None, :, :3] - 0.005, instance_box[None, :, 3:] + 0.005) # N_points, N_box

    # none_cond = (instance_cls[in_s:in_t] == -100)
    # bb_occupancy[:, none_cond] = 0

    inst_per_point = torch.ones(n_points_, dtype=torch.long, device=coords_float.device) * -100
    # activations_per_point = [np.argwhere(bb_occupancy[:, i] == 1) for i in range(len(scene['positions']))]
    # number of BBs a point is in
    num_BBs_per_point = bb_occupancy.long().sum(dim=1)

    # NOTE process points within multi boxes
    point_inds, box_inds = torch.nonzero(bb_occupancy[(num_BBs_per_point > 1),:], as_tuple=True)
    volume_box = instance_box_volume[box_inds]
    min_volume, argmin_volume = torch_scatter.scatter_min(volume_box, point_inds, dim=0)
    assert len(min_volume) == (num_BBs_per_point > 1).sum()
    corres_multibox_active = box_inds[argmin_volume]

    corres_box_active = torch.nonzero(bb_occupancy[(num_BBs_per_point == 1),:], as_tuple=True)[1]

    inst_per_point[(num_BBs_per_point == 1)] = corres_box_active # NOTE: 1 box -> assign corres box
    inst_per_point[(num_BBs_per_point == 0)] = -1 # NOTE: nobox -> background -1
    inst_per_point[(num_BBs_per_point > 1)] = corres_multibox_active # NOTE: multibox -> assign box with the smallest volume

    if dataset_name == 'scannetv2':

        _, spp = torch.unique(spp, return_inverse=True)

        inst_per_point = torch.where(inst_per_point >= 0, inst_per_point+1, 0)
        inst_per_point = spp_align_label(spp, inst_per_point, n_classes=n_boxes+1)
        inst_per_point = torch.where(inst_per_point > 0, inst_per_point-1, -1).int()


    ps_semantic_label = torch.ones(n_points_, dtype=torch.int, device=coords_float.device) * -100
    ps_instance_label = torch.ones(n_points_, dtype=torch.int, device=coords_float.device) * -100
    ps_semantic_label[inst_per_point >= 0] = instance_cls[inst_per_point[inst_per_point >= 0].long()].int()
    ps_semantic_label[inst_per_point == -1] = instance_classes

    ps_instance_label[inst_per_point >= 0] = inst_per_point[inst_per_point >= 0].int()

    return ps_semantic_label, ps_instance_label


def gen_pseudo_label_gaussian_process(coords_float, mask_feats, spp, instance_cls, instance_box, instance_box_volume, wall_box, wall_box_volume, instance_classes=18, dataset_name="scannetv2", ground_h=0.1, training_iter=50, thresh_spp_occu=0.8):
    max_num = 1000000
    n_points = len(coords_float)
    n_fg_instances = len(instance_box)

    unique_spps, spp = torch.unique(spp, return_inverse=True)
    n_spps = len(unique_spps)

    gp_feats = mask_feats.float()

    min_range = torch.min(coords_float, dim=0)[0]
    max_range = torch.max(coords_float, dim=0)[0]
    floor_min_z = min_range[2]
    floor_max_z = floor_min_z + ground_h
    floor_box = torch.tensor([min_range[0], min_range[1], min_range[2], max_range[0], max_range[1], floor_max_z])[None, :].to(coords_float.device) # 1x6
    floor_box_volume = torch.prod(torch.clamp(floor_box[:, 3:] - floor_box[:, :3], min=0.001), dim=1)


    if len(wall_box) > 0:
        boxes = torch.cat([instance_box, wall_box, floor_box])
        boxes_cls = torch.cat([instance_cls, torch.ones((len(wall_box) + 1), dtype=instance_cls.dtype, device=instance_cls.device) * instance_classes], dim=0)
        boxes_volume = torch.cat([instance_box_volume, wall_box_volume, floor_box_volume], dim=0)
    else:
        boxes = torch.cat([instance_box, floor_box])
        boxes_cls = torch.cat([instance_cls, torch.ones(1, dtype=instance_cls.dtype, device=instance_cls.device) * instance_classes], dim=0)
        boxes_volume = torch.cat([instance_box_volume, floor_box_volume], dim=0)

    n_boxes = len(boxes)

    bb_occupancy = is_within_bb_torch(coords_float[:, None, :], boxes[None, :, :3] - 0.005, boxes[None, :, 3:] + 0.005) # N_points, N_box
    num_BBs_per_point = bb_occupancy.long().sum(dim=1)

    coords_float_spp = torch_scatter.scatter(coords_float, spp[:, None].expand(-1, coords_float.shape[1]), dim=0, reduce="mean")
    gp_feats_spp = torch_scatter.scatter(gp_feats, spp[:, None].expand(-1, gp_feats.shape[1]), dim=0, reduce="mean")

    bb_occupancy_spp = torch_scatter.scatter(
        bb_occupancy.float(), spp[:, None].expand(n_points, n_boxes), dim=0, reduce="mean"
    ) # n_spp, n_label
    bb_occupancy_spp = (bb_occupancy_spp >= thresh_spp_occu)
    n_bbs_per_spp = bb_occupancy_spp.long().sum(dim=1)

    inst_per_point = torch.ones(n_spps, dtype=torch.int, device=coords_float.device) * -100
    is_determined = torch.zeros(n_spps, dtype=torch.long, device=coords_float.device)
    ps_prob_label = torch.zeros(n_spps, dtype=torch.float, device=coords_float.device)
    ps_mu_label = torch.zeros(n_spps, dtype=torch.float, device=coords_float.device) - 100.0
    ps_variance_label = torch.zeros(n_spps, dtype=torch.float, device=coords_float.device) - 100.0


    # no box -> -1, 1 box -> corres

    corres_box_active = torch.nonzero(bb_occupancy_spp[(n_bbs_per_spp == 1)], as_tuple=True)[1]

    inst_per_point[(n_bbs_per_spp == 1)] = corres_box_active.int() # NOTE: 1 box -> assign corres box
    ps_prob_label[(n_bbs_per_spp == 1)] = 1
    is_determined[(n_bbs_per_spp == 1)] = max_num

    # inst_one_hot[:, torch.nonzero(num_BBs_per_point == 0).view(-1)] = 1
    # inst_one_hot_prob[:, (num_BBs_per_point == 0)] = 1
    inst_per_point[(n_bbs_per_spp == 0)] = -1 # NOTE: nobox -> background -1
    ps_prob_label[(n_bbs_per_spp == 0)] = 1
    is_determined[(n_bbs_per_spp == 0)] = max_num

    cross_box_iou, _ = batch_giou_cross(boxes, boxes) # N_box, N_box
    cross_box_iou.fill_diagonal_(0.0)

    box_visited = torch.zeros(n_boxes, dtype=torch.bool, device=coords_float.device)
    
    for b1 in tqdm(range(n_boxes)):
        b1_ious = cross_box_iou[b1]

        overlap_cond = (b1_ious > 0.0001) & (box_visited == 0)
        overlap_inds = torch.nonzero(overlap_cond).view(-1).int()
        n_overlap_ = len(overlap_inds)

        if n_overlap_ == 0:
            box_visited[b1] = 1
            continue
        
        for b2 in overlap_inds:
            assert b1 != b2
            intersect_cond = (bb_occupancy_spp[:, b1] == 1) & (bb_occupancy_spp[:, b2] == 1)

            intersect_inds = torch.nonzero(intersect_cond).view(-1)
            num_intersect_points = len(intersect_inds)
            
            if num_intersect_points == 0:
                continue

            if is_box1_in_box2(boxes[b1], boxes[b2], offset=0.1):
                inst_per_point[intersect_inds] = b1
                is_determined[intersect_inds] = max_num
                ps_prob_label[intersect_inds] = 1
                box_visited[b1] = 1
                break

            if is_box1_in_box2(boxes[b2], boxes[b1], offset=0.1):
                inst_per_point[intersect_inds] = b2
                is_determined[intersect_inds] = max_num
                ps_prob_label[intersect_inds] = 1
                box_visited[b2] = 1
                continue

            if b1_ious[b2] >= 0.6: continue

            b1_inds = torch.nonzero((inst_per_point == b1) & (n_bbs_per_spp == 1)).view(-1)
            b2_inds = torch.nonzero((inst_per_point == b2) & (n_bbs_per_spp == 1)).view(-1)

            if len(b1_inds) == 0 or len(b2_inds) == 0:
                continue
            
            # try:
            pred_probs, pred_probs_new, pred_labels, pred_mu, pred_variance = fit_gp_spp(coords_float_spp, gp_feats_spp, b1_inds, b2_inds, intersect_inds, training_iter=training_iter)
            overwrite_cond = (ps_prob_label[intersect_inds] < pred_probs_new)

            inst_per_point[intersect_inds[overwrite_cond][pred_labels[overwrite_cond] == 1]] = b2
            inst_per_point[intersect_inds[overwrite_cond][pred_labels[overwrite_cond] == 0]] = b1
            ps_prob_label[intersect_inds[overwrite_cond]] = pred_probs_new[overwrite_cond]
            ps_mu_label[intersect_inds[overwrite_cond]] = pred_mu[overwrite_cond]
            ps_variance_label[intersect_inds[overwrite_cond]] = pred_variance[overwrite_cond]

            is_determined[intersect_inds[overwrite_cond]] = len(intersect_inds)
 

        box_visited[b1] = 1

    point_inds, box_inds = torch.nonzero(bb_occupancy_spp[(n_bbs_per_spp > 1) & (is_determined == 0), :], as_tuple=True)
    min_volume, argmin_volume = torch_scatter.scatter_min(boxes_volume[box_inds], point_inds, dim=0)
    sum_volume = torch_scatter.scatter(boxes_volume[box_inds].float(), point_inds, dim=0, reduce="sum")

    corres_multibox_active = box_inds[argmin_volume]

    # inst_one_hot[corres_multibox_active, point_inds[argmin_volume]] = 1
    # inst_one_hot_prob[corres_multibox_active, point_inds[argmin_volume]] = 1.0

    inst_per_point[(n_bbs_per_spp > 1) & (is_determined == 0)] = corres_multibox_active.int() # NOTE: multibox -> assign box with the smallest volume
    ps_prob_label[(n_bbs_per_spp > 1) & (is_determined == 0)] = 1.0
    # ps_prob_label[(n_bbs_per_spp > 1) & (is_determined == 0)] = 1 - (min_volume / sum_volume).float()

    ps_semantic_label_spp = torch.ones(n_spps, dtype=torch.int, device=coords_float.device) * -100
    ps_instance_label_spp = torch.ones(n_spps, dtype=torch.int, device=coords_float.device) * -100

    ps_semantic_label_spp[inst_per_point >= 0] = boxes_cls[inst_per_point[inst_per_point >= 0].long()].int()
    ps_semantic_label_spp[inst_per_point == -1] = instance_classes

    ps_instance_label_spp[inst_per_point >= 0] = inst_per_point[inst_per_point >= 0]

    ps_instance_label_spp[ps_instance_label_spp >= n_fg_instances] = -100
    ps_semantic_label_spp[ps_instance_label_spp >= n_fg_instances] = instance_classes

    ps_semantic_label = ps_semantic_label_spp[spp].int()
    ps_instance_label = ps_instance_label_spp[spp].int()
    ps_prob_label = ps_prob_label[spp].float()

    return ps_semantic_label, ps_instance_label, ps_prob_label, ps_mu_label, ps_variance_label


def gen_pseudo_label(coords_float, spp, instance_cls, instance_box, instance_box_volume, instance_classes=18, dataset_name="scannetv2", heuristic_rule='volume'):
    n_points_ = len(coords_float)
    n_boxes = len(instance_box)

    instance_center = (instance_box[:, :3] + instance_box[:, 3:]) / 2.0


    inst_per_point = torch.ones(n_points_, dtype=torch.int, device=coords_float.device) * -100

    bb_occupancy = is_within_bb_torch(coords_float[:, None, :], instance_box[None, :, :3] - 0.005, instance_box[None, :, 3:] + 0.005) # N_points, N_box

    num_BBs_per_point = bb_occupancy.long().sum(dim=1)

    corres_box_active = torch.nonzero(bb_occupancy[(num_BBs_per_point == 1),:], as_tuple=True)[1]

    inst_per_point[(num_BBs_per_point == 1)] = corres_box_active.int() # NOTE: 1 box -> assign corres box
    inst_per_point[(num_BBs_per_point == 0)] = -1 # NOTE: nobox -> background -1
    
    point_inds, box_inds = torch.nonzero(bb_occupancy[(num_BBs_per_point > 1),:], as_tuple=True)

    if heuristic_rule=='volume':
        # NOTE process points within multi boxes
        
        volume_box = instance_box_volume[box_inds]
        min_volume, argmin_volume = torch_scatter.scatter_min(volume_box, point_inds, dim=0)
        assert len(min_volume) == (num_BBs_per_point > 1).sum()
        corres_multibox_active = box_inds[argmin_volume]
        inst_per_point[(num_BBs_per_point > 1)] = corres_multibox_active.int() # NOTE: multibox -> assign box with the smallest volume
    elif heuristic_rule=='dist':
        dist_to_center = torch.sum((coords_float[point_inds] - instance_center[box_inds])**2, dim=-1)
        min_dist, argmin_dist = torch_scatter.scatter_min(dist_to_center, point_inds, dim=0)
        assert len(min_dist) == (num_BBs_per_point > 1).sum()
        corres_multibox_active = box_inds[argmin_dist]
        inst_per_point[(num_BBs_per_point > 1)] = corres_multibox_active.int() # NOTE: multibox -> assign box with the smallest volume
    elif heuristic_rule=='none':
        inst_per_point[(num_BBs_per_point > 1)] = -2 # NOTE: multibox -> assign box with the smallest volume
    else:
        raise Exception


    # # NOTE superpoint alignment

    if dataset_name == 'scannetv2':
        _, spp = torch.unique(spp, return_inverse=True)
        bb_occupancy_spp = torch_scatter.scatter(bb_occupancy.permute(1,0).float(), spp[None, :].expand(n_boxes, n_points_), dim=1, reduce="mean") # n_label, n_spp
        # bb_occupancy_spp = (bb_occupancy_spp == 1)
        bb_occupancy_spp = (bb_occupancy_spp >= 0.7)
        # if dataset_name == 'scannetv2':
        #     bb_occupancy_spp = (bb_occupancy_spp >= 0.999)
        # else:
        #     bb_occupancy_spp = (bb_occupancy_spp >= 0.1)

        inst_per_point = torch.where(inst_per_point >= 0, inst_per_point+1, 0)
        inst_per_point = spp_align_label(spp, inst_per_point, n_classes=bb_occupancy_spp.shape[0]+1, bb_occupancy_spp=bb_occupancy_spp)
        inst_per_point = torch.where(inst_per_point > 0, inst_per_point-1, -1).int()

    ps_semantic_label = torch.ones(n_points_, dtype=torch.int, device=coords_float.device) * -100
    ps_instance_label = torch.ones(n_points_, dtype=torch.int, device=coords_float.device) * -100

    ps_semantic_label[inst_per_point >= 0] = instance_cls[inst_per_point[inst_per_point >= 0].long()].int()
    ps_semantic_label[inst_per_point == -1] = instance_classes

    ps_instance_label[inst_per_point >= 0] = inst_per_point[inst_per_point >= 0]


    ps_semantic_label[inst_per_point == -2] = -100
    ps_instance_label[inst_per_point == -2] = -100
    

    return ps_semantic_label, ps_instance_label