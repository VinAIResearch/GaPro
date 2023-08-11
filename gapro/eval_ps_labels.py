import os

import torch
import torch.nn.functional as F
from tqdm import tqdm


CLASSES = (
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

TRUE_LABELS_FOLDER = "dataset/scannetv2/train"
# PS_LABLES_FOLDER = "/home/ubuntu/3dis_ws/BoxSup3DIS/dataset/scannetv2/gpfeats_pseudo_labels_variance_align_spp_pool"
PS_LABLES_FOLDER = "dataset/scannetv2/pseudo_labels"
PROB_LABLES_FOLDER = "dataset/scannetv2/gpfeats_pseudo_labels_spp_pool_debug"


def cal_iou(label, ps_label):
    intersection = torch.mm(label, ps_label.t())  # (nProposal, nProposal), float, cuda
    label_pointnum = label.sum(1).unsqueeze(-1).repeat(1, ps_label.shape[0])  # (nProposal), float, cuda
    ps_label_pointnum = ps_label.sum(1).unsqueeze(0).repeat(label.shape[0], 1)  # (nProposal), float, cuda

    ious = intersection / (label_pointnum + ps_label_pointnum - intersection + 1e-4)

    return ious  # n_label, n_ps_label


def get_miou_scene2(
    semantic_label, instance_label, ps_semantic_label, ps_instance_label, ps_instance_label_onehot, repeat=1
):
    n_inst = instance_label.max() + 1
    instance_label_cls = torch.ones(n_inst).cuda()
    for i in range(0, n_inst):
        idx_ = torch.nonzero(instance_label == i).view(-1)
        if len(idx_) == 0:
            instance_label_cls[i] = -1
        else:
            instance_label_cls[i] = semantic_label[idx_[0]]

    n_ps_inst = ps_instance_label.max() + 1
    ps_instance_label_cls = torch.ones(n_ps_inst).cuda()
    for i in range(0, n_ps_inst):
        idx_ = torch.nonzero(ps_instance_label == i).view(-1)
        if len(idx_) == 0:
            ps_instance_label_cls[i] = -1
        else:
            ps_instance_label_cls[i] = ps_semantic_label[idx_[0]]

    if repeat > 1:
        ps_instance_label_cls = ps_instance_label_cls.repeat(repeat)

    # print(ps_instance_label_cls1 - ps_instance_label_cls)

    # max_inst = instance_label.max().long().item()
    instance_label = torch.where(instance_label < 0, 0, instance_label + 1)
    instance_label_onehot = F.one_hot(instance_label.long(), num_classes=n_inst + 1).permute(1, 0)[
        1:, :
    ]  # remove < 0 label

    # ps_instance_label = ps_instance_label[valid_inds]
    # ps_instance_label = torch.where(ps_instance_label < 0, 0, ps_instance_label + 1)
    # ps_instance_label_onehot = F.one_hot(ps_instance_label.long(), num_classes=n_ps_inst+1).permute(1,0)[1:, :] # remove < 0 label

    # print(instance_label_onehot.shape, ps_instance_label_onehot.shape)
    ious = cal_iou(instance_label_onehot.float(), ps_instance_label_onehot.float())

    cls_cond = (instance_label_cls[:, None] == ps_instance_label_cls[None, :]).float()
    ious = ious * cls_cond

    max_ious, _ = torch.max(ious, dim=1)

    max_ious = max_ious[instance_label_cls >= 0]
    # max_ious = torch.diagonal(ious)
    # print(max_ious)

    # for i in range(len(max_ious)):
    #     if instance_label_cls[i] < 0: continue
    #     print(f'Sem: {CLASSES[int(instance_label_cls[i].item())]}, IoU: {max_ious[i]}')

    return max_ious


def get_miou_scene(semantic_label, instance_label, ps_semantic_label, ps_instance_label):
    n_inst = instance_label.max() + 1
    instance_label_cls = torch.ones(n_inst).cuda()
    for i in range(0, n_inst):
        idx_ = torch.nonzero(instance_label == i).view(-1)
        if len(idx_) == 0:
            instance_label_cls[i] = -1
        else:
            instance_label_cls[i] = semantic_label[idx_[0]]

    n_ps_inst = ps_instance_label.max() + 1
    ps_instance_label_cls = torch.ones(n_ps_inst).cuda()
    for i in range(0, n_ps_inst):
        idx_ = torch.nonzero(ps_instance_label == i).view(-1)
        if len(idx_) == 0:
            ps_instance_label_cls[i] = -1
        else:
            ps_instance_label_cls[i] = ps_semantic_label[idx_[0]]

    # max_inst = instance_label.max().long().item()
    instance_label = torch.where(instance_label < 0, 0, instance_label + 1)
    instance_label_onehot = F.one_hot(instance_label.long(), num_classes=n_inst + 1).permute(1, 0)[
        1:, :
    ]  # remove < 0 label

    # ps_instance_label = ps_instance_label[valid_inds]
    ps_instance_label = torch.where(ps_instance_label < 0, 0, ps_instance_label + 1)
    ps_instance_label_onehot = F.one_hot(ps_instance_label.long(), num_classes=n_ps_inst + 1).permute(1, 0)[
        1:, :
    ]  # remove < 0 label

    # print(instance_label_onehot.shape, ps_instance_label_onehot.shape)
    ious = cal_iou(instance_label_onehot.float(), ps_instance_label_onehot.float())

    cls_cond = (instance_label_cls[:, None] == ps_instance_label_cls[None, :]).float()
    ious = ious * cls_cond

    max_ious, _ = torch.max(ious, dim=1)

    max_ious = max_ious[instance_label_cls >= 0]
    # max_ious = torch.diagonal(ious)
    # print(max_ious)

    # for i in range(len(max_ious)):
    #     if instance_label_cls[i] < 0: continue
    #     print(f'Sem: {CLASSES[int(instance_label_cls[i].item())]}, IoU: {max_ious[i]}')

    return max_ious
    # except:
    #     return None


def get_scene_sem_conf(semantic_label, ps_semantic_label, num_classes=19):
    pos_inds = semantic_label != -100

    semantic_label = semantic_label[pos_inds]
    ps_semantic_label = ps_semantic_label[pos_inds]

    # ps_semantic_label[ps_semantic_label == -100] = semantic_label[ps_semantic_label == -100]
    ps_semantic_label[ps_semantic_label == -100] = torch.where(
        semantic_label[ps_semantic_label == -100] < 18,
        semantic_label[ps_semantic_label == -100] + 1,
        semantic_label[ps_semantic_label == -100] - 1,
    )

    x = ps_semantic_label + num_classes * semantic_label

    # breakpoint()
    bincount_2d = torch.bincount(x.long(), minlength=num_classes**2)
    # assert bincount_2d.size == num_classes**2
    conf = bincount_2d.reshape((num_classes, num_classes))

    return conf


def main():
    scenes = os.listdir(TRUE_LABELS_FOLDER)
    scenes = [s[:12] for s in scenes]
    scenes = sorted(scenes)[::10]

    # scenes = ['scene0169_00']

    ious_arr = []
    sem_acc_arr = []

    num_classes = 19
    conf_metric = torch.zeros((num_classes, num_classes), dtype=torch.long).cuda()
    # conf_metric.fill(0)

    for scene in tqdm(scenes):
        _, _, semantic_label, instance_label = torch.load(
            os.path.join(TRUE_LABELS_FOLDER, f"{scene}_inst_nostuff.pth")
        )
        semantic_label = torch.from_numpy(semantic_label).cuda().int()
        instance_label = torch.from_numpy(instance_label).cuda().int()

        # semantic_label_ori = semantic_label.clone()
        semantic_label[semantic_label != -100] -= 2
        semantic_label[(semantic_label == -1) | (semantic_label == -2)] = 18
        valid_inds = (semantic_label != -100) & (semantic_label != 18)

        if not os.path.exists(os.path.join(PS_LABLES_FOLDER, f"{scene}.pth")):
            continue

        # ps_semantic_label, ps_instance_label, ps_uncertainty_label = torch.load(os.path.join(PS_LABLES_FOLDER, f'{scene}.pth'))
        # ps_semantic_label = torch.from_numpy(ps_semantic_label).cuda().int()
        # ps_instance_label = torch.from_numpy(ps_instance_label).cuda().int()
        # ps_uncertainty_label = torch.from_numpy(ps_uncertainty_label).cuda().float()

        ps_semantic_label, ps_instance_label = torch.load(os.path.join(PS_LABLES_FOLDER, f"{scene}.pth"))
        ps_semantic_label = torch.from_numpy(ps_semantic_label).cuda().int()
        ps_instance_label = torch.from_numpy(ps_instance_label).cuda().int()

        # _, _, ps_prob_label = torch.load(os.path.join(PROB_LABLES_FOLDER, f'{scene}.pth'))

        # certain_cond = (ps_uncertainty_label == 0)
        # semantic_label = semantic_label[certain_cond]
        # instance_label = instance_label[certain_cond]
        # ps_semantic_label = ps_semantic_label[certain_cond]
        # ps_instance_label = ps_instance_label[certain_cond]

        # print(certain_cond.sum()/certain_cond.numel())

        # correct_inds = (ps_uncertainty_label > 0.1)
        # ps_semantic_label[correct_inds] = semantic_label[correct_inds]
        # ps_instance_label[correct_inds] = instance_label[correct_inds]

        # print('ratio', correct_inds.sum() / torch.numel(correct_inds))
        # # # if correct_inds.sum() > 0:
        # # #     breakpoint()

        # # # continue
        # ps_semantic_label[correct_inds] = semantic_label[correct_inds]
        # ps_instance_label[correct_inds] = instance_label[correct_inds]
        # # ps_semantic_label[ps_instance_label==-100] = -100

        ious = get_miou_scene(semantic_label, instance_label, ps_semantic_label, ps_instance_label)
        ious_arr.append(ious)

        conf = get_scene_sem_conf(semantic_label, ps_semantic_label)
        conf_metric += conf

        # if valid_inds.sum() > 0:
        # sem_acc_arr.append((ps_semantic_label[valid_inds] == semantic_label[valid_inds]).sum()/len(semantic_label[valid_inds]))

    ious_arr = torch.cat(ious_arr, dim=0)
    print("mean inst iou", torch.mean(ious_arr))

    true_positive = torch.diag(conf_metric)
    false_positive = torch.sum(conf_metric, 0) - true_positive
    false_negative = torch.sum(conf_metric, 1) - true_positive

    # Just in case we get a division by 0, ignore/hide the error
    # with np.errstate(divide="ignore", invalid="ignore"):
    iou = true_positive / (true_positive + false_positive + false_negative)
    iou = iou * 100
    miou = torch.nanmean(iou)
    print("sem iou", iou)
    print("sem miou", miou)

    # print('mean sem acc', torch.mean(torch.tensor(sem_acc_arr)))


if __name__ == "__main__":
    main()
