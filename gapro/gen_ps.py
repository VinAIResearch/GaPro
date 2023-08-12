import os
import os.path as osp
from glob import glob

import argparse
import numpy as np
import torch
from eval_ps_labels import get_miou_scene
from gen_ps_utils import gen_pseudo_label_gaussian_process, getInstanceInfo
from scannet_planes import get_wall_boxes
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GaPro_GenPS")

    parser.add_argument("--save_folder", type=str, default="dataset/scannetv2/gaussian_process_kl_pseudo_labels")
    parser.add_argument("--use_deepfeat", action="store_true")
    parser.add_argument("--deepfeat_folder", type=str, default="dataset/scannetv2/pretrain_maskfeats2")
    parser.add_argument("--eval_pslabel", action="store_true")
    args = parser.parse_args()

    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filenames_train = glob(osp.join("dataset/scannetv2", "train", "*" + "_inst_nostuff.pth"))
    filenames_val = glob(osp.join("dataset/scannetv2", "val", "*" + "_inst_nostuff.pth"))

    # filenames = filenames_val + filenames_train
    filenames = filenames_train
    filenames = sorted(filenames)

    ious_arr = []
    ratios = []
    for filename in tqdm(filenames):
        scan_name = filename.split("/")[-1][:12]

        save_path = os.path.join(save_folder, scan_name + ".pth")
        if os.path.exists(save_path):
            continue

        spp_filename = "dataset/scannetv2/superpoints/" + scan_name + ".pth"

        xyz, rgb, semantic_label, instance_label = torch.load(filename)
        spp = torch.load(spp_filename)

        if args.use_deepfeat:
            feats_filename = os.path.join(args.deepfeat_folder, scan_name + ".pth")
            # print(feats_filename)
            # if not os.path.exists(feats_filename):
            #     raise Exception()
            mask_feats = torch.load(feats_filename)
        else:
            mask_feats = np.concatenate([xyz, rgb], axis=-1)

        # NOTE Align 3D point cloud:
        meta_file = "dataset/scannetv2/scans_transform/" + os.path.join(scan_name, scan_name + ".txt")
        lines = open(meta_file).readlines()
        for line in lines:
            if "axisAlignment" in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

        pts = np.ones((xyz.shape[0], 4))
        pts[:, 0:3] = xyz[:, 0:3]
        pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
        xyz = pts[:, :3]

        # NOTE Get 3D axis-aligned bounding boxes
        instance_num, instance_cls, instance_box, instance_box_volume, corners_label = getInstanceInfo(
            xyz, instance_label=instance_label, semantic_label=semantic_label
        )

        # NOTE Get wall box:
        wall_cls, wall_box, wall_volume = get_wall_boxes(scan_name)

        instance_cls = torch.from_numpy(instance_cls).cuda().long()
        instance_box = torch.from_numpy(instance_box).cuda().float()
        instance_box_volume = torch.from_numpy(instance_box_volume).cuda().float()
        xyz = torch.from_numpy(xyz).cuda()
        rgb = torch.from_numpy(rgb).cuda()
        spp = torch.from_numpy(spp).cuda()
        mask_feats = torch.from_numpy(mask_feats).float().cuda()

        if len(wall_box) > 0:
            wall_box = torch.from_numpy(wall_box).float().cuda()
            wall_volume = torch.from_numpy(wall_volume).float().cuda()

        (
            ps_semantic_label,
            ps_instance_label,
            ps_prob_label,
            ps_mu_label,
            ps_variance_label,
        ) = gen_pseudo_label_gaussian_process(
            xyz,
            mask_feats,
            spp,
            instance_cls,
            instance_box,
            instance_box_volume,
            wall_box,
            wall_volume,
            instance_classes=18,
            dataset_name="scannetv2",
            ground_h=0.1,
            training_iter=50,
            thresh_spp_occu=0.999,
        )
        # ps_semantic_label, ps_instance_label = gen_pseudo_label_box2mask(xyz, spp, \
        #                                                                             instance_cls, instance_box, instance_box_volume, \
        #                                                                             instance_classes=18, dataset_name="scannetv2")

        if args.eval_pslabel:
            semantic_label = torch.from_numpy(semantic_label).cuda().int()
            instance_label = torch.from_numpy(instance_label).cuda().int()
            semantic_label_ori = semantic_label.clone()
            semantic_label[semantic_label != -100] -= 2
            semantic_label[(semantic_label == -1) | (semantic_label == -2)] = 18
            ious = get_miou_scene(semantic_label.long(), instance_label.long(), ps_semantic_label.long(), ps_instance_label.long())
            print('miou', ious)
            ious_arr.append(ious)

        ps_semantic_label = ps_semantic_label.int().cpu().numpy()
        ps_instance_label = ps_instance_label.int().cpu().numpy()
        ps_prob_label = ps_prob_label.cpu().numpy()
        ps_mu_label = ps_mu_label.cpu().numpy()
        ps_variance_label = ps_variance_label.cpu().numpy()

        torch.save((ps_semantic_label, ps_instance_label, ps_prob_label, ps_mu_label, ps_variance_label), save_path)
    if args.eval_pslabel:
        ious_arr = torch.cat(ious_arr, dim=0)
        print('Mean instance iou of pseudo labels', torch.mean(ious_arr).item())

    print("Finish")
