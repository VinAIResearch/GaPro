import os

import numpy as np
import torch
from bs3dis.evaluation import ScanNetEval
from bs3dis.util import rle_encode_gpu_batch
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

TRUE_LABELS_FOLDER = "./dataset/scannetv2/train"
# PS_LABLES_FOLDER = "./dataset/scannetv2/linear_regression_ps_nospp"
PS_LABLES_FOLDER = "dataset/scannetv2/gpfeats_pseudo_labels_variance_align_spp_pool"
# PS_LABLES_FOLDER = "dataset/scannetv2/gp_regression_deep_truekl"
# PS_LABLES_FOLDER = "dataset/scannetv2/linear_regression_ps"

if __name__ == "__main__":
    scannet_eval = ScanNetEval(CLASSES, dataset_name="scannetv2", start_iou=0.5, step_iou=0.05)

    scenes = os.listdir(TRUE_LABELS_FOLDER)
    # scenes = sorted(scenes)[::4]
    scenes = [s[:12] for s in scenes]

    # scenes = ['scene0169_00']

    num_classes = 19

    all_pred_insts, all_sem_labels, all_ins_labels = [], [], []
    for scene in tqdm(scenes):

        if not os.path.exists(os.path.join(PS_LABLES_FOLDER, f"{scene}.pth")):
            continue

        _, _, semantic_label, instance_label = torch.load(
            os.path.join(TRUE_LABELS_FOLDER, f"{scene}_inst_nostuff.pth")
        )

        # semantic_label_ori = semantic_label.clone()
        semantic_label[semantic_label != -100] -= 2
        semantic_label[(semantic_label == -1) | (semantic_label == -2)] = 18
        # valid_inds = (semantic_label != -100) & (semantic_label != 18)

        all_sem_labels.append(semantic_label)
        all_ins_labels.append(instance_label)

        # a = torch.load(os.path.join(PS_LABLES_FOLDER, f'{scene}.pth'))
        # breakpoint()
        ps_semantic_label, ps_instance_label, ps_uncertainty_label = torch.load(
            os.path.join(PS_LABLES_FOLDER, f"{scene}.pth")
        )

        # FIXME correct ps label:
        # k = (ps_uncertainty_label > 0.01).sum() // 4
        # k = ps_uncertainty_label.shape[0] // 25
        # topk_inds = np.argpartition(ps_uncertainty_label, -k)[-k:]
        # topk_cond = np.zeros(ps_uncertainty_label.shape[0], dtype=np.bool)
        # topk_cond[topk_inds] = 1
        # cond_ = (ps_uncertainty_label >= 0.1) & topk_cond

        k = ps_uncertainty_label.shape[0] // 25

        inds = np.nonzero((ps_uncertainty_label < 0.05))[0]
        # breakpoint()
        # print(ps_uncertainty_label[ps_uncertainty_label < 0.05].shape, len(inds))
        # topk_inds = np.argpartition(ps_uncertainty_label[inds], -k)[-k:]

        # topk_inds = inds[topk_inds.astype(np.long)]

        topk_inds = np.random.choice(inds, size=min(k, len(inds)), replace=False)
        topk_cond = np.zeros(ps_uncertainty_label.shape[0], dtype=np.bool)
        topk_cond[topk_inds] = 1
        cond_ = (ps_uncertainty_label < 0.05) & topk_cond

        print((ps_uncertainty_label < 0.05).sum(), (ps_uncertainty_label > 0.1).sum())
        ps_semantic_label[cond_] = semantic_label[cond_]
        ps_instance_label[cond_] = instance_label[cond_]

        # print(ps_uncertainty_label.min(), ps_uncertainty_label.max())
        # ps_semantic_label, ps_instance_label = torch.load(os.path.join(PS_LABLES_FOLDER, f'{scene}.pth'))

        instances = []
        unique_inst_inds = np.unique(ps_instance_label)

        n_point = ps_instance_label.shape[0]

        for unique_idx in unique_inst_inds:
            if unique_idx == -100:
                continue

            pred = {}
            pred["scan_id"] = scene

            ind_ = np.nonzero(ps_instance_label == unique_idx)[0]
            sem_ = ps_semantic_label[ind_[0]]

            # breakpoint()

            mask_ = np.zeros((n_point), dtype=np.bool)
            mask_[ind_] = 1

            encoded_mask_ = rle_encode_gpu_batch(torch.from_numpy(mask_[None]))[0]

            # breakpoint()
            pred["conf"] = 1.0
            pred["label_id"] = sem_ + 1
            pred["pred_mask"] = encoded_mask_

            instances.append(pred)
        all_pred_insts.append(instances)

    eval_res = scannet_eval.evaluate(all_pred_insts, all_sem_labels, all_ins_labels, threshold=0.9)
    del all_pred_insts, all_sem_labels, all_ins_labels

    print(
        "AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}".format(
            eval_res["all_ap"], eval_res["all_ap_50%"], eval_res["all_ap_25%"]
        )
    )
