import multiprocessing as mp
import os
import os.path as osp

import numpy as np

from .mask_encoder import rle_decode


def save_single_instance(root, scan_id, insts, benchmark_sem_id):
    f = open(osp.join(root, f"{scan_id}.txt"), "w")
    os.makedirs(osp.join(root, "predicted_masks"), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst["scan_id"]

        # NOTE process to map label id to benchmark
        label_id = inst["label_id"]  # 1-> 18
        label_id = label_id + 1  # 2-> 19 , 0,1: background
        label_id = benchmark_sem_id[label_id]

        # NOTE box
        conf = inst["conf"]

        f.write(f"predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n")
        # f.write(f"predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f} " + box_string + "\n")
        mask_path = osp.join(root, "predicted_masks", f"{scan_id}_{i:03d}.txt")
        mask = rle_decode(inst["pred_mask"])
        np.savetxt(mask_path, mask, fmt="%d")
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts, benchmark_sem_id):
    # root = osp.join(root, name)
    # os.makedirs(root, exist_ok=True)
    # roots = [root] * len(scan_ids)
    # nyu_ids = [nyu_id] * len(scan_ids)
    # pool = mp.Pool()
    # pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, nyu_ids))
    # pool.close()
    # pool.join()

    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    benchmark_sem_ids = [benchmark_sem_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, benchmark_sem_ids))
    pool.close()
    pool.join()


def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt="%d")


def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f"{i}.txt") for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()
