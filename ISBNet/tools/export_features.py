import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
from functools import partial

import numpy as np
import torch
import yaml
from isbnet.data import build_dataloader, build_dataset
from isbnet.evaluation import PointWiseEval, S3DISEval, ScanNetEval
from isbnet.model import ISBNet
from isbnet.util import get_root_logger, init_dist, load_checkpoint, rle_decode
from munch import Munch
from torch.nn.parallel import DistributedDataParallel


def get_args():
    parser = argparse.ArgumentParser("ISBNet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--save_deepfeatures_path", type=str, help="path to save deep features")
    
    args = parser.parse_args()
    return args


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f"{i}.npy") for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


def save_single_instance(root, scan_id, insts, benchmark_sem_id):
    f = open(osp.join(root, f"{scan_id}.txt"), "w")
    os.makedirs(osp.join(root, "predicted_masks"), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst["scan_id"]

        # NOTE process to map label id to benchmark
        label_id = inst["label_id"]  # 1-> 18
        label_id = label_id + 1  # 2-> 19 , 0,1: background
        label_id = benchmark_sem_id[label_id]

        conf = inst["conf"]

        f.write(f"predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n")
        # f.write(f"predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f} " + box_string + "\n")
        mask_path = osp.join(root, "predicted_masks", f"{scan_id}_{i:03d}.txt")
        mask = rle_decode(inst["pred_mask"])
        np.savetxt(mask_path, mask, fmt="%d")
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts, benchmark_sem_id):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    benchmark_sem_ids = [benchmark_sem_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, benchmark_sem_ids))
    pool.close()
    pool.join()


def save_gt_instances(root, name, scan_ids, gt_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f"{i}.txt") for i in scan_ids]
    pool = mp.Pool()
    map_func = partial(np.savetxt, fmt="%d")
    pool.starmap(map_func, zip(paths, gt_insts))
    pool.close()
    pool.join()


def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    model = ISBNet(**cfg.model, dataset_name=cfg.data.train.type, save_deepfeatures_path=args.save_deepfeatures_path).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)

    if not os.path.exists(args.save_deepfeatures_path):
        os.makedirs(args.save_deepfeatures_path)

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):

            # NOTE avoid OOM during eval s3dis with full resolution
            if cfg.data.test.type == "s3dis":
                torch.cuda.empty_cache()

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                model(batch)

            if i % 10 == 0:
                logger.info(f"Infer scene {i+1}/{len(dataset)}")

    # save output
if __name__ == "__main__":
    main()
