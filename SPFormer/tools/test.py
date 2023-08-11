import argparse

import gorilla
import torch
from spformer.dataset import build_dataloader, build_dataset
from spformer.evaluation import ScanNetEval
from spformer.model import SPFormer
from spformer.utils import get_root_logger, save_gt_instances, save_pred_instances
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("SoftGroup")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--out", type=str, help="directory for output results")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger()

    model = SPFormer(**cfg.model).cuda()
    logger.info(f"Load state dict from {args.checkpoint}")
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    results, scan_ids, pred_insts, gt_insts = [], [], [], []
    sem_labels, ins_labels = [], []
    coords = []

    scannet_eval = ScanNetEval(dataset.CLASSES, dataset_name=cfg.data.train.type)

    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for b, batch in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                result = model(batch, mode="predict")

            results.append(result)

            if not cfg.data.test.prefix == "test":
                xyz, _, semantic_label, instance_label = dataset.load(dataset.filenames[b])

                if cfg.data.train.type == "scannetv2":
                    semantic_label[semantic_label != -100] -= 2
                    semantic_label[(semantic_label == -1) | (semantic_label == -2)] = -100

                sem_labels.append(semantic_label)
                ins_labels.append(instance_label)
                coords.append(xyz)

            progress_bar.update()
        progress_bar.close()

    for res in results:
        scan_ids.append(res["scan_id"])
        pred_insts.append(res["pred_instances"])
        gt_insts.append(res["gt_instances"])

    if not cfg.data.test.prefix == "test":
        logger.info("Evaluate instance segmentation")
        scannet_eval.evaluate(pred_insts, sem_labels, ins_labels)

        scannet_eval.evaluate_box(pred_insts, coords, sem_labels, ins_labels)

    # save output
    if args.out:
        logger.info("Save results")
        save_pred_instances(args.out, "pred_instance", scan_ids, pred_insts, dataset.BENCHMARK_SEMANTIC_IDXS)
        # nyu_id = dataset.NYU_ID
        # save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        # if not cfg.data.test.prefix == 'test':
        #     save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)


if __name__ == "__main__":
    main()
