import argparse
import datetime
import os
import os.path as osp
import shutil
import time

import gorilla
import torch
from spformer.dataset import build_dataloader, build_dataset
from spformer.evaluation import ScanNetEval
from spformer.model import SPFormer
from spformer.utils import AverageMeter, get_root_logger
from tensorboardX import SummaryWriter
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("SPFormer")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--resume", type=str, help="path to resume from")
    parser.add_argument("--work_dir", type=str, help="working directory")
    parser.add_argument("--skip_validate", action="store_true", help="skip validation")
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()
    return args


def train(epoch, model, dataloader, optimizer, lr_scheduler, cfg, logger, writer, scaler):
    model.train()
    iter_time = AverageMeter()
    data_time = AverageMeter()
    meter_dict = {}
    end = time.time()

    for i, batch in enumerate(dataloader, start=1):
        data_time.update(time.time() - end)
        loss, log_vars = model(batch, mode="loss")

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch, mode="loss")

        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # backward
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # time and print
        remain_iter = len(dataloader) * (cfg.train.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]["lr"]
        if i % 10 == 0:
            log_str = f"Epoch [{epoch}/{cfg.train.epochs}][{i}/{len(dataloader)}]  "
            log_str += f"lr: {lr:.2g}, eta: {remain_time}, "
            log_str += f"data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}"
            for k, v in meter_dict.items():
                log_str += f", {k}: {v.val:.4f}"
            logger.info(log_str)

    # update lr
    lr_scheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    # log and save
    writer.add_scalar("train/learning_rate", lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f"train/{k}", v.avg, epoch)
    save_file = osp.join(cfg.work_dir, "lastest.pth")
    meta = dict(epoch=epoch)
    gorilla.save_checkpoint(model, save_file, optimizer, lr_scheduler, meta)


@torch.no_grad()
def eval(epoch, model, dataloader, cfg, logger, writer):
    logger.info("Validation")
    pred_insts, gt_insts = [], []
    sem_labels, ins_labels = [], []
    progress_bar = tqdm(total=len(dataloader))
    val_dataset = dataloader.dataset

    scannet_eval = ScanNetEval(val_dataset.CLASSES, dataset_name=cfg.data.train.type)

    model.eval()
    for b, batch in enumerate(dataloader):

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            result = model(batch, mode="predict")
        # gt_insts.append(result['gt_instances'])

        _, _, semantic_label, instance_label = val_dataset.load(val_dataset.filenames[b])
        if cfg.data.train.type == "scannetv2":
            semantic_label[semantic_label != -100] -= 2
            semantic_label[(semantic_label == -1) | (semantic_label == -2)] = -100
            # semantic_label = np.where(semantic_label < 2

        # print(val_dataset.filenames[b], instance_label.shape, batch['superpoints'].shape)

        pred_insts.append(result["pred_instances"])
        sem_labels.append(semantic_label)
        ins_labels.append(instance_label)

        progress_bar.update()
    progress_bar.close()

    # evaluate
    logger.info("Evaluate instance segmentation")
    eval_res = scannet_eval.evaluate(pred_insts, sem_labels, ins_labels)

    writer.add_scalar("val/AP", eval_res["all_ap"], epoch)
    writer.add_scalar("val/AP_50", eval_res["all_ap_50%"], epoch)
    writer.add_scalar("val/AP_25", eval_res["all_ap_25%"], epoch)
    logger.info(
        "AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}".format(
            eval_res["all_ap"], eval_res["all_ap_50%"], eval_res["all_ap_25%"]
        )
    )

    # save
    save_file = osp.join(cfg.work_dir, f"epoch_{epoch:04d}.pth")
    gorilla.save_checkpoint(model, save_file)


def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0], args.exp_name)
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(f"config: {args.config}")
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # seed
    gorilla.set_random_seed(cfg.train.seed)

    # model
    model = SPFormer(**cfg.model).cuda()
    count_parameters = gorilla.parameter_count(model)[""]
    logger.info(f"Parameters: {count_parameters / 1e6:.2f}M")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # optimizer and scheduler
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # pretrain or resume
    start_epoch = 1
    if args.resume:
        logger.info(f"Resume from {args.resume}")
        meta = gorilla.resume(model, args.resume, optimizer, lr_scheduler)
        start_epoch = meta["epoch"]
    elif cfg.train.pretrain:
        logger.info(f"Load pretrain from {cfg.train.pretrain}")
        # gorilla.load_checkpoint(model, cfg.train.pretrain, strict=False)

        checkpoint = torch.load(cfg.train.pretrain)

        # get model state_dict from checkpoint
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "net" in checkpoint:
            state_dict = checkpoint["net"]
        else:
            state_dict = checkpoint

        # breakpoint()
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        gorilla.load_state_dict(model, state_dict, strict=False)

    # train and val dataset
    train_dataset = build_dataset(cfg.data.train, logger)
    train_loader = build_dataloader(train_dataset, **cfg.dataloader.train)
    if not args.skip_validate:
        val_dataset = build_dataset(cfg.data.val, logger)
        val_loader = build_dataloader(val_dataset, training=False, **cfg.dataloader.val)

    # eval(16, model, val_loader, cfg, logger, writer)
    # train and val
    logger.info("Training")
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train(epoch, model, train_loader, optimizer, lr_scheduler, cfg, logger, writer, scaler)
        if not args.skip_validate and (epoch % cfg.train.interval == 0):
            eval(epoch, model, val_loader, cfg, logger, writer)
        writer.flush()


if __name__ == "__main__":
    main()
