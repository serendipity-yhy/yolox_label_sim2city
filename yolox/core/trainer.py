#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
import math
from copy import deepcopy

import torchvision.transforms
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
    postprocess
)


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        # ema_update_modelT
        self.updates = 0
        self.decay_num = 0.9999
        self.iter_num = 0
        # self.decay = lambda x: self.decay_num * (1 - math.exp(-x / 2000))


        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            # self.after_iter()

# 修改
    def train_one_iter(self):
        iter_start_time = time.time()
        for i in range(2):
            if i == 0:
                inps, inps_aug, targets = self.prefetcher_S.next()
                inps = inps.to(self.data_type)
                inps_aug = inps_aug.to(self.data_type)
                targets = targets.to(self.data_type)
            else:
                inps, inps_aug, targets = self.prefetcher_T.next()
                inps = inps.to(self.data_type)
                inps_aug = inps_aug.to(self.data_type)
                targets = targets.to(self.data_type)

            data_end_time = time.time()

            if i == 1:
                with torch.no_grad():
                    self.ema_model_T.ema.eval()
                    outputs_t = self.ema_model_T.ema(inps)
                    p_targets = postprocess(outputs_t, num_classes=8, conf_thre=0.4, nms_thre=0.65)

                    # pseudo label
                    targets_p = torch.zeros_like(torch.randn(len(p_targets), 120, 5))
                    targets_p.requires_grad = False
                    for idx in range(len(p_targets)):
                        if p_targets[idx] == None:
                            continue
                        if p_targets[idx].shape[0] > targets_p.shape[1]:
                            p_targets[idx] = p_targets[idx][0:targets_p.shape[1], :]

                        # xyxy -> xywh
                        targets_p[idx][0:p_targets[idx].shape[0], 1] = (p_targets[idx][:, 2] + p_targets[idx][:, 0]) / 2.0
                        targets_p[idx][0:p_targets[idx].shape[0], 2] = (p_targets[idx][:, 3] + p_targets[idx][:, 1]) / 2.0
                        targets_p[idx][0:p_targets[idx].shape[0], 3] = (p_targets[idx][:, 2] - p_targets[idx][:, 0])
                        targets_p[idx][0:p_targets[idx].shape[0], 4] = (p_targets[idx][:, 3] - p_targets[idx][:, 1])
                        targets_p[idx][0:p_targets[idx].shape[0], 0] = p_targets[idx][:, -1]

                    targets_p = targets_p.detach()
                    targets_p = targets_p.to(self.data_type)
                    targets_p = targets_p.cuda()


            # inps_aug, targets_p = self.exp.preprocess(inps_aug, targets_p, self.input_size)
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                self.model_S.train()
                if i == 1:
                    inps_aug, targets_p = self.exp.preprocess(inps_aug, targets_p, self.input_size)
                    outputs = self.model_S(inps_aug, targets_p)
                    outputs["feature"] = None

                    self.iter_num = self.iter_num + 1
                    outputs["total_loss"] = outputs["total_loss"] * (1 - math.exp(-self.iter_num / 500.))

                    loss = outputs["total_loss"]
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                if i == 0:
                    inps, targets_p = self.exp.preprocess(inps, targets, self.input_size)
                    outputs = self.model_S(inps, targets_p)
                    outputs["feature"] = None
                    loss = outputs["total_loss"]
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            if self.use_model_ema:
                self.ema_model_S.update(self.model_S)
                self.ema_model_T.update(self.ema_model_S.ema)

            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            iter_end_time = time.time()
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                data_time=data_end_time - iter_start_time,
                lr=lr,
                **outputs,
            )

            self.after_iter(i)

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model_S = self.exp.get_model()
        model_T = self.exp.get_model()

        logger.info(
            "Model Summary: {}".format(get_model_info(model_T, self.exp.test_size))
        )
        model_S.to(self.device)
        model_T.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model_S = self.resume_train(model_S)
        model_T = self.resume_train(model_T)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader_s = self.exp.get_data_loader_s(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        self.train_loader_t = self.exp.get_data_loader_t(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher_S = DataPrefetcher(self.train_loader_s)
        self.prefetcher_T = DataPrefetcher(self.train_loader_t)
        # max_iter means iters per epoch
        self.max_iter = int(len(self.train_loader_t))

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model_S = DDP(model_S, device_ids=[self.local_rank], broadcast_buffers=False)
            model_T = DDP(model_T, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model_S = ModelEMA(model_S, 0.9998)
            self.ema_model_S.updates = self.max_iter * self.start_epoch

            self.ema_model_T = ModelEMA(model_S, 0.9998)
            self.ema_model_T.updates = self.max_iter * self.start_epoch

        self.model_S = model_S
        self.model_T = model_T

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                wandb_params = dict()
                for k, v in zip(self.args.opts[0::2], self.args.opts[1::2]):
                    if k.startswith("wandb-"):
                        wandb_params.update({k.lstrip("wandb-"): v})
                self.wandb_logger = WandbLogger(config=vars(self.exp), **wandb_params)
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model_S))




    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader_s.close_mosaic()
            self.train_loader_t.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model_S.module.head.use_l1 = True
                self.model_T.module.head.use_l1 = True
            else:
                self.model_S.head.use_l1 = True
                self.model_T.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model_S)
            all_reduce_norm(self.model_T)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self, target=0):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        domain_str = "_t" if target else "_s"
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter, domain_str
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "wandb":
                    self.wandb_logger.log_metrics({k: v.latest for k, v in loss_meter.items()})
                    self.wandb_logger.log_metrics({"lr": self.meter["lr"].latest})

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader_t, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        '''
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
        '''
        evalmodel = self.ema_model_T.ema
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            ap50_95, ap50, summary = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "epoch": self.epoch + 1,
                })
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, keep_old_best=True, old_best_tag=f"epoch{self.epoch+1}_ap50_{ap50}")
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, keep_old_best=False, old_best_tag="prior"):
        if self.rank == 0:
            # save_model = self.ema_model_T.ema if self.use_model_ema else self.model_S
            save_model = self.ema_model_T.ema
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
                keep_old_best,
                old_best_tag=old_best_tag
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(self.file_name, ckpt_name, update_best_ckpt)
