import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import torch

from model import compute_metrics

# Seed
pl.seed_everything(42, workers=True)


def build_trainer(args):
    # Get folder name
    name = f"{args.model_name}_{args.data_name}"

    # Callbacks
    enable_checkpointing = "test" not in args.mode

    if enable_checkpointing:
        callbacks = []

        callbacks += [
            ModelCheckpoint(
                filename="best", monitor=args.monitor, mode="max", save_last=True
            )
        ]

        callbacks += [
            EarlyStopping(
                monitor=args.monitor,
                mode="max",
                min_delta=1e-3,
                patience=10,
                strict=True,
            )
        ]

        if args.lr == 0:
            args.lr = None
        else:
            callbacks += [LearningRateMonitor()]

        logger = TensorBoardLogger(save_dir="checkpoints", name=name)
    else:
        callbacks = None
        logger = False

    n_gpus = torch.cuda.device_count()

    return pl.Trainer.from_argparse_args(
        args,
        accelerator="ddp" if n_gpus > 1 else None,
        check_val_every_n_epoch=1,
        gpus=n_gpus,
        plugins=DDPPlugin(find_unused_parameters=False) if n_gpus > 1 else None,
        default_root_dir="checkpoints",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        fast_dev_run=args.fast_dev_run,
        enable_checkpointing=enable_checkpointing,
    )
