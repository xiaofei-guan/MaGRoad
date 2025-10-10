from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from datamodule import SamRoadDataModule

from utils import load_config
from model import SAMRoad

# import wandb

# import lightning.pytorch as pl
import pytorch_lightning as pl

# from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
# from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import LearningRateMonitor
import datetime



parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--resume", default=None, help="checkpoint of the last epoch of the model"
)
parser.add_argument(
    "--precision", default="16-mixed", help="32 or 16-mixed"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--dev_run", default=False, action='store_true'
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run

    # Remove wandb initialization
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="sam_road",
    #     # track hyperparameters and run metadata
    #     config=config,
    #     # disable wandb if debugging
    #     mode='disabled' if dev_run else None,
    #     # mode='disabled',
    # )

    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high') # 设置混合精度
    pl.seed_everything(seed=3407, workers=True) # 设置随机种子 3407

    net = SAMRoad(config)
    dm = SamRoadDataModule(config, dev_run=dev_run)

    checkpoint_callback = ModelCheckpoint(
        # dirpath=f"/data20t/guanwenfei/ckpt/Globalscale/V_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        dirpath="/data20t/guanwenfei/ckpt/wild_data/version1",
        filename="{epoch:02d}-{step:05d}-{val_loss:.4f}",
        every_n_epochs=1,
        save_top_k=-1,
        save_weights_only=False,
        monitor='val_loss',  # Specify the metric to monitor
        mode='min',            # Specify that lower values of 'val_loss' are better
        save_last=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Replace wandb logger with TensorBoardLogger
    # wandb_logger = WandbLogger()
    tb_logger = TensorBoardLogger("lightning_logs", name="wild_data_train", version="version1")

    # from lightning.pytorch.profilers import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath='profile', filename='result_fast_matcher')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,  # Use TensorBoard logger instead of WandbLogger
        fast_dev_run=args.fast_dev_run,
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        # profiler=profiler
        log_every_n_steps=4,
        # --- DDP 配置 ---
        strategy='ddp_find_unused_parameters_true', # 或者 'ddp_find_unused_parameters_true' 如果遇到未使用参数的错误
        # strategy='auto',
        accelerator="gpu",          # 指定使用 GPU
        devices=[2, 3],
        )


    if args.resume:
        trainer.fit(net, datamodule=dm, ckpt_path="/data20t/guanwenfei/ckpt/Globalscale/version2/epoch=974-step=165751-val_loss=0.1201.ckpt")
    else:
        trainer.fit(net, datamodule=dm)
    # trainer.fit(net, datamodule=dm)