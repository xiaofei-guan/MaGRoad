from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from datamodule import SamRoadDataModule

from utils import load_config
from model import SAMRoad

import wandb

# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
# from lightning.pytorch.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    # "--precision", default=16, help="32 or 16"
    "--precision", default="16-mixed", help="32 or 16-mixed"
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)

    
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    

    net = SAMRoad(config)
    dm = SamRoadDataModule(config, dev_run=False)

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    tb_logger = TensorBoardLogger("lightning_logs", name="wild_data_test")


    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        accelerator="gpu",          # 指定使用 GPU
        devices=[0],
        logger=tb_logger,
        # profiler=profiler
        )

    trainer.test(net, datamodule=dm, ckpt_path=args.checkpoint)