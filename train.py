from argparse import ArgumentParser
import torch
from datasets.datamodule import DataSetModule

from utils import load_config
from model import MaGRoad
from sam_road_plus_model import SAMRoadplus

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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


    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(seed=3407, workers=True)

    model_name = config.get("MODEL_NAME", "MaGRoad")

    if model_name == "MaGRoad":
        net = MaGRoad(config)
    elif model_name == "SAMRoadplus":
        net = SAMRoadplus(config)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    dm = DataSetModule(config, dev_run=dev_run)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_logs/wild_road_training/ckpt/V_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        filename="{epoch:02d}-{step:05d}-{val_loss:.4f}",
        every_n_epochs=1,
        save_top_k=-1,
        save_weights_only=False,
        monitor='val_loss',  # Specify the metric to monitor
        mode='min',            # Specify that lower values of 'val_loss' are better
        save_last=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    tb_logger = TensorBoardLogger("lightning_logs", name="wild_road_train")

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        fast_dev_run=args.fast_dev_run,
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        # profiler=profiler
        log_every_n_steps=4,
        strategy='ddp_find_unused_parameters_true',
        # strategy='auto',
        accelerator="gpu",
        devices=[3],
        )


    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.fit(net, datamodule=dm, ckpt_path=args.resume)
    else:
        trainer.fit(net, datamodule=dm)
