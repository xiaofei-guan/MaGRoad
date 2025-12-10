import os, cv2

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .globalscale import GlobalScaleSatMapDataset
from .wildroad import WildRoadSatMapDataset
from .wildroad import graph_collate_fn



def _limit_lib_threads(_):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


class DataSetModule(pl.LightningDataModule):
    def __init__(self, config, dev_run=False):
        super().__init__()
        self.config = config
        self.dev_run = dev_run
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if self.config.DATASET == 'globalscale':
                self.train_ds = GlobalScaleSatMapDataset(self.config, is_train=True, dev_run=self.dev_run)
                self.val_ds = GlobalScaleSatMapDataset(self.config, is_train=False, dev_run=self.dev_run)
            elif self.config.DATASET == 'wildroad':
                self.train_ds = WildRoadSatMapDataset(self.config, is_train=True, dev_run=self.dev_run)
                self.val_ds = WildRoadSatMapDataset(self.config, is_train=False, dev_run=self.dev_run)
            else:
                raise ValueError(f'Invalid dataset: {self.config.DATASET}')
        elif stage in (None, "test"):
            if self.config.DATASET == 'globalscale':
                self.test_ds = GlobalScaleSatMapDataset(self.config, is_train=False, dev_run=self.dev_run)
            elif self.config.DATASET == 'wildroad':
                self.test_ds = WildRoadSatMapDataset(self.config, is_train=False, dev_run=self.dev_run)
            else:
                raise ValueError(f'Invalid dataset: {self.config.DATASET}')

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.DATA_WORKER_NUM,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=1,
            worker_init_fn=_limit_lib_threads,
            collate_fn=graph_collate_fn,
        )
    
    # For the Globalscale dataset, you can adjust the parameters of Dataloader to speed up training,
    # but be careful to avoid CPU memory explosion!

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.DATA_WORKER_NUM,
            persistent_workers=True,
            prefetch_factor=1,
            pin_memory=True,
            worker_init_fn=_limit_lib_threads,
            collate_fn=graph_collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.DATA_WORKER_NUM,
            pin_memory=True,
            collate_fn=graph_collate_fn,
        )


