import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import SatMapDataset, graph_collate_fn
import os, cv2


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
            self.train_ds = SatMapDataset(self.config, is_train=True, dev_run=self.dev_run)
            self.val_ds = SatMapDataset(self.config, is_train=False, dev_run=self.dev_run)
        elif stage in (None, "test"):
            self.test_ds = SatMapDataset(self.config, is_train=False, dev_run=self.dev_run)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.DATA_WORKER_NUM,
            persistent_workers=True,
            pin_memory=False,
            prefetch_factor=1,
            worker_init_fn=_limit_lib_threads,
            collate_fn=graph_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.DATA_WORKER_NUM,
            persistent_workers=True,
            prefetch_factor=1,
            worker_init_fn=_limit_lib_threads,
            pin_memory=False,
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


