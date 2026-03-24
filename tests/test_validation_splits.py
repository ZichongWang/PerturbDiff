import logging
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_module.data_module import PBMCPerturbationDataModule
from src.data.data_module.data_module_setup import _enabled_split_indices


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


class FakeDataset:
    def __init__(self):
        self.data_args = AttrDict(
            use_cell_set=1,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=None,
        )
        self.dataset_path_map = {"fake_ds": "/tmp/fake_ds.h5ad"}
        self.grouped_pert_data_indices = {
            "fake_ds": {("fake_pert", "fake_celltype", "fake_batch"): np.array([0, 1], dtype=np.int64)}
        }
        self.control_type = {"fake_ds": "control"}

    def _compute_index(self, ds_name_hint, gidx):
        return ds_name_hint, int(gidx)

    def collate_fn(self, batch):
        return batch

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"index": index}


def make_datamodule(validation_splits=None):
    data_args = AttrDict(
        validation_splits=validation_splits,
        use_fixed_pairing=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=None,
    )
    dm = PBMCPerturbationDataModule(
        seed=0,
        micro_batch_size=2,
        data_args=data_args,
        py_logger=logging.getLogger("test_validation_splits"),
    )
    dm.validation_dataset = FakeDataset()
    dm.test_dataset = FakeDataset()
    return dm


def test_default_uses_validation_and_test():
    dm = make_datamodule()
    loaders = dm.val_dataloader()

    assert dm.all_split_names == ["validation", "test"]
    assert len(loaders) == 2


def test_can_disable_test_split_for_validation():
    dm = make_datamodule(["validation"])
    loaders = dm.val_dataloader()

    assert dm.all_split_names == ["validation"]
    assert len(loaders) == 1


def test_rejects_unknown_validation_split_name():
    with pytest.raises(ValueError, match="Unsupported validation split names"):
        make_datamodule(["validation", "unknown"])


def test_setup_filters_out_disabled_test_split():
    dm = make_datamodule(["validation"])
    split_indices = {
        "train": [1, 2],
        "validation": [3],
        "test": [4],
    }

    assert _enabled_split_indices(dm, split_indices) == {
        "train": [1, 2],
        "validation": [3],
    }
