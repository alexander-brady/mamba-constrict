from typing import Literal

import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def load_dataloader(data_cfg: DictConfig, split: Literal["train", "validation"]):
    """
    Load a PyTorch DataLoader for a specific data split.

    Args:
        data_cfg (DictConfig): Configuration dictionary (cfg.data).
        split (Literal["train", "validation"]): The data split to load.

    Returns:
        DataLoader: A PyTorch DataLoader for the specified data split.
    """
    path = data_cfg[split].save_dir
    ds = load_from_disk(path)
    ds.set_format(type="torch", columns=["input_ids", "labels"])

    # Create dataloader
    dataloader = DataLoader(
        ds,
        shuffle=(split == "train"),
        **data_cfg.dataloader_kwargs,
    )
    return dataloader
