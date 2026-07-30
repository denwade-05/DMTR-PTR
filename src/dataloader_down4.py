"""Synthetic random dataloader for smoke tests and API demos.

Input:  [4, 192, 384]
Target: [1, 192, 384]
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RandomDown4(Dataset):
    def __init__(
        self,
        data_mode="train",
        ret_datetime=True,
        num_samples=32,
        channels=4,
        height=192,
        width=384,
        seed=0,
    ):
        assert data_mode in ("train", "test")
        self.ret_datetime = ret_datetime
        self.num_samples = int(num_samples)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        # Offset seed so train/test draws differ.
        self._seed = int(seed) + (0 if data_mode == "train" else 10_000)
        self.xshape = (self.channels, self.height, self.width)
        self.tshape = (1, self.height, self.width)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self._seed + int(idx))
        x = rng.random(self.xshape, dtype=np.float32)
        t = rng.random(self.tshape, dtype=np.float32)
        datetime_str = f"{self._seed + int(idx):012d}"
        if self.ret_datetime:
            return x, t, datetime_str
        return x, t


def get_dataset_down4(data_mode="train", ret_datetime=True, num_samples=32, **_kwargs):
    return RandomDown4(
        data_mode=data_mode,
        ret_datetime=ret_datetime,
        num_samples=num_samples,
    )


def get_dataloader(args):
    n_train = int(getattr(args, "dummy_train_samples", 32))
    n_test = int(getattr(args, "dummy_test_samples", 8))

    if args.test:
        train_loader = None
        val_dataset = get_dataset_down4("test", num_samples=n_test)
    else:
        train_dataset = get_dataset_down4("train", num_samples=n_train)
        val_dataset = get_dataset_down4("test", num_samples=n_test)

    if args.distributed:
        train_sampler = (
            None
            if args.test
            else torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None

    if args.test:
        train_loader = None
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, train_sampler, val_loader, val_sampler
