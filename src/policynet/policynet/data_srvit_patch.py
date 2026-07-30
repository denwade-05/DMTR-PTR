"""Random PolicyNet dataset for smoke tests (no external files)."""
import numpy as np
import torch
from torch.utils.data import Dataset


def generate_policy_target(radar: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """
    Patch-wise max of normalized radar as regression target.

    radar: [1, H, W]
    returns: [H//P, W//P] float32
    """
    assert radar.ndim == 3 and radar.shape[0] == 1, (
        f"radar shape should be [1,H,W], got {radar.shape}"
    )
    _, H, W = radar.shape
    p = int(patch_size)
    assert H % p == 0 and W % p == 0
    radar = radar.squeeze(0)
    radar_patch = radar.view(H // p, p, W // p, p)
    radar_patch = radar_patch.permute(0, 2, 1, 3).contiguous()
    patch_max = radar_patch.reshape(H // p, W // p, -1).max(dim=-1).values
    return patch_max.to(torch.float32)


class TokenSharingPolicyDatasetSrvitDown4(Dataset):
    def __init__(
        self,
        data_mode: str = "train",
        ret_datetime: bool = False,
        patch_size: int = 4,
        num_samples: int = 32,
        channels: int = 4,
        height: int = 192,
        width: int = 384,
        seed: int = 0,
    ):
        assert data_mode in ("train", "test")
        self.ret_datetime = ret_datetime
        self.patch_size = int(patch_size)
        self.num_samples = int(num_samples)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self._seed = int(seed) + (0 if data_mode == "train" else 10_000)
        self.xshape = (self.channels, self.height, self.width)
        self.tshape = (1, self.height, self.width)
        assert self.height % self.patch_size == 0 and self.width % self.patch_size == 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self._seed + int(idx))
        x = torch.from_numpy(rng.random(self.xshape, dtype=np.float32))
        t = torch.from_numpy(rng.random(self.tshape, dtype=np.float32))
        policy_target = generate_policy_target(t, patch_size=self.patch_size)
        datetime_str = f"{self._seed + int(idx):012d}"
        if self.ret_datetime:
            return x, policy_target, datetime_str
        return x, policy_target


if __name__ == "__main__":
    for p in (3, 4):
        ds = TokenSharingPolicyDatasetSrvitDown4("train", ret_datetime=True, patch_size=p)
        x, pl, _ = ds[0]
        H, W = x.shape[1], x.shape[2]
        assert pl.shape == (H // p, W // p), (pl.shape, p)
        print(f"patch_size={p}: x {x.shape}, target {pl.shape}, ok")
