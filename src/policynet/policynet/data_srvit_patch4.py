import torch

from .data_srvit_patch import TokenSharingPolicyDatasetSrvitDown4


class TokenSharingPolicyDatasetSrvitDown4Patch4(TokenSharingPolicyDatasetSrvitDown4):
    """patch_size=4"""

    def __init__(self, data_mode="train", ret_datetime=False):
        super().__init__(data_mode=data_mode, ret_datetime=ret_datetime, patch_size=4)


if __name__ == "__main__":
    print("Testing TokenSharingPolicyDatasetSrvitDown4Patch4...")
    ds = TokenSharingPolicyDatasetSrvitDown4Patch4("train", ret_datetime=True)
    x, pl, s = ds[0]
    print("x", x.shape, "target", pl.shape, "id", s)
    assert pl.shape == (48, 96), pl.shape
    assert pl.dtype == torch.float32
    assert pl.min() >= 0 and pl.max() <= 1
    print("Test passed.")
