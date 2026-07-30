import torch

from .data_srvit_patch import TokenSharingPolicyDatasetSrvitDown4


class TokenSharingPolicyDatasetSrvitDown4Patch3(TokenSharingPolicyDatasetSrvitDown4):
    """patch_size=3"""

    def __init__(self, data_mode="train", ret_datetime=False):
        super().__init__(data_mode=data_mode, ret_datetime=ret_datetime, patch_size=3)


if __name__ == "__main__":
    print("Testing TokenSharingPolicyDatasetSrvitDown4Patch3...")
    dataset = TokenSharingPolicyDatasetSrvitDown4Patch3(data_mode="train", ret_datetime=True)
    x, policy_target, datetime_str = dataset[0]
    print(f"datetime: {datetime_str}, x: {x.shape}, target: {policy_target.shape}")
    assert x.shape == (4, 192, 384)
    assert policy_target.shape == (64, 128)
    assert policy_target.dtype == torch.float32
    assert policy_target.min() >= 0 and policy_target.max() <= 1
    print("Test passed.")
