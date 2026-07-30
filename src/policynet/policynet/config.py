from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """
    PolicyNet 训练默认配置。

    DDP（torchrun --nproc_per_node=K）时：
      - batch_size_train 为每张 GPU 上的 batch；
      - 近似全局 batch ≈ batch_size_train × K。
    """

    batch_size_train: int = 8
    batch_size_test: int = 32

    lr: float = 5e-5
    lr_momentum: float = 0.9
    weight_decay: float = 1e-3

    num_iterations: int = 50000
    log_iterations: int = 100
    summary_iterations: int = 20
    eval_iterations: int = 10000

    optimizer: str = "AdamW"
    enable_cuda: bool = True

    logdir: str = "./logdir/"
    dataset: str = "srvit_down4_patch4"

    # 每进程各有一个 DataLoader；K 卡时总 worker≈ num_workers×K，可按机器调小
    num_workers: int = 4

    # torch.distributed.init_process_group backend
    ddp_backend: str = "nccl"

    # MSEGenexp(score, target) 的 (w0, w1, w2)，见 policynet.net.MSEGenexp
    mse_genexp_weight: Tuple[float, float, float] = (1.0, 5.0, 4.0)

    # PolicyNet 非对称 U-Net 各层通道 (c0..c5)
    block_out_channels: Tuple[int, int, int, int, int, int] = (
        16,
        32,
        64,
        128,
        256,
        256,
    )

    # 验证「强对流」patch：max(t) ≥ 该阈值视为 ≥35 dBZ（t = dBZ/60）
    conv_patch_t_threshold: float = 35.0 / 60.0

    # 按预测 score 取全局 Top ρ 的 patch 集合，统计真值强对流的 recall = 命中数/强对流patch总数
    eval_recall_top_fracs: Tuple[float, ...] = (0.02, 0.03, 0.04, 0.05)
