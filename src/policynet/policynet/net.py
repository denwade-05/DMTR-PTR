import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEGenexp(nn.Module):
    """加权 MSE：w0 * exp(w1 * T^w2) * (Y - T)^2，偏大目标 T 权重大。"""

    def __init__(self, weight=(1.0, 0.0, 0.0)):
        super().__init__()
        self.weight = weight

    def forward(self, Y, T):
        w0, w1, w2 = self.weight
        coef = w0 * torch.exp(w1 * torch.pow(T, w2))
        return torch.mean(coef * torch.square(Y - T))


class Conv2D(nn.Module):
    def __init__(self, in_chs, out_chs, activate_last_layer=True):
        super().__init__()
        if activate_last_layer:
            self.conv2d = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_chs),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chs, out_chs, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_chs),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chs, out_chs, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_chs),
                nn.ReLU(inplace=True),
            )
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=True),
                nn.BatchNorm2d(out_chs),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2d = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_chs),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chs, out_chs, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_chs),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chs, out_chs, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_chs),
            )
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=True),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        return self.conv1x1(x) + self.conv2d(x)


class UpSkip(nn.Module):
    """上采样后与 encoder skip 拼接，再卷积融合。"""

    def __init__(self, up_in: int, up_out: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(up_in, up_out, 4, 2, 1, bias=True),
            nn.BatchNorm2d(up_out),
            nn.ReLU(inplace=True),
        )
        self.merge = Conv2D(2 * up_out, up_out)

    def forward(self, x, skip):
        x = self.up(x)
        return self.merge(torch.cat([x, skip], dim=1))


class PolicyAsymmetricUNetBackbone(nn.Module):
    """
    非对称 U-Net：5×下采样至 H/32，3×上采样至 H/4（含 skip）。
    input (192,384) 时输出 48×96。
    """

    def __init__(self, in_channels: int, out_channels: int, channels: tuple):
        super().__init__()
        c0, c1, c2, c3, c4, c5 = channels
        self.pool = nn.MaxPool2d(2, 2)

        self.enc0 = Conv2D(in_channels, c0)
        self.enc1 = Conv2D(c0, c1)
        self.enc2 = Conv2D(c1, c2)
        self.enc3 = Conv2D(c2, c3)
        self.enc4 = Conv2D(c3, c4)
        self.bottleneck = Conv2D(c4, c5)

        self.dec2 = UpSkip(c5, c4)
        self.dec1 = UpSkip(c4, c3)
        self.dec0 = UpSkip(c3, c2)
        self.head = Conv2D(c2, out_channels, activate_last_layer=False)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d2 = self.dec2(b, e4)
        d1 = self.dec1(d2, e3)
        d0 = self.dec0(d1, e2)
        return self.head(d0)


class PolicyNet(nn.Module):
    """
    Patch 级雷达强度估计：非对称 U-Net → sigmoid → score ∈ [0,1]^(B×1×H/P×W/P)。

    训练：MSEGenexp(score, target)，和 Dataset 中 patch 内 max(t) 对齐。
    """

    def __init__(
        self,
        patch_size: int = 4,
        input_hw: tuple = (192, 384),
        block_out_channels=(16, 32, 64, 128, 256, 256),
        mse_genexp_weight=(1.0, 5.0, 4.0),
    ):
        super().__init__()
        h_in, w_in = int(input_hw[0]), int(input_hw[1])
        if h_in % patch_size != 0 or w_in % patch_size != 0:
            raise ValueError(
                f"input_hw {input_hw} must be divisible by patch_size={patch_size}"
            )
        self.patch_size = int(patch_size)
        self.input_hw = (h_in, w_in)
        self.out_h = h_in // self.patch_size
        self.out_w = w_in // self.patch_size

        if h_in % 32 != 0 or w_in % 32 != 0:
            raise ValueError(
                f"H,W must be divisible by 32 (5 encoder pools), got {input_hw}"
            )

        if len(block_out_channels) != 6:
            raise ValueError("block_out_channels must be length-6 tuple (c0..c5)")

        self.backbone = PolicyAsymmetricUNetBackbone(
            in_channels=4,
            out_channels=1,
            channels=block_out_channels,
        )
        self.criterion = MSEGenexp(weight=mse_genexp_weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, targets=None):
        raw = self.backbone(x)
        if raw.shape[2] != self.out_h or raw.shape[3] != self.out_w:
            raw = F.interpolate(
                raw,
                size=(self.out_h, self.out_w),
                mode="bilinear",
                align_corners=False,
            )
        score = torch.sigmoid(raw)
        out_dict = {"score": score}
        if targets is not None:
            out_dict["loss"] = self.loss(score, targets)
        return out_dict

    def loss(self, score, targets):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1).to(dtype=score.dtype)
        else:
            targets = targets.to(dtype=score.dtype)
        return self.criterion(score, targets)


if __name__ == "__main__":
    B, h, w = 2, 192, 384
    x = torch.randn(B, 4, h, w)

    m4 = PolicyNet(patch_size=4, input_hw=(h, w))
    assert m4.backbone(x).shape == (B, 1, h // 4, w // 4)

    for P in (3, 4):
        targ = torch.rand(B, h // P, w // P)
        model = PolicyNet(patch_size=P, input_hw=(h, w))
        model.train()
        out = model(x, targ)
        assert out["score"].shape == (B, 1, h // P, w // P)
        print(f"patch_size={P}: score {tuple(out['score'].shape)}, loss {out['loss'].item():.4f}")

    print("PolicyNet (regression) tests passed.")
