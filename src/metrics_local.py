import math
from typing import Dict, List

import torch
import torch.nn as nn

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None

try:
    import lpips
except ImportError:
    lpips = None


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0.0
        self.cnt = 0

    def forward(self, pre: torch.Tensor, label: torch.Tensor) -> None:
        pre = pre.reshape(-1)
        label = label.reshape(-1)
        self.loss += torch.sum(torch.abs(pre - label)).item()
        self.cnt += label.numel()

    def calculate(self) -> float:
        return self.loss / max(self.cnt, 1)


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0.0
        self.cnt = 0

    def forward(self, pre: torch.Tensor, label: torch.Tensor) -> None:
        pre = pre.reshape(-1)
        label = label.reshape(-1)
        self.loss += torch.sum((pre - label) ** 2).item()
        self.cnt += label.numel()

    def calculate(self) -> float:
        return self.loss / max(self.cnt, 1)


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0.0
        self.cnt = 0

    def forward(self, pre: torch.Tensor, label: torch.Tensor) -> None:
        pre = pre.reshape(-1)
        label = label.reshape(-1)
        self.loss += torch.sum((pre - label) ** 2).item()
        self.cnt += label.numel()

    def calculate(self) -> float:
        return math.sqrt(self.loss / max(self.cnt, 1))


class SSIM(nn.Module):
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.ssim_sum = 0.0
        self.cnt = 0
        self.enabled = structural_similarity is not None
        self.data_range = float(data_range)

    def forward(self, pre: torch.Tensor, label: torch.Tensor) -> None:
        if not self.enabled:
            return
        pre_np = pre.detach().cpu().numpy()
        label_np = label.detach().cpu().numpy()
        batch_size = pre_np.shape[0]
        for i in range(batch_size):
            self.ssim_sum += structural_similarity(
                pre_np[i],
                label_np[i],
                data_range=self.data_range,
                K1=0.001,
                K2=0.003,
            )
        self.cnt += batch_size

    def calculate(self) -> float:
        if not self.enabled:
            return float("nan")
        return self.ssim_sum / max(self.cnt, 1)


class ClassificationMetrics:
    def __init__(self, thresholds: List[int] = None):
        self.thresholds = thresholds or [20, 25, 30, 35, 40, 45, 50, 55]
        self.tp = {th: 0.0 for th in self.thresholds}
        self.tn = {th: 0.0 for th in self.thresholds}
        self.fp = {th: 0.0 for th in self.thresholds}
        self.fn = {th: 0.0 for th in self.thresholds}

    def add_data(self, pre: torch.Tensor, label: torch.Tensor) -> None:
        for th in self.thresholds:
            pre_flag = (pre > th).int()
            label_flag = (label > th).int()
            self.tp[th] += ((pre_flag == 1) & (label_flag == 1)).sum().item()
            self.tn[th] += ((pre_flag == 0) & (label_flag == 0)).sum().item()
            self.fp[th] += ((pre_flag == 1) & (label_flag == 0)).sum().item()
            self.fn[th] += ((pre_flag == 0) & (label_flag == 1)).sum().item()

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    def get_all_scores(self) -> Dict[str, float]:
        """CSI → POD → HSS → FAR, each block sorted by ascending threshold."""
        rows = []
        for th in self.thresholds:
            tp, tn, fp, fn = self.tp[th], self.tn[th], self.fp[th], self.fn[th]
            csi = self._safe_div(tp, tp + fn + fp)
            pod = self._safe_div(tp, tp + fn)
            far = self._safe_div(fp, tp + fp)
            denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
            hss = (2.0 * (tp * tn - fn * fp) / denom) if denom > 0 else 0.0
            rows.append((th, csi, pod, hss, far))
        out: Dict[str, float] = {}
        for th, csi, _, _, _ in rows:
            out[f"csi_{th}"] = csi
        for th, _, pod, _, _ in rows:
            out[f"pod_{th}"] = pod
        for th, _, _, hss, _ in rows:
            out[f"hss_{th}"] = hss
        for th, _, _, _, far in rows:
            out[f"far_{th}"] = far
        return out


class ScoresAllLocal:
    def __init__(
        self,
        cls_dbz_scale: float = 60.0,
        ssim_data_range: float = 1.0,
        normalized_01: bool = True,
    ):
        self.cls_dbz_scale = float(cls_dbz_scale)
        self.normalized_01 = bool(normalized_01)
        self.mae = MAE()
        self.mse = MSE()
        self.rmse = RMSE()
        self.ssim = SSIM(data_range=ssim_data_range)
        self.cls = ClassificationMetrics()

        self.lpips_enabled = lpips is not None
        self.lpips_model = None
        self.lpips_sum = 0.0
        self.lpips_cnt = 0
        if self.lpips_enabled:
            self.lpips_model = lpips.LPIPS(net="alex").to("cpu").eval()

    def _lpips_prep(self, x: torch.Tensor) -> torch.Tensor:
        # LPIPS expects [-1, 1]; metrics are evaluated after restoring dBZ scale.
        x = x.unsqueeze(1)  # [B,1,H,W]
        x = (x / 30.0) - 1.0
        return x.clamp(-1, 1).to("cpu")

    @torch.no_grad()
    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        out = output.detach().to("cpu")
        tgt = target.detach().to("cpu")
        scale = self.cls_dbz_scale if self.normalized_01 else 1.0
        out_dbz = out * scale
        tgt_dbz = tgt * scale

        self.mae(out_dbz, tgt_dbz)
        self.mse(out_dbz, tgt_dbz)
        self.rmse(out_dbz, tgt_dbz)
        self.ssim(out_dbz, tgt_dbz)
        self.cls.add_data(out_dbz, tgt_dbz)

        if self.lpips_enabled and self.lpips_model is not None:
            out_lp = self._lpips_prep(out_dbz)
            tgt_lp = self._lpips_prep(tgt_dbz)
            self.lpips_sum += self.lpips_model(out_lp, tgt_lp).mean().item()
            self.lpips_cnt += 1

    def get_scores(self) -> Dict[str, float]:
        scores = {
            "MAE": self.mae.calculate(),
            "MSE": self.mse.calculate(),
            "RMSE": self.rmse.calculate(),
            "SSIM": self.ssim.calculate(),
            "LPIPS": self.lpips_sum / max(self.lpips_cnt, 1) if self.lpips_enabled else float("nan"),
        }
        scores.update(self.cls.get_all_scores())
        return scores
