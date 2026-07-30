"""DMTR-PTR transformer with configurable 2-1-1 bypass stages."""
import torch
import torch.nn as nn

from einops import rearrange

from vit.model import FeedForward


class Attention(nn.Module):
    """Same pattern as vit.model.Attention (pre-norm inside branch; residual outside)."""

    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, active_mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if active_mask is not None:
            if active_mask.dtype != torch.bool:
                active_mask = active_mask.bool()
            key_mask = active_mask[:, None, None, :]
            dots = dots.masked_fill(~key_mask, torch.finfo(dots.dtype).min)
        attn = self.attend(dots)
        if active_mask is not None:
            query_mask = active_mask[:, None, :, None].to(attn.dtype)
            attn = attn * query_mask
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SuperTokenBypass(nn.Module):
    """Residual LN + Linear for super tokens when skipping full self-attention."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.proj(self.norm(x))


class BaseTokenBypass(nn.Module):
    """Residual LN + Linear for base tokens filtered from MHSA."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.proj(self.norm(x))


class DMTRPTRBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        *,
        eb_capable: bool,
    ):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim, mlp_dim)
        self.eb_capable = eb_capable
        self.super_bypass = SuperTokenBypass(dim) if eb_capable else None
        self.base_bypass = BaseTokenBypass(dim) if eb_capable else None

    def _compact_attn_residual(self, x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """DTP-style compact MHSA: gather active tokens, run attention, scatter back."""
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.bool()

        outs = []
        bsz = x.shape[0]
        for b in range(bsz):
            x_b = x[b : b + 1]  # [1, N, C]
            keep_b = keep_mask[b]
            x_new = x_b.clone()
            if keep_b.any():
                x_active = x_b[:, keep_b, :]
                x_active = self.attn(x_active) + x_active
                x_new[:, keep_b, :] = x_active
            outs.append(x_new)
        return torch.cat(outs, dim=0)

    def forward(self, x, n1: int, n2: int, n0: int, stage_mode: str, base_bypass_mask=None):
        if stage_mode == "full":
            x = self.attn(x) + x
            x = self.ff(x) + x
            return x

        if not self.eb_capable or self.super_bypass is None:
            x = self.attn(x) + x
            x = self.ff(x) + x
            return x

        if n1 + n2 + n0 == 0:
            return x

        x1 = x[:, :n1]
        x2 = x[:, n1 : n1 + n2]
        x0 = x[:, n1 + n2 :]

        if n1 + n0 > 0:
            x_bs = torch.cat([x1, x0], dim=1)
            use_compact = stage_mode == "super_base" and n1 > 0 and base_bypass_mask is not None
            if use_compact:
                if base_bypass_mask.dtype != torch.bool:
                    base_bypass_mask = base_bypass_mask.bool()
                keep_base = ~base_bypass_mask
                keep_sub = torch.ones(
                    keep_base.shape[0],
                    n0,
                    device=keep_base.device,
                    dtype=torch.bool,
                )
                keep_mask = torch.cat(
                    [
                        keep_base,
                        keep_sub,
                    ],
                    dim=1,
                )
                x_bs = self._compact_attn_residual(x_bs, keep_mask)
            else:
                x_bs = self.attn(x_bs) + x_bs
            x1_new = x_bs[:, :n1]
            x0_new = x_bs[:, n1:]
        else:
            x1_new = x1
            x0_new = x0

        if stage_mode == "super_base" and n1 > 0 and base_bypass_mask is not None and self.base_bypass is not None:
            base_bg = x1 + self.base_bypass(x1)
            x1_new = torch.where(base_bypass_mask.unsqueeze(-1), base_bg, x1_new)

        if n2 > 0:
            x2_new = x2 + self.super_bypass(x2)
        else:
            x2_new = x2

        x = torch.cat([x1_new, x2_new, x0_new], dim=1)

        x = self.ff(x) + x
        return x


class TransformerDMTRPTR(nn.Module):
    """
    Token order: [scale1 base | scale2 super/merge | scale0 sub].

    For a depth-4 backbone, the default schedule is:
    - layers 1-2: full attention
    - layer 3: super-token bypass
    - layer 4: super-token bypass + partial base-token bypass
    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        *,
        use_early_bypass: bool = False,
        super_bypass_layer: int = 3,
        base_bypass_layer: int = 4,
        enable_base_bypass: bool = False,
    ):
        super().__init__()
        if use_early_bypass:
            if super_bypass_layer < 1 or super_bypass_layer > depth:
                raise ValueError(
                    f"use_early_bypass requires 1 <= super_bypass_layer <= depth; "
                    f"got super_bypass_layer={super_bypass_layer}, depth={depth}"
                )
            if enable_base_bypass and (base_bypass_layer < super_bypass_layer or base_bypass_layer > depth):
                raise ValueError(
                    f"enable_base_bypass requires super_bypass_layer <= base_bypass_layer <= depth; "
                    f"got super_bypass_layer={super_bypass_layer}, base_bypass_layer={base_bypass_layer}, depth={depth}"
                )
        self.depth = depth
        self.use_early_bypass = bool(use_early_bypass)
        self.super_bypass_layer = int(super_bypass_layer)
        self.base_bypass_layer = int(base_bypass_layer)
        self.enable_base_bypass = bool(enable_base_bypass)
        self.layers = nn.ModuleList()
        for i in range(depth):
            eb_capable = use_early_bypass and i >= (self.super_bypass_layer - 1)
            self.layers.append(
                DMTRPTRBlock(
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    eb_capable=eb_capable,
                )
            )

    def forward(self, x, n1: int, n2: int, n0: int, base_bypass_mask=None):
        for layer_idx, block in enumerate(self.layers):
            if not self.use_early_bypass or layer_idx < (self.super_bypass_layer - 1):
                stage_mode = "full"
            elif self.enable_base_bypass and layer_idx >= (self.base_bypass_layer - 1):
                stage_mode = "super_base"
            else:
                stage_mode = "super"
            x = block(x, n1, n2, n0, stage_mode=stage_mode, base_bypass_mask=base_bypass_mask)
        return x
