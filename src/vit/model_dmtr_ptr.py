import os
import torch
import torch.nn as nn

from einops import rearrange

from vit.model import pair, posemb_sincos_2d
from vit.transformer_dmtr_ptr import TransformerDMTRPTR
from vit.efficient import (
    images_to_patches,
    patches_to_images,
    policy_indices_no_sharing,
    policy_indices_split_merge,
)
from vit.policynet import PolicyNet


def _load_policynet_state(policynet: PolicyNet, path: str) -> None:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"PolicyNet checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and all(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = policynet.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[PolicyNet] load_state_dict strict=False missing={missing} unexpected={unexpected}")


class TokenPatchEmbedding(nn.Module):
    def __init__(self, in_chans, dim, patch_size):
        super().__init__()
        if isinstance(patch_size, int):
            ph = pw = patch_size
        else:
            ph, pw = pair(patch_size)
        self.ph, self.pw = ph, pw
        self.proj = nn.Conv2d(
            in_chans, dim, kernel_size=(ph, pw), stride=(ph, pw)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.reshape(b * n, c, h, w)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.view(b, n, -1)


class TokenPatchEmbeddingSub(nn.Module):
    def __init__(self, in_chans, dim, patch_half):
        super().__init__()
        p = int(patch_half)
        self.p = p
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=(p, p), stride=(p, p))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.reshape(b * n, c, h, w)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.view(b, n, -1)


class ViTDMTRPTR(nn.Module):
    """
    DMTR + PTR: token resampling with progressive token routing.
    In PTR layers, base+sub run compacted MHSA while super uses LN+Linear bypass.
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size=(3, 3),
        in_chans=4,
        out_chans=1,
        dim=512,
        depth=6,
        heads=12,
        mlp_dim=512,
        dim_head=64,
        policynet_path=None,
        bg_merge_ratio=0.5,
        split_ratio=0.05,
        merge_size=2,
        disable_merge=False,
        use_early_bypass=False,
        super_bypass_layer=3,
        base_bypass_layer=4,
        enable_base_bypass=False,
        base_bg_threshold=0.9,
        base_keep_min_ratio=0.2,
    ):
        super().__init__()
        self.image_size = image_height, image_width = pair(image_size)
        self.patch_size = ph, pw = pair(patch_size)
        if ph != pw:
            raise NotImplementedError("ViTDMTRPTR: non-square patch_size not supported by efficient.py")
        self._ps = ph
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dim = dim
        self.bg_merge_ratio = float(bg_merge_ratio)
        self.split_ratio = float(split_ratio)
        self.merge_size = merge_size
        self.disable_merge = disable_merge
        self.use_early_bypass = bool(use_early_bypass)
        self.super_bypass_layer = int(super_bypass_layer)
        self.base_bypass_layer = int(base_bypass_layer)
        self.enable_base_bypass = bool(enable_base_bypass)
        self.base_bg_threshold = float(base_bg_threshold)
        self.base_keep_min_ratio = float(base_keep_min_ratio)
        if merge_size != 2:
            raise NotImplementedError("ViTDMTRPTR: only merge_size=2 is supported")

        assert image_height % ph == 0 and image_width % pw == 0, (
            "image dimensions must be divisible by the patch size"
        )
        if not disable_merge and self.split_ratio > 0 and self._ps % 2 != 0:
            raise ValueError("ViTDMTRPTR: split_ratio>0 requires even patch_size (e.g. P=4)")

        self.policynet = PolicyNet(
            patch_size=ph,
            input_hw=(image_height, image_width),
        )
        if not disable_merge:
            if not policynet_path or not os.path.isfile(policynet_path):
                raise FileNotFoundError(
                    "ViTDMTRPTR needs --policynet-path to an existing file when merge is enabled"
                )
            _load_policynet_state(self.policynet, policynet_path)
        for p in self.policynet.parameters():
            p.requires_grad = False
        self.policynet.eval()

        self.token_embed = TokenPatchEmbedding(in_chans, dim, self._ps)
        ph2 = max(1, ph // 2)
        self.token_embed_sub = TokenPatchEmbeddingSub(in_chans, dim, ph2)
        aux_hidden = max(dim // 2, 16)
        self.base_bg_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, aux_hidden),
            nn.GELU(),
            nn.Linear(aux_hidden, 1),
        )
        self.transformer = TransformerDMTRPTR(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            use_early_bypass=self.use_early_bypass,
            super_bypass_layer=self.super_bypass_layer,
            base_bypass_layer=self.base_bypass_layer,
            enable_base_bypass=self.enable_base_bypass,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_chans * ph * pw, bias=False),
        )
        self.head_sub = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_chans * ph2 * ph2, bias=False),
        )
        self.base_h = image_height // ph
        self.base_w = image_width // pw
        self._cached_pe_base = None
        self._cached_pe_base_key = None
        self._cached_pe_fine = None
        self._cached_pe_fine_key = None
        self._last_base_bg_logits = None
        self._last_base_bg_coords = None
        self._last_base_bypass_mask = None
        self._last_base_bypassed = 0

    def _pos_embed(self, device, dtype):
        key = (device, dtype, self.base_h, self.base_w, self.dim)
        if key != self._cached_pe_base_key:
            gh, gw = self.base_h, self.base_w
            grid = torch.zeros(1, gh, gw, self.dim, device=device, dtype=dtype)
            pe_flat = posemb_sincos_2d(grid)
            self._cached_pe_base = pe_flat.view(gh, gw, -1)
            self._cached_pe_base_key = key
        return self._cached_pe_base

    def _pos_embed_fine(self, device, dtype):
        key = (device, dtype, self.base_h, self.base_w, self.dim)
        if key != self._cached_pe_fine_key:
            fh, fw = 2 * self.base_h, 2 * self.base_w
            grid = torch.zeros(1, fh, fw, self.dim, device=device, dtype=dtype)
            pe_flat = posemb_sincos_2d(grid)
            self._cached_pe_fine = pe_flat.view(fh, fw, -1)
            self._cached_pe_fine_key = key
        return self._cached_pe_fine

    @staticmethod
    def _gather_pe(pe2d, gh, gw):
        gh = gh.clamp(0, pe2d.shape[0] - 1)
        gw = gw.clamp(0, pe2d.shape[1] - 1)
        return pe2d[gh, gw, :]

    def _make_base_bypass_mask(self, probs):
        if probs is None:
            return None
        cand = probs > self.base_bg_threshold
        bsz, n1 = cand.shape
        if n1 == 0:
            return None
        min_keep = max(1, int(round(n1 * self.base_keep_min_ratio)))
        max_bypass = max(0, n1 - min_keep)
        if max_bypass <= 0:
            return torch.zeros_like(cand, dtype=torch.bool)
        topk_idx = probs.topk(k=max_bypass, dim=1, largest=True).indices
        topk_mask = torch.zeros_like(cand, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)
        return cand & topk_mask

    def forward(self, x, return_aux: bool = False):
        b, c, h, w = x.shape
        assert (h, w) == (self.image_size[0], self.image_size[1])
        device = x.device
        dtype = x.dtype

        if self.disable_merge:
            policy_indices = policy_indices_no_sharing(x, self._ps)
        else:
            with torch.no_grad():
                pred = self.policynet(x)
                score = pred["score"]
            policy_indices = policy_indices_split_merge(
                score,
                self._ps,
                split_ratio=self.split_ratio,
                merge_ratio=self.bg_merge_ratio,
            )

        p1, p2, p0, policy_code = images_to_patches(x, self._ps, policy_indices)

        pc0 = policy_code[0, :, 0]
        n1 = int((pc0 == 1).sum().item())
        n2 = int((pc0 == 2).sum().item())
        n0 = int((pc0 == 0).sum().item())
        self._last_token_breakdown = (n1, n2, n0)

        tok_parts = []
        pe_b = self._pos_embed(device, dtype)
        pe_f = self._pos_embed_fine(device, dtype)
        base_tok = None
        base_coords = None

        if n1 > 0:
            t1 = self.token_embed(p1[:, :n1])
            c1 = policy_code[:, :n1]
            pe1 = self._gather_pe(pe_b, c1[:, :, 1].long(), c1[:, :, 2].long())
            base_tok = t1 + pe1
            base_coords = c1[:, :, 1:3].long()
            tok_parts.append(base_tok)
        if n2 > 0:
            t2 = self.token_embed(p2[:, :n2])
            c2 = policy_code[:, n1 : n1 + n2]
            pe2 = self._gather_pe(pe_b, c2[:, :, 1].long(), c2[:, :, 2].long())
            tok_parts.append(t2 + pe2)
        if n0 > 0:
            t0 = self.token_embed_sub(p0[:, :n0])
            c0 = policy_code[:, n1 + n2 :]
            gh = c0[:, :, 1].long()
            gw = c0[:, :, 2].long()
            si = c0[:, :, 3].long()
            di = torch.div(si, 2, rounding_mode="trunc")
            dj = si % 2
            pe0 = self._gather_pe(pe_f, gh * 2 + di, gw * 2 + dj)
            tok_parts.append(t0 + pe0)

        base_bg_logits = None
        base_bypass_mask = None
        if n1 > 0:
            base_bg_logits = self.base_bg_head(base_tok).squeeze(-1)
            if self.use_early_bypass and self.enable_base_bypass:
                base_bg_probs = torch.sigmoid(base_bg_logits)
                base_bypass_mask = self._make_base_bypass_mask(base_bg_probs)
        if base_bg_logits is None:
            base_bg_logits = torch.empty(b, 0, device=device, dtype=dtype)
            base_coords = torch.empty(b, 0, 2, device=device, dtype=torch.long)
        self._last_base_bg_logits = base_bg_logits
        self._last_base_bg_coords = base_coords
        self._last_base_bypass_mask = base_bypass_mask
        self._last_base_bypassed = int(base_bypass_mask.sum().item()) if base_bypass_mask is not None else 0

        toks = torch.cat(tok_parts, dim=1)
        h = self.transformer(toks, n1, n2, n0, base_bypass_mask=base_bypass_mask)

        idx = 0
        outs = []
        ph, pw = self._ps, self._ps
        ph2 = ph // 2

        def run_head(seq, hb):
            o = hb
            for block in seq:
                o = block(o)
            return o

        if n1 > 0:
            h1 = h[:, idx : idx + n1]
            idx += n1
            y1 = run_head(self.head, h1).view(b, n1, self.out_chans, ph, pw)
            outs.append(y1)
        else:
            outs.append(torch.empty(b, 0, self.out_chans, ph, pw, device=device, dtype=dtype))

        if n2 > 0:
            h2 = h[:, idx : idx + n2]
            idx += n2
            y2 = run_head(self.head, h2).view(b, n2, self.out_chans, ph, pw)
            outs.append(y2)
        else:
            outs.append(torch.empty(b, 0, self.out_chans, ph, pw, device=device, dtype=dtype))

        if n0 > 0:
            h0 = h[:, idx : idx + n0]
            y0 = run_head(self.head_sub, h0).view(b, n0, self.out_chans, ph2, ph2)
            outs.append(y0)
        else:
            outs.append(torch.empty(b, 0, self.out_chans, ph2, ph2, device=device, dtype=dtype))

        y1, y2, y0 = outs
        P = ph
        strip_parts = [rearrange(y1, "b n c ph pw -> b c ph (n pw)", n=n1)] if n1 > 0 else []
        strip_parts += [rearrange(y2, "b n c ph pw -> b c ph (n pw)", n=n2)] if n2 > 0 else []
        if strip_parts:
            y_strip = torch.cat(strip_parts, dim=-1)
        else:
            y_strip = torch.zeros(b, self.out_chans, P, 0, device=device, dtype=dtype)

        y = patches_to_images(
            y_strip,
            policy_code,
            (self.base_h, self.base_w),
            self._ps,
            y_scale0=y0 if n0 > 0 else None,
        )
        if return_aux:
            return y, base_bg_logits, base_coords
        return y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 4, 192, 384, device=device)
    m = ViTDMTRPTR(
        image_size=x.shape[2:],
        patch_size=4,
        in_chans=4,
        out_chans=1,
        dim=64,
        depth=4,
        heads=4,
        mlp_dim=128,
        dim_head=32,
        policynet_path=None,
        disable_merge=True,
        use_early_bypass=True,
        super_bypass_layer=3,
        base_bypass_layer=4,
        enable_base_bypass=True,
        base_bg_threshold=0.9,
        base_keep_min_ratio=0.2,
    ).to(device)
    m.train()
    y = m(x)
    print("out", y.shape)
    assert y.shape == (1, 1, 192, 384), y.shape
