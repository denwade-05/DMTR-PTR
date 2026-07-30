import numpy as np
import math
import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange

# for evaluation of all types of policy generation approaches, the output of this function is not bounded by the
# grouping schedules, all consistent potential 2*2 patches will be True.
def policy_gt_gen_for_eval(seg_map, patch_size):
    B, H, W = seg_map.size()
    assert B == 1
    seg_map = seg_map[0]
    group_if_all_ignore = True  # If true, a patch is grouped if it contains only the ignore label 0 (otherwise set to ignore)
    # ignore_if_one_class_plus_ignore = False  # If true, a patch is ignored if it contains one class + ignore label 0 (otherwise set to 'don't group')
    # ignore_if_one_class_plus_ignore_at_edges = True
    patch_groups = np.zeros((H//patch_size//2, H//patch_size//2), dtype=np.uint8)
    for i in range(patch_groups.shape[0]):
        for j in range(patch_groups.shape[1]):
            patch = seg_map[i * patch_size*2:i * patch_size*2 + patch_size*2, j * patch_size*2:j * patch_size*2 + patch_size*2]
            unique = np.unique(patch)
            patch_groups[i, j] = unique.shape[0] == 1

            if not group_if_all_ignore:
                if unique.shape[0] == 1:
                    if np.unique(patch)[0] == 0:  # 0 is the ignore label in GT
                        patch_groups[i, j] = 255  # 255 is the ignore label in the new patch grouping GT

            # if ignore_if_one_class_plus_ignore:
            #     if unique.shape[0] == 2:
            #         if 0 in unique:
            #             patch_groups[i, j] = 255
            #
            # if ignore_if_one_class_plus_ignore_at_edges:
            #     if i in [0, H//patch_size//2] or j in [0, W//patch_size//2]:
            #         if unique.shape[0] == 2:
            #             if 0 in unique:
            #                 patch_groups[i, j] = 255

    return None, patch_groups


def policy_indices_by_policynet_pred(images, patch_size, policy_schedule, policynet_pred):
    B, C, H, W = images.size()
    assert H % patch_size == 0 and W % patch_size == 0
    clue = policynet_pred['logits'].to(torch.float)

    base_grid_H, base_grid_W = H // (patch_size * 2), W // (patch_size * 2)
    num_scale_1, num_scale_2 = policy_schedule

    group_scores = torch.softmax(clue, dim=1)[:, 1]

    selected_msk_scale_1_per_img = list()
    selected_msk_scale_2_per_img = list()

    group_scores = rearrange(group_scores, 'b h w-> b (h w)')
    group_scores_sorted, group_scores_idx = torch.sort(group_scores, descending=True, dim=1)

    for b in range(B):
        grouped_mask = torch.zeros((base_grid_H, base_grid_W)).bool()
        grouped_mask = rearrange(grouped_mask, 'h w-> (h w)')

        group_scores_idx_selected = group_scores_idx[b, 0: num_scale_2]

        grouped_mask[group_scores_idx_selected] = True
        grouped_mask = grouped_mask.view((base_grid_H, base_grid_W))

        selected_msk_scale_2_per_img.append(grouped_mask)

        grouped_mask_large = F.interpolate(grouped_mask.float().unsqueeze(0).unsqueeze(0),
                                           size=(base_grid_H*2, base_grid_W*2),
                                           mode='nearest').squeeze(0).squeeze(0).bool()
        selected_msk_scale_1_per_img.append(torch.logical_not(grouped_mask_large))

    selected_msk_scale_1 = torch.stack(selected_msk_scale_1_per_img, dim=0)
    selected_msk_scale_2 = torch.stack(selected_msk_scale_2_per_img, dim=0)

    assert num_scale_1 == torch.div(torch.sum(selected_msk_scale_1), B, rounding_mode='floor')
    assert num_scale_2 == torch.div(torch.sum(selected_msk_scale_2), B, rounding_mode='floor')

    return selected_msk_scale_1, selected_msk_scale_2


def policy_indices_by_bg_topk_ratio(images, patch_size, bg_merge_ratio, policynet_pred):
    """
    Coarse 2x2 合并：在 (H//(2P), W//(2P)) 格点上，用 P(背景) 取 top K，
    K = round(bg_merge_ratio * G)。logits 若与 coarse 尺寸不一致会双线性插值。
    """
    B, C, H, W = images.size()
    assert H % patch_size == 0 and W % patch_size == 0
    assert 0.0 <= bg_merge_ratio <= 1.0

    clue = policynet_pred['logits'].to(torch.float)
    base_grid_H, base_grid_W = H // (patch_size * 2), W // (patch_size * 2)
    G = base_grid_H * base_grid_W

    if clue.shape[2] != base_grid_H or clue.shape[3] != base_grid_W:
        clue = F.interpolate(
            clue, size=(base_grid_H, base_grid_W), mode='bilinear', align_corners=False
        )

    prob_bg = torch.softmax(clue, dim=1)[:, 0]  # [B, h, w]
    prob_bg = rearrange(prob_bg, 'b h w -> b (h w)')

    bg_topk = int(round(bg_merge_ratio * G))
    bg_topk = max(0, min(G, bg_topk))

    selected_msk_scale_1_per_img = []
    selected_msk_scale_2_per_img = []
    for b in range(B):
        grouped_mask = torch.zeros(
            (base_grid_H, base_grid_W), dtype=torch.bool, device=images.device
        )
        grouped_mask = rearrange(grouped_mask, 'h w -> (h w)')

        if bg_topk > 0:
            _, top_idx = torch.topk(
                prob_bg[b], k=bg_topk, dim=0, largest=True, sorted=False
            )
            grouped_mask[top_idx] = True
        grouped_mask = grouped_mask.view((base_grid_H, base_grid_W))

        selected_msk_scale_2_per_img.append(grouped_mask)
        grouped_mask_large = F.interpolate(
            grouped_mask.float().unsqueeze(0).unsqueeze(0),
            size=(base_grid_H * 2, base_grid_W * 2),
            mode='nearest',
        ).squeeze(0).squeeze(0).bool()
        selected_msk_scale_1_per_img.append(torch.logical_not(grouped_mask_large))

    selected_msk_scale_1 = torch.stack(selected_msk_scale_1_per_img, dim=0)
    selected_msk_scale_2 = torch.stack(selected_msk_scale_2_per_img, dim=0)
    return selected_msk_scale_1, selected_msk_scale_2


def policy_indices_no_sharing(images, patch_size):
    B, C, H, W = images.size()
    assert H % patch_size ==0 and W % patch_size == 0
    base_grid_H, base_grid_W = H // patch_size, W // patch_size

    selected_msk_scale_2 = torch.zeros((B, base_grid_H // 2, base_grid_W // 2), dtype=torch.bool)
    selected_msk_scale_1 = torch.ones((B, base_grid_H, base_grid_W), dtype=torch.bool)

    return (selected_msk_scale_1, selected_msk_scale_2)


def policy_indices_split_merge(
    score,
    patch_size,
    split_ratio=0.05,
    merge_ratio=0.5,
):
    """
    score: (B,1,Hf,Wf)，与 base patch 网格对齐（如 P=4, 192×384 → 48×96）。
    1) 全图 fine 格按分数取 top split_ratio 做 split（subpatch）。
    2) 2×2 不重叠粗化，每格用 max-pool；与 split 相交的粗格禁止 merge。
    3) 在可 merge 粗格中，按分数升序取全局 merge_ratio×G 个做 superpatch merge。
    返回 (msk_scale_1, msk_scale_2, split_mask)，均为 bool；若 split_ratio=0 且无实现需求可退化为两档。
    """
    B, C, Hf, Wf = score.shape
    assert C == 1
    device = score.device
    if patch_size % 2 != 0:
        raise ValueError(
            f"policy_indices_split_merge requires even patch_size for subpatches, got {patch_size}"
        )

    base_grid_H = Hf
    base_grid_W = Wf
    n_fine = base_grid_H * base_grid_W
    coarse_h = base_grid_H // 2
    coarse_w = base_grid_W // 2
    G = coarse_h * coarse_w

    msk1_list, msk2_list, split_list = [], [], []

    for b in range(B):
        s = score[b, 0]
        flat = s.flatten()
        Ks = int(round(split_ratio * n_fine))
        Ks = max(0, min(n_fine, Ks))
        split_flat = torch.zeros(n_fine, dtype=torch.bool, device=device)
        if Ks > 0:
            _, topi = torch.topk(flat, k=Ks, largest=True)
            split_flat[topi] = True
        split_mask = split_flat.view(base_grid_H, base_grid_W)

        s4 = s.unsqueeze(0).unsqueeze(0)
        coarse_scores = F.max_pool2d(s4, kernel_size=2, stride=2).squeeze(0).squeeze(0)

        split_block = F.max_pool2d(split_mask.float().unsqueeze(0).unsqueeze(0), 2, 2).squeeze()
        split_block = split_block > 0.0
        eligible = (~split_block).flatten()

        Km = int(round(merge_ratio * G))
        Km = max(0, min(G, Km))

        coarse_flat = coarse_scores.flatten()
        big = torch.full((G,), float("inf"), device=device, dtype=coarse_flat.dtype)
        big[eligible] = coarse_flat[eligible]
        if Km > 0:
            merge_idx = torch.topk(big, k=Km, largest=False, dim=0).indices
        else:
            merge_idx = torch.tensor([], device=device, dtype=torch.long)
        grouped = torch.zeros(G, dtype=torch.bool, device=device)
        grouped[merge_idx] = True
        merge_coarse = grouped.view(coarse_h, coarse_w)

        merge_large = F.interpolate(
            merge_coarse.float().unsqueeze(0).unsqueeze(0),
            size=(base_grid_H, base_grid_W),
            mode="nearest",
        ).squeeze(0).squeeze(0).bool()

        msk1 = (~merge_large) & (~split_mask)
        msk1_list.append(msk1)
        msk2_list.append(merge_coarse)
        split_list.append(split_mask)

    return (
        torch.stack(msk1_list, dim=0),
        torch.stack(msk2_list, dim=0),
        torch.stack(split_list, dim=0),
    )


def images_to_patches(images, patch_size, policy_indices):
    # policy_indices: (msk1, msk2) 或 (msk1, msk2, split_mask)；后者启用 subpatch(scale=0)
    device = images.device
    B, C, H, W = images.size()
    assert H % patch_size ==0 and W % patch_size == 0
    base_grid_H, base_grid_W = H // patch_size, W // patch_size

    split_mask = None
    if len(policy_indices) == 3:
        selected_msk_scale_1, selected_msk_scale_2, split_mask = policy_indices
    else:
        selected_msk_scale_1, selected_msk_scale_2 = policy_indices

    selected_msk_scale_1 = selected_msk_scale_1.to(device)
    selected_msk_scale_2 = selected_msk_scale_2.to(device)
    if split_mask is not None:
        split_mask = split_mask.to(device)

    patch_scale_1 = rearrange(images, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H, gw=base_grid_W)
    scale_value_1 = torch.ones(
        [B, base_grid_H, base_grid_W, 1], device=device, dtype=torch.float32
    )
    patch_code_scale_1 = torch.cat([
        scale_value_1,
        torch.linspace(0, base_grid_H - 1, base_grid_H, device=device).view(-1, 1, 1).expand_as(scale_value_1),
        torch.linspace(0, base_grid_W - 1, base_grid_W, device=device).view(1, -1, 1).expand_as(scale_value_1),
        torch.zeros([B, base_grid_H, base_grid_W, 1], device=device),
    ], dim=3)

    patch_scale_2 = rearrange(
        F.interpolate(
            images, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False
        ),
        'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w',
        gh=base_grid_H // 2,
        gw=base_grid_W // 2,
    )
    patch_code_scale_2 = torch.clone(patch_code_scale_1)[:, ::2, ::2, :]
    patch_code_scale_2[:, :, :, 0] = 2

    patch_scale_1 = patch_scale_1.to(device)
    patch_scale_2 = patch_scale_2.to(device)

    patch_code_scale_2_selected = patch_code_scale_2[selected_msk_scale_2]
    patch_code_scale_2_selected = rearrange(patch_code_scale_2_selected, '(b np) c -> b np c', b=B)
    patch_scale_2_selected = patch_scale_2[selected_msk_scale_2]
    patch_scale_2_selected = rearrange(patch_scale_2_selected, '(b np) c h w -> b np c h w', b=B)

    patch_code_scale_1_selected = patch_code_scale_1[selected_msk_scale_1]
    patch_code_scale_1_selected = rearrange(patch_code_scale_1_selected, '(b np) c -> b np c', b=B)
    patch_scale_1_selected = patch_scale_1[selected_msk_scale_1]
    patch_scale_1_selected = rearrange(patch_scale_1_selected, '(b np) c h w -> b np c h w', b=B)

    patch_list = [patch_scale_1_selected, patch_scale_2_selected]
    code_list = [patch_code_scale_1_selected, patch_code_scale_2_selected]

    if split_mask is not None:
        p = patch_size // 2
        P = patch_size
        if not split_mask.any():
            patch_scale_0 = images.new_zeros((B, 0, C, p, p), device=device)
            patch_code_scale_0 = images.new_zeros((B, 0, 4), device=device)
        else:
            ks_per = split_mask.view(B, -1).sum(dim=1)
            if ks_per.min() != ks_per.max():
                raise RuntimeError(
                    "split_mask: per-batch True counts differ; vectorized subpatch expects fixed K per sample"
                )
            ks = int(ks_per[0].item())

            patch_grid = rearrange(
                images,
                "b c (nh ph) (nw pw) -> b nh nw c ph pw",
                ph=P,
                pw=P,
            )
            nz = split_mask.nonzero(as_tuple=False)
            b_idx, gh_i, gw_i = nz[:, 0], nz[:, 1], nz[:, 2]
            patches_K = patch_grid[b_idx, gh_i, gw_i]
            sub00 = patches_K[:, :, :p, :p]
            sub01 = patches_K[:, :, :p, p:]
            sub10 = patches_K[:, :, p:, :p]
            sub11 = patches_K[:, :, p:, p:]
            sub4 = torch.stack([sub00, sub01, sub10, sub11], dim=1)
            patch_scale_0 = rearrange(
                sub4, "(b ks) four c p q -> b (ks four) c p q", b=B, ks=ks, four=4
            )

            K = nz.shape[0]
            gh = nz[:, 1].float().unsqueeze(1).expand(-1, 4).contiguous().view(-1)
            gw = nz[:, 2].float().unsqueeze(1).expand(-1, 4).contiguous().view(-1)
            si = (
                torch.arange(4, device=device, dtype=torch.float32)
                .view(1, 4)
                .expand(K, 4)
                .contiguous()
                .reshape(-1)
            )
            z0 = torch.zeros(K * 4, device=device, dtype=torch.float32)
            patch_code_scale_0 = torch.stack([z0, gh, gw, si], dim=1).view(B, ks * 4, 4)
    else:
        p = patch_size // 2
        patch_scale_0 = images.new_zeros((B, 0, C, p, p), device=device)
        patch_code_scale_0 = images.new_zeros((B, 0, 4), device=device)

    patch_code_total = torch.cat([code_list[0], code_list[1], patch_code_scale_0], dim=1)
    return patch_scale_1_selected, patch_scale_2_selected, patch_scale_0, patch_code_total


def patches_to_images(patches, policy_code, grid_size, patch_size, y_scale0=None):
    """
    Strip 仅含 scale1+scale2（高度=P、宽度 n1·P+n2·P）；若含 subpatch，
    将 y_scale0 作为 (B,n0,C,P/2,P/2) 另行传入；否则 n0>0 时段落在 strip 尾部（旧行为）。
    """
    batch_size, dim_patch, ph, pw_tot = patches.size()
    num_grid_h, num_grid_w = grid_size
    num_total_grid = num_grid_h * num_grid_w

    if policy_code.size(-1) < 4:
        z = patches.new_zeros(batch_size, policy_code.size(1), 4 - policy_code.size(-1))
        policy_code = torch.cat([policy_code, z], dim=-1)

    srow = policy_code[0, :, 0]
    n1 = int((srow == 1).sum().item())
    n2 = int((srow == 2).sum().item())
    n0 = int((srow == 0).sum().item())

    P = patch_size
    p = P // 2
    if ph != P:
        raise ValueError(f"strip height {ph} must equal patch_size {P}")
    if y_scale0 is not None:
        tail_w = 0
    else:
        tail_w = n0 * p
    expected_w = n1 * P + n2 * P + tail_w
    if pw_tot != expected_w:
        raise ValueError(f"patches width {pw_tot} != {expected_w} (n1,n2,n0={n1,n2,n0})")

    if n1 > 0:
        y1 = rearrange(patches[:, :, :, : n1 * P], "b c ph (n pw) -> b n c ph pw", n=n1, ph=P, pw=P)
    else:
        y1 = patches.new_zeros(batch_size, 0, dim_patch, P, P)

    if n2 > 0:
        y2 = rearrange(
            patches[:, :, :, n1 * P : n1 * P + n2 * P],
            "b c ph (n pw) -> b n c ph pw",
            n=n2,
            ph=P,
            pw=P,
        )
    else:
        y2 = patches.new_zeros(batch_size, 0, dim_patch, P, P)

    if y_scale0 is not None:
        y0 = y_scale0
        if y0.shape[1] != n0:
            raise ValueError(f"y_scale0 n mismatch {y0.shape[1]} vs n0={n0}")
    else:
        if n0 > 0:
            y0 = rearrange(
                patches[:, :, :, n1 * P + n2 * P :],
                "b c ph (n pw) -> b n c ph pw",
                n=n0,
                ph=p,
                pw=p,
            )
        else:
            y0 = patches.new_zeros(batch_size, 0, dim_patch, p, p)

    code1 = policy_code[:, :n1]
    code2 = policy_code[:, n1 : n1 + n2]
    code0 = policy_code[:, n1 + n2 :]

    patch_scale_1 = y1
    if n1 > 0:
        grid_coord_1 = code1[:, :, 1:3]
    else:
        grid_coord_1 = patches.new_zeros(batch_size, 0, 2, device=patches.device, dtype=torch.long)

    if n2 > 0:
        grid_coord_2 = code2[:, :, 1:3].unsqueeze(2)
        grid_coord_2 = torch.cat(
            [
                grid_coord_2,
                grid_coord_2 + grid_coord_2.new_tensor([[[[0, 1]]]]),
                grid_coord_2 + grid_coord_2.new_tensor([[[[1, 0]]]]),
                grid_coord_2 + grid_coord_2.new_tensor([[[[1, 1]]]]),
            ],
            dim=2,
        )
        grid_coord_2 = rearrange(grid_coord_2, "b np ng c -> b (np ng) c")

        patch_scale_2 = rearrange(y2, "b n c h w -> (b n) c h w")
        patch_scale_2 = F.interpolate(
            patch_scale_2, scale_factor=2, mode="bilinear", align_corners=False,
            recompute_scale_factor=False,
        )
        patch_scale_2 = rearrange(
            patch_scale_2, "bn c (h1 h) (w1 w) -> bn (h1 w1) c h w", h1=2, w1=2
        )
        patch_scale_2 = rearrange(
            patch_scale_2, "(b np) ng c ps_h ps_w -> b (np ng) c ps_h ps_w", b=batch_size
        )
    else:
        grid_coord_2 = patches.new_zeros(batch_size, 0, 2, device=patches.device, dtype=torch.long)
        patch_scale_2 = patches.new_zeros(batch_size, 0, dim_patch, P, P)

    if n0 > 0:
        if n0 % 4 != 0:
            raise ValueError(f"scale0 token count n0={n0} must be divisible by 4")
        ks = n0 // 4
        sub = y0.view(batch_size, ks, 4, dim_patch, p, p)
        row0 = torch.cat([sub[:, :, 0], sub[:, :, 1]], dim=-1)
        row1 = torch.cat([sub[:, :, 2], sub[:, :, 3]], dim=-1)
        patch_fused = torch.cat([row0, row1], dim=-2)
        grid_f = code0[:, 0::4, 1:3].long()
    else:
        patch_fused = patches.new_zeros(batch_size, 0, dim_patch, P, P)
        grid_f = patches.new_zeros(batch_size, 0, 2, device=patches.device, dtype=torch.long)

    patches_uni = torch.cat([patch_scale_1, patch_scale_2, patch_fused], dim=1)
    grid_coord_uni = torch.cat([grid_coord_1, grid_coord_2, grid_f], dim=1)

    grid_uni_value = grid_coord_uni[:, :, 0] * num_grid_w + grid_coord_uni[:, :, 1]
    batch_offset = (
        torch.arange(batch_size, device=grid_uni_value.device, dtype=grid_uni_value.dtype)
        .view(batch_size, 1)
        .expand_as(grid_uni_value)
        * num_total_grid
    )
    grid_sort_global = (batch_offset + grid_uni_value).reshape(-1)
    patch_uni_global = rearrange(patches_uni, "b np c h w -> (b np) c h w")
    patch_uni_global = patch_uni_global[torch.argsort(grid_sort_global)]
    patch_uni_global = rearrange(patch_uni_global, "(b np) c h w -> b np c h w", b=batch_size)
    images = rearrange(
        patch_uni_global, "b (hp wp) c h w -> b c (hp h) (wp w)", hp=num_grid_h, wp=num_grid_w
    )
    return images
