# DMTR-PTR

Official code for **DMTR-PTR**: Dynamic Multi-scale Token Resampling with Progressive Token Routing for efficient satellite-to-radar reconstruction.

> Paper citation and preprint link will be updated when available.

## Highlights

- **DMTR**: PolicyNet-guided merge / split to form multi-scale tokens (base / super / sub).
- **PTR**: Progressive routing across transformer layers (default **2-1-1** schedule on a depth-4 backbone).
- Built-in **random dataloader** (`4×192×384` → `1×192×384`) so the training loop can be smoke-tested without external data.

## Repository layout

```text
DMTR_PTR-github/
├── configs/                 # Paper main hyper-parameters
├── scripts/                 # Train / eval / smoke helpers
├── src/
│   ├── main.py
│   ├── trainer.py
│   ├── dataloader_down4.py  # Synthetic random tensors
│   ├── metrics_local.py
│   ├── vit/                 # DMTR-PTR model
│   └── policynet/           # PolicyNet (CTS, Apache-2.0)
├── assets/                  # Optional figures
└── docs/                    # Data notes
```

This release includes the **main model only** (no ablation variants). **Pretrained weights are not redistributed.**

## Requirements

```bash
pip install -r requirements.txt
```

## Smoke test (random data)

```bash
bash scripts/smoke_random.sh
```

This trains for a few epochs on synthetic batches with `--disable-merge` (no PolicyNet checkpoint required).

## Train (paper hyper-parameters, random data)

```bash
bash scripts/train_main.sh
```

When you have a PolicyNet checkpoint and real data integration, remove `--disable-merge` and pass `--policynet-path <checkpoint>`. Real data access follows the **SRViT** release (**link TBD**); see [`docs/DATA.md`](docs/DATA.md).

## Paper main configuration

| Setting | Value |
|--------|--------|
| Model | `vit_dmtr_ptr` |
| Patch size | 4 |
| Depth / dim / heads / mlp | 4 / 64 / 8 / 128 |
| PTR schedule | `--use-early-bypass` + `--enable-base-bypass` (layers 3 / 4) |
| `base-bg-threshold` | 0.98 |
| `base-keep-min-ratio` | 0.4 |
| `bg-merge-ratio` / `split-ratio` | 0.4 / 0.05 |
| Aux BCE weight | 0.1 |
| Loss | `genexp` |

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

PolicyNet is adapted from CTS (CVPR 2023), Apache-2.0 (`src/policynet/LICENSE`).

## Citation

```bibtex
@inproceedings{TODO_dmtr_ptr,
  title     = {TODO: DMTR-PTR paper title},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```
