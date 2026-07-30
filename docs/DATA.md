# Data

This release ships a **synthetic random dataloader** only:

| Tensor | Shape |
|--------|--------|
| Input `x` | `(4, 192, 384)` |
| Target `t` | `(1, 192, 384)` |

Values are uniform in `[0, 1]`. See `src/dataloader_down4.py`.

Real satellite / radar data is **not** redistributed here. Refer to the related **SRViT** paper / code release once it is public (**link TBD**).
