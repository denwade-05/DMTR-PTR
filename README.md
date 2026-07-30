# DMTR-PTR

Official code for **DMTR-PTR: Distribution-Aware Token Resampling and Routing for Large-Scale Satellite-to-Radar Reconstruction**.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/smoke_random.sh
```

This runs a short training loop on **synthetic random tensors** (input `4×192×384`, target `1×192×384`) so the pipeline can be verified without external data.

## Data

Raw satellite / radar data follows [SRViT](https://github.com/stockeh/srvit).  
We apply spatial downsampling; the training resolution used in this work is **192×384**.  
Real-data loaders will be added later; the public repo currently ships the random dataloader for a runnable demo.

## Acknowledgements

This codebase builds on:

- Training / ViT framework from [stockeh/srvit](https://github.com/stockeh/srvit)
- Mixed-resolution tokenization ideas from [TomerRonen34/mixed-resolution-vit](https://github.com/TomerRonen34/mixed-resolution-vit)
- PolicyNet from CTS (CVPR 2023; see `src/policynet/`)

## Citation

TODO

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
