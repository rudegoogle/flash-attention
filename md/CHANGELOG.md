# Changelog

Fork-only release history for `ussoewwin/flash-attention`. **Independent** of the upstream Dao-AILab changelog and tags. Upstream changes: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

## v1.1 — 2026-05-13

- **Summary:** Fork **v2.8.4** vs **v2.8.3** — upstream v2.9.0-equivalent merge + fork-only fixes (split-KV `num_splits==1` branch removed for `sm_80` ptxas stability; CUTLASS bump). Details: `md/v2.8.4_WHAT_CHANGED.md`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.1

## v1.0 — 2026-05-13

- Fork baseline; later fork releases append under new headings.
- CUDA install path: `setup.py` sets `install_requires` to include `torch>=2.10`.
- ROCm Triton install path: `install_requires` does not list `torch` (users supply PyTorch separately).
