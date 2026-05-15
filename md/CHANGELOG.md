# Changelog

Fork-only release history for `ussoewwin/flash-attention`. **Independent** of the upstream Dao-AILab changelog and tags. Upstream changes: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

## v1.3 — 2026-05-15

- **Summary:** Fork release **v1.3** / package **`flash_attn` 2.9.0** — **CUDA 13.2** Windows wheel port on top of v1.2 kernels. Target stack: PyTorch **`2.12.0+cu132`**, CUDA toolkit **13.2** (`CUDA_HOME` → `v13.2`). Build fixes: MSVC **`/Zc:preprocessor`** for CCCL on CUDA 13.2; default **`FLASH_ATTN_CUDA_ARCHS=80;90;100;120`** (no Thor **`sm_110`** / `sm_101` gencode). Reference wheel tag: **`cu132torch2.12.0`**. Fork smoke tests live under **`tests/`** (`test_a2_smoke.py`, `test_triton_smoke.py`). Full port log: `md/CUDA_13.0_TO_13.2_BUILD_FIX.md`; validation: `md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.3

## v1.2 — 2026-05-15

- **Summary:** Fork release **v1.2** / package **`flash_attn` 2.9.0** — FA2 CUDA for **sm_80+** on Windows (PyTorch **2.10+**, CUDA **13+**). Forward backports **A-1** (rescale threshold skip in `softmax_rescale_o`) and **A-2** (packed `fma.rn.f32x2` in `scale_apply_exp2` on sm_100 / sm_120). Also: Triton `flash_attn_triton.py` fix; split-KV launch countermeasure 1 in `flash_fwd_launch_template.h`. Multi-arch wheel build default **`FLASH_ATTN_CUDA_ARCHS=80;90;100;120`** only (Ampere, Hopper, Blackwell datacenter/consumer — **no Thor / sm_101 / sm_110**); see `setup.py` `cuda_archs()` / `add_cuda_gencodes()`. Details: `md/FA2_CHANGES_v1.2.md`, `md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.2

## v1.1 — 2026-05-13

- **Summary:** Fork **v2.8.4** vs **v2.8.3** — upstream v2.9.0-equivalent merge + fork-only fixes (split-KV `num_splits==1` branch removed for `sm_80` ptxas stability; CUTLASS bump). Details: `md/v2.8.4_WHAT_CHANGED.md`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.1

## v1.0 — 2026-05-13

- Fork baseline; later fork releases append under new headings.
- CUDA install path: `setup.py` sets `install_requires` to include `torch>=2.10`.
- ROCm Triton install path: `install_requires` does not list `torch` (users supply PyTorch separately).
