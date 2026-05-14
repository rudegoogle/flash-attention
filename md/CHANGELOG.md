とっとと消せ豚消せ豚# Changelog

Fork-only release history for `ussoewwin/flash-attention`. **Independent** of the upstream Dao-AILab changelog and tags. Upstream changes: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

## v1.1 — 2026-05-13

- **FA2 fork focus:** Keeps investing in the FA2 C++/CUDA stack under `csrc/flash_attn/` while upstream concentrates on FA3 (Hopper) and FA4 (CuTeDSL). Rationale and phased optimizations are tracked in-repo as `AI/FA2_BACKPORT_FROM_FA4_PLAN.md`.
- **Applied FA2 optimizations (plan A-priority):** A-1 running-statistics rescale skip in the softmax path; A-2 `fma.rn.f32x2` packed FMA in `scale_apply_exp2` on `sm_100` and above with fallback on older SMs (`csrc/flash_attn/src/softmax.h`, `csrc/flash_attn/src/flash_fwd_kernel.h`).
- **Build / ABI:** `setup.py` requires `torch>=2.10` on the CUDA wheel path; `csrc/flash_attn/flash_api.cpp` includes `<torch/extension.h>` for the PyTorch 2.10 extension layout.
- **Packaging marker:** `flash_attn.__version__` set to `2.9.0` for this fork’s feature line.
- **Upstream sync:** Merged upstream `main` while retaining fork-specific softmax and build policy (see git history around merge `20314f8` and perf commit `2e3dff7`).
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.1

## v1.0 — 2026-05-13

- Fork baseline; later fork releases append under new headings.
- CUDA install path: `setup.py` sets `install_requires` to include `torch>=2.10`.
- ROCm Triton install path: `install_requires` does not list `torch` (users supply PyTorch separately).
