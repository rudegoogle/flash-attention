# Changelog

Fork-only release history for `ussoewwin/flash-attention`. **Independent** of the upstream Dao-AILab changelog and tags. Upstream changes: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

## v1.2 — 2026-05-15

Fork release **v1.2** ships Python package **`flash_attn` 2.9.0** — the first fork feature line after upstream shifted primary development to FA3/FA4. This release keeps the **FA2 CUDA** path viable on **Windows** and **sm_80+** (Ampere through Blackwell) with **PyTorch 2.10+** and **CUDA 13+**.

### Overview (2.9.0)

- **Package version:** `flash_attn/__init__.py` → **2.9.0** (fork marker; independent of upstream Dao-AILab package numbering).
- **FA2 kernel backports (forward):**
  - **A-1** — optional rescale skip in `softmax_rescale_o` when row-max change is below threshold (`Use_rescale_threshold`, four call sites in `flash_fwd_kernel.h`).
  - **A-2** — packed `fma.rn.f32x2` pre-`exp2f` path in `scale_apply_exp2` on **`sm_100` / `sm_120`** (`__CUDA_ARCH__ >= 1000`).
- **Other 2.9.0 fork fixes (see validation guide):** Triton `flash_attn_triton.py` compatibility fix; split-KV launch **countermeasure 1** (`num_splits == 1` branch removed in `flash_fwd_launch_template.h` for `sm_80` `ptxas` stability).
- **Build / install:** CUDA `install_requires` includes **`torch>=2.10`**; extension uses `<torch/extension.h>`; default `FLASH_ATTN_CUDA_ARCHS="80;90;100;110;120"`.

### Documentation

- Kernel changes (A-1 / A-2): [md/FA2_CHANGES_v1.2.md](FA2_CHANGES_v1.2.md)
- Test and validation record: [md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md](2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md)

### Release

- **Release notes (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.2

## v1.1 — 2026-05-13

- **Summary:** Fork **v2.8.4** vs **v2.8.3** — upstream v2.9.0-equivalent merge + fork-only fixes (split-KV `num_splits==1` branch removed for `sm_80` ptxas stability; CUTLASS bump). Details: `md/v2.8.4_WHAT_CHANGED.md`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.1

## v1.0 — 2026-05-13

- Fork baseline; later fork releases append under new headings.
- CUDA install path: `setup.py` sets `install_requires` to include `torch>=2.10`.
- ROCm Triton install path: `install_requires` does not list `torch` (users supply PyTorch separately).
