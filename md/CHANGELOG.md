# Changelog

Fork-only release history for `ussoewwin/flash-attention`. **Independent** of the upstream Dao-AILab changelog and tags. Upstream changes: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

## v1.4 — 2026-05-23

- **Summary:** Fork release **v1.4** / package **`flash_attn` 2.9.1** — patch release on the 2.9.0 feature line (Plan A-1 / A-2). **A-1:** file-scope `kSoftmaxRescaleSkipThreshold` and skip rescale in `softmax_rescale_o` when `scaled_diff > -0.01f`, with `static_assert(!(Is_first && Use_rescale_threshold))`. **A-2:** `scale_apply_exp2` uses packed `fma.rn.f32x2` on sm_100+ via `fma_f32x2` in `utils.h`, `static_assert(N1 % 2 == 0)`, optional `#pragma message` when `UNFUSE_FMA` is set. SASS gate check: `bench/check_sass_gates.py`. Same multi-arch wheel policy as v1.2 (**`80;90;100;120`**). GitHub Release tag and wheel: *to be published*.
- **Release notes:** [FA2 2.9.0 → 2.9.1](FA2_2.9.0_to_2.9.1_RELEASE.md)

## v1.3 — 2026-05-15

- **Summary:** Fork release **v1.3** — Windows FA2 **CUDA 13.2** build (PyTorch **2.12.0+cu132**, CUDA toolkit **13.2**, MSVC `/Zc:preprocessor`, default arch **`80;90;100;120`**, wheel tag **`cu132torch2.12.0`**). Smoke tests under `tests/`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.3

## v1.2 — 2026-05-15

- **Summary:** Fork release **v1.2** / package **`flash_attn` 2.9.0** — FA2 CUDA for **sm_80+** on Windows (PyTorch **2.10+**, CUDA **13+**). Forward backports **A-1** (rescale threshold skip in `softmax_rescale_o`) and **A-2** (packed `fma.rn.f32x2` in `scale_apply_exp2` on sm_100 / sm_120). Also: Triton `flash_attn_triton.py` fix; split-KV launch countermeasure 1 in `flash_fwd_launch_template.h`. Multi-arch wheel build default **`FLASH_ATTN_CUDA_ARCHS=80;90;100;120`** only (Ampere, Hopper, Blackwell datacenter/consumer — **no Thor / sm_101 / sm_110**); see `setup.py` `cuda_archs()` / `add_cuda_gencodes()`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.2

## v1.1 — 2026-05-13

- **Summary:** Fork **v2.8.4** vs **v2.8.3** — upstream v2.9.0-equivalent merge + fork-only fixes (split-KV `num_splits==1` branch removed for `sm_80` ptxas stability; CUTLASS bump).
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.1

## v1.0 — 2026-05-13

- Fork baseline; later fork releases append under new headings.
- CUDA install path: `setup.py` sets `install_requires` to include `torch>=2.10`.
- ROCm Triton install path: `install_requires` does not list `torch` (users supply PyTorch separately).
