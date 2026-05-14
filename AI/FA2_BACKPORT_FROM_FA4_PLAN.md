# FA2 Backport-from-FA4 Optimization Plan

| Field | Value |
|---|---|
| Document ID | `FA2_BACKPORT_FROM_FA4_PLAN` |
| Target repo | `D:\USERFILES\fp8e4m3\flash-attention` (fork of `Dao-AILab/flash-attention`) |
| Base version | `flash_attn` `2.9.0` (set in `flash_attn/__init__.py`) |
| Scope | FA2 C++/CUDA kernel only (`csrc/flash_attn/**`). FA3 (`hopper/`) and FA4 (`flash_attn/cute/`) are out of scope for this plan. |
| Build assumptions | PyTorch `>=2.10`, CUDA `>=13.0`, MSVC + nvcc on Windows, default `FLASH_ATTN_CUDA_ARCHS="80;90;100;120"`. |
| Author | This plan is generated to direct further work on the fork. |

---

## 1. Executive Summary

The official `Dao-AILab/flash-attention` repository has effectively ceased active FA2 development and routes all new optimization effort into FA3 (Hopper) and FA4 (CuTeDSL, Blackwell+). FA4 is Linux-only at runtime due to the native CuTeDSL shared-objects shipped only as `manylinux` wheels (`nvidia-cutlass-dsl-libs-base` / `nvidia-cutlass-dsl-libs-cu13`). This fork therefore still has long-term value: an FA2 C++/CUDA kernel that builds natively on Windows with PyTorch 2.10+/CUDA 13.0+.

FA2's kernel is built on `SM80_16x8x16` MMA and `cp.async`, both of which are forward-compatible: the same binary runs on sm_80 / sm_86 / sm_89 / sm_90 / sm_100 / sm_120. It works on the operator's RTX 5060 Ti (sm_120), but does not exploit any Blackwell-only instruction (`fma.f32x2`, `add.f32x2`, `WGMMA`, `TMA`, `UMMA`, `TMEM`).

This plan defines five backport-from-FA4 optimizations, all of which preserve the existing fork's compatibility surface (PyTorch 2.10+, Windows, multi-arch SASS):

| Plan ID | Title | Phase | Status |
|---|---|---|---|
| A-1 | `rescale_threshold` skip in `softmax_rescale_o` | 1 | **Applied** (commit pending) |
| A-2 | `fma.rn.f32x2` packed FMA in `scale_apply_exp2` (sm_100+ guarded) | 1 | **Applied** (commit pending) |
| B-1 | `exp2` polynomial emulation switchover | 2 | Planned |
| B-2 | `MUFU.EX2` + polynomial mixed mode on sm_100+ | 2 | Planned |
| C-1 | hdim-bucketed block-size retuning for sm_120 / sm_100 | 3 | Planned |

A-priority changes are arch-agnostic in their implementation guards: A-1 is enabled by template flag at the call site, A-2 falls back to plain FMA on sm_80/86/89/90 by `__CUDA_ARCH__` guard. No existing call-site outside this fork's `flash_fwd_kernel.h` is forced to change behavior.

---

## 2. Background

### 2.1 Current state of the fork

This fork carries the following deltas against upstream `main`:

- `setup.py`: PyTorch `>=2.10` floor and CUDA 13+ wheel-tag handling.
- `csrc/flash_attn/flash_api.cpp`: `#include <torch/extension.h>` instead of `<torch/python.h>` for PyTorch 2.10 ABI.
- `WindowsWhlBuilder_cuda.bat`: explicit `FLASH_ATTENTION_FORCE_BUILD=TRUE`, `DISTUTILS_USE_SDK=1` and post-build wheel rename.
- `WindowsWhlBuilder_fa4.bat`: FA4 wheel build entry (the produced wheel is non-functional on Windows; retained for symmetry with the upstream layout).
- `flash_attn/__init__.py`: `__version__ = "2.9.0"` to mark the fork-specific feature set.
- `csrc/flash_attn/src/softmax.h`, `csrc/flash_attn/src/flash_fwd_kernel.h`: Plan A-1 / A-2 applied (see §11).

### 2.2 Why FA4 is not a path on Windows

FA4 (`flash_attn/cute/`) is a CuTeDSL package. Its runtime imports `cuda.bindings.driver` and `cutlass.cute` whose native components are distributed only via:

- `nvidia-cutlass-dsl` (pure Python; available)
- `nvidia-cutlass-dsl-libs-base` (`.so`, `manylinux_2_28_x86_64` only)
- `nvidia-cutlass-dsl-libs-cu13` (`.so`, `manylinux_2_28_x86_64` only)

No corresponding `win_amd64` wheel exists, no upstream build script targets Windows, and the underlying libraries are NVIDIA-proprietary. Building equivalents from source is not feasible because the sources are not released. FA4 on Windows is therefore not pursued in this plan.

### 2.3 What FA4 contains that is portable

FA4's CuTeDSL source (`flash_attn/cute/softmax.py`, `flash_attn/cute/flash_fwd_sm100.py`) contains arithmetic-level optimizations that are independent of CuTeDSL infrastructure:

1. Skipping the running-statistics rescale when the new and old row-max are close.
2. Issuing softmax FMA in pairs through `fma.f32x2` on Blackwell.
3. Replacing `MUFU.EX2` with a polynomial approximation, or mixing the two, to gain SFU/FMA pipeline parallelism.
4. A `max_offset` bias on the running max to bound the post-exp range and let downstream FMA stay in a regime that benefits from `f32x2`.

Items (1)-(3) are the basis of plans A-1, A-2, B-1, B-2. Item (4) is folded into B-1 design notes.

---

## 3. Terms and Preconditions

### 3.1 Glossary

| Term | Meaning |
|---|---|
| FA2 | The C++/CUDA kernel under `csrc/flash_attn/`. Default attention impl on sm_80 and above. |
| FA3 | The Hopper-specific kernel under `hopper/`. Uses WGMMA and TMA. |
| FA4 | The CuTeDSL kernel under `flash_attn/cute/`. Targets Blackwell. Out of scope on Windows. |
| sm_NN | CUDA compute capability, e.g. sm_80 = Ampere, sm_90 = Hopper, sm_100 = B200, sm_120 = GeForce Blackwell. |
| MMA | Matrix-multiply-accumulate tensor-core instruction. FA2 uses `SM80_16x8x16_F32F16F16F32_TN`. |
| `fma.rn.f32x2` | Packed FMA on a pair of `f32` lanes in one issue. Blackwell-only (`sm_100+`). |
| `MUFU.EX2` | The SFU `exp2f` instruction. |
| `scaled_diff` | `(prev_row_max - cur_row_max) * softmax_scale_log2`. |
| `scores_scale` | `exp2f(scaled_diff)`. Always `<= 1.0` in normal control flow. |
| Rescale | The multiplication of running `O` and `row_sum` by `scores_scale` performed each non-first attention block. |
| `keep_window_size` / `softcap` / `Is_local` | Existing FA2 feature flags carried in `Flash_fwd_params`. |
| Backport | Reimplementing an FA4 idea in FA2 C++/CUDA without taking on any FA4 runtime dependency. |

### 3.2 Target architectures

| Arch | Code | Carrier hardware | Status in this plan |
|---|---|---|---|
| sm_80 | Ampere A100 | A100, A30 | Supported, no behavior change unless explicitly noted. |
| sm_86 | GA10x | RTX 30xx | Same as sm_80. |
| sm_89 | AD10x | RTX 40xx | Same as sm_80. |
| sm_90 | Hopper | H100, H200 | Same as sm_80 from FA2's standpoint; WGMMA path is FA3's responsibility. |
| sm_100 | Blackwell datacenter | B200 | Receives A-2, B-2 (`__CUDA_ARCH__ >= 1000` guarded code). |
| sm_120 | Blackwell consumer | RTX 50xx incl. RTX 5060 Ti | Receives A-2, B-2 (same guard); receives C-1 retuning. |

### 3.3 Build preconditions

- nvcc that supports `-gencode arch=compute_120,code=sm_120` (CUDA `>= 12.8`; default with CUDA 13.0).
- For `compute_100f` / `compute_120f` family-specific encoding, CUDA `>= 12.9`. The fork's `add_cuda_gencodes()` handles both branches.
- PTX `fma.rn.f32x2` requires `.target sm_100` or above and ptxas from CUDA 12.8+ (CUDA 13.0 used in production here).
- Windows host: MSVC 14.4x with the Windows 10/11 SDK matching the nvcc requirement. Build entry is `WindowsWhlBuilder_cuda.bat`.

---

## 4. Code Analysis

### 4.1 FA2 forward kernel skeleton

The forward kernel template `compute_attn_1rowblock` is instantiated by `flash_fwd_launch_template.h` and lives in `flash_fwd_kernel.h`. The block loop runs:

1. Load `K` tile through `cp.async` into smem.
2. Compute `S = Q * K^T` via `gemm()` over `MMA_M x MMA_N` MMA atoms (using `Mma1::Operation = SM80_16x8x16_F32F16F16F32_TN`).
3. Apply mask (`Mask::apply_mask`).
4. Apply softmax: `softmax_rescale_o<Is_first, Check_inf>(acc_s, acc_o, scale_softmax_log2)`. The `Is_first` flag selects the "no running stats yet" branch.
5. Convert `acc_s` from fp32 to fp16/bf16.
6. Compute `O += P * V` (second `gemm`).
7. Advance to next `K`/`V` tile.

The softmax step is the only one that does floating-point arithmetic outside MMA. It is therefore the only place where FA4's softmax improvements can be backported without rewriting the entire kernel pipeline. The MMA and memory pipelines are intentionally left untouched in this plan; rewriting them would require WGMMA/TMA paths and falls under FA3-style work.

### 4.2 Current softmax implementation (post Plan A)

`csrc/flash_attn/src/softmax.h` now contains, in order:

| Symbol | Purpose | Modified by A-* |
|---|---|---|
| `thread_reduce_`, `reduce_max`, `reduce_sum`, `Allreduce<>` | Warp-level reductions over `kNRows`. | No |
| `fma_f32x2(d0,d1, a0,a1, b0,b1, c0,c1)` (new) | One `fma.rn.f32x2` on sm_100+, two scalar FMAs otherwise. | Added by A-2 |
| `scale_apply_exp2<Scale_max>(tensor, max, scale)` | For each row: compute `exp2(t*scale - max_scaled)` element-wise. | Rewritten by A-2 to use packed FMA on sm_100+ |
| `max_scale_exp2_sum<zero_init>(tensor, max, sum, scale)` | Fused max-reduce + scale + exp + sum-reduce. Used by FA2 bwd. | No (B-2 will revisit). |
| `Softmax<kNRows>::softmax_rescale_o<Is_first,Check_inf,Use_rescale_threshold>` | Per-block softmax + running-stats update. | Extended by A-1 with `Use_rescale_threshold` template flag. |
| `Softmax<kNRows>::normalize_softmax_lse` | Final normalization. | No |

### 4.3 Current MMA and memory instructions

| Concern | Instruction in FA2 | Forward-compat on sm_100/120? |
|---|---|---|
| Q @ K^T (and P @ V) | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (via `SM80_16x8x16_F32F16F16F32_TN`) | Yes, runs as an Ampere-class MMA on Blackwell. |
| Load Q / K / V | `cp.async.cg.shared.global` | Yes. Blackwell adds `TMA` but the `cp.async` path still functions. |
| `exp` | `MUFU.EX2` via `exp2f` intrinsic | Yes. Blackwell still has the SFU. |
| Scalar arithmetic | `FFMA`, `FMUL`, `FADD` | Yes. Packed `fma.f32x2` is Blackwell-only and is opt-in only inside the new `fma_f32x2()` helper. |

There is no instruction in the current FA2 kernel that **cannot** execute on sm_120. The performance gap is purely "instructions used vs. instructions available", not "binary cannot run".

### 4.4 sm_120 (RTX 5060 Ti) runtime evidence

The operator has confirmed the existing FA2 binary runs correctly on RTX 5060 Ti (sm_120). This rules out the earlier mistaken claim that "FA2 is sm_80 only". The default `FLASH_ATTN_CUDA_ARCHS="80;90;100;120"` in `setup.py` emits SASS for all those archs in addition to PTX for the newest one. `add_cuda_gencodes()` chooses `compute_100f` / `compute_120f` when CUDA `>= 12.9` so the kernel is also "family-locked" for forward-compatible PTX-JIT scenarios.

---

## 5. Improvement Catalog

This section is the table-of-contents for §7 (detailed design) and §8 (validation). Each entry is intentionally short here; see §7 for full design.

### 5.1 Plan A-1 — `rescale_threshold` skip

**Location:** `csrc/flash_attn/src/softmax.h::Softmax::softmax_rescale_o` (Is_first=false branch).
**Idea:** When `scaled_diff = (prev_max - cur_max) * log2_scale` is in `[-T, 0]` for small `T`, `scores_scale = exp2(scaled_diff)` is so close to `1.0` that multiplying running `O` and `row_sum` by it is wasted work. Skip the multiply (and the `exp2f`) and keep the new `row_max` for the subsequent `scale_apply_exp2`. `T = 0.01` ⇒ worst case `scores_scale >= 0.993`, relative error `<= 0.7%`.
**Arch effect:** Roughly equal speedup on sm_80/90/100/120 because the saved work (one `exp2f` + N `mul`s) is the same in cycle terms across these arches.
**Risk:** Very mild accuracy regression on the running `O` accumulation, bounded by the threshold.

### 5.2 Plan A-2 — `fma.rn.f32x2` in `scale_apply_exp2` (sm_100+ guard)

**Location:** `csrc/flash_attn/src/softmax.h::scale_apply_exp2`.
**Idea:** The inner loop computes `tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled)` per element. The pre-`exp2f` term is `fma(t, scale, -max_scaled)`. On sm_100+, `fma.rn.f32x2` performs two such FMAs in one instruction. The `exp2f` itself stays scalar (`MUFU.EX2` is not f32x2). Net effect: roughly half the FMA issues for the pre-`exp2f` work; `MUFU.EX2` becomes the dominant cost, opening room for B-2.
**Arch effect:** sm_100/120 only. sm_80/86/89/90 fall back via `#if __CUDA_ARCH__ >= 1000` to the existing scalar code (binary unchanged on those arches).
**Risk:** PTX inline asm needs the correct lane packing. The implementation packs two `f32` into a `.b64` register and unpacks back; the helper is the only place this asm exists.

### 5.3 Plan B-1 — `exp2` polynomial emulation

**Location:** `csrc/flash_attn/src/softmax.h::scale_apply_exp2`, and the matching `max_scale_exp2_sum` for FA2 bwd (`csrc/flash_attn/src/flash_bwd_kernel.h:536`).
**Idea:** Replace `MUFU.EX2` with a fused multiply-add polynomial of degree 3-5 over `[-1, 0]` after a range-reduce step. This trades SFU throughput for FMA throughput. On sm_80/86/89/90 it is a small win (FMA peak is much higher than SFU peak). On sm_100/120 the win can be larger when combined with A-2 because polynomial evaluation maps well to `fma.f32x2`. FA4's `ex2_emulation` in `flash_attn/cute/softmax.py` is the reference.
**Risk:** Accuracy. The polynomial must be evaluated with care on the range that comes out of the `t*scale - max_scaled` step, otherwise softmax tail values can drift. A `max_offset` bias (FA4 idea) helps bound the input range. Plan B-1 only ships behind a template flag (default off) until validated.

### 5.4 Plan B-2 — Mixed `MUFU.EX2` / polynomial mode

**Location:** Same as B-1 but with periodic SFU correction.
**Idea:** Issue the cheap polynomial for `N-1` of every `N` elements and a true `MUFU.EX2` for the `N`-th element. The SFU lane is used to correct accumulated drift. FA4 exposes this as `ex2_emu_freq`. The mixed mode can be cheaper than full polynomial under load (because polynomial occupancy is FMA-bound) and more accurate.
**Risk:** Tuning. The optimal `N` depends on tile size and hdim. Initial cut: `N=4` for hdim 64/128; `N=2` for hdim 256.

### 5.5 Plan C-1 — hdim-bucketed block-size retuning for Blackwell

**Location:** `csrc/flash_attn/src/flash_fwd_launch_template.h` and `csrc/flash_attn/src/kernel_traits.h`.
**Idea:** The current block sizes (kBlockM/kBlockN) and num_warps were tuned for sm_80 (A100) and reused on sm_120. Blackwell consumer parts have a different smem-to-tensor-core ratio. A retune over hdim ∈ {64, 96, 128, 192, 256} should yield single-digit percent improvements on sm_120 without touching kernel logic.
**Risk:** Compile-time blowup if every hdim x arch combination gets a distinct instantiation. Mitigate by limiting the new tuning to sm_100/120 and reusing sm_80 traits on older arches.

---

## 6. Implementation Phases

| Phase | Plans | Gating criterion to enter phase | Gating criterion to exit phase |
|---|---|---|---|
| 0 | Measurement scaffolding | None | `bench/` script that returns numeric latency for hdim ∈ {64,128,256}, seqlen ∈ {1k, 4k, 8k, 16k}, dtype ∈ {fp16, bf16} on sm_80, sm_90, sm_120; baseline recorded. |
| 1 | A-1, A-2 | Phase 0 baseline recorded | Phase 0 numeric regression on sm_80 ≤ 1%; numeric improvement on sm_120 in any cell. |
| 2 | B-1, B-2 | Phase 1 merged | sm_120 improvement on at least one (hdim, seqlen) cell with accuracy drop ≤ 1e-2 (fp16 rel) on a reference attention test set. |
| 3 | C-1 | Phase 2 merged | hdim-bucketed retune yields ≥ 5% on sm_120 for any (hdim, seqlen) cell with no regression on sm_80. |

Phase 1 is already substantially complete (A-1 and A-2 are in the working tree). Phase 0 measurement scaffolding has **not** been built yet and is a prerequisite for objectively gating Phase 2.

### 6.1 Sequencing constraints

- A-2 must land before B-1 / B-2 to give those plans access to the `fma_f32x2()` helper.
- B-2 depends on B-1 (mixed mode reuses the polynomial implementation).
- C-1 is independent and could be done in parallel; sequencing it last reduces conflict surface against B-* changes inside `kernel_traits.h`.

---

## 7. Detailed Design

### 7.1 Plan A-1 detailed design

**Signature change (applied):**

```183:183:d:\USERFILES\fp8e4m3\flash-attention\csrc\flash_attn\src\softmax.h
    template<bool Is_first, bool Check_inf=false, bool Use_rescale_threshold=false, typename Tensor0, typename Tensor1>
```

**Logic change (applied):**

```203:215:d:\USERFILES\fp8e4m3\flash-attention\csrc\flash_attn\src\softmax.h
                float scaled_diff = (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2;
                // Optionally skip the O / row_sum rescale when the new row_max is virtually the same as
                // the previous one (i.e. scaled_diff is a very small negative number, so
                // scores_scale = exp2(scaled_diff) ~= 1.0). The threshold -0.01 corresponds to
                // scores_scale >= ~0.993 (worst-case relative error <= 0.7%).
                // The newly reduced row_max is still used by the upcoming scale_apply_exp2 below,
                // so correctness of the current block is preserved; only the rescale of the running O
                // is approximated. This trades a tiny accuracy hit for one fewer exp2 + N mul per row.
                if constexpr (Use_rescale_threshold) {
                    constexpr float kRescaleSkipThreshold = -0.01f;
                    if (scaled_diff >= kRescaleSkipThreshold) { continue; }
                }
                float scores_scale = exp2f(scaled_diff);
```

**Why correctness is preserved:** the new `row_max` reduced by `reduce_max</*zero_init=*/false>` is written before the threshold check. `scale_apply_exp2(scores, row_max, ...)` runs after the threshold loop and uses that fresh `row_max`. The only thing skipped is the multiplicative correction on the running accumulators `row_sum(mi)` and `acc_o_rowcol(mi, *)`. The skipped factor is `>= 0.993` by construction, so the running `O` is at most `0.7%` off per skipped block, and only on rows where `scaled_diff` actually fell in the skip window. The terminal `normalize_softmax_lse` then divides by `row_sum`, which carries the same skipped factor, so the relative error in the normalized output is bounded by the threshold and does not cascade.

**Threshold choice:** `-0.01` is the most conservative useful value. Stricter (`-0.005` ≈ 0.35%) reduces the hit rate substantially; looser (`-0.05` ≈ 3.4%) crosses what we are willing to call "virtually unchanged". The threshold is `constexpr`, not a runtime parameter, so it is fully constant-folded. Promoting it to a runtime parameter is **not** done in Phase 1 because it would force every call-site to wire an extra argument; consider revisiting at Phase 2 if accuracy ablation indicates a need.

**Call-site changes (applied):** four `Is_first=false` call sites in `flash_fwd_kernel.h` (lines 344, 407, 918, 985) now pass `/*Use_rescale_threshold=*/true`. The two `Is_first=true` call sites (343, 917) do not need the flag because there is no `prev` to compare against.

**Bwd impact:** `flash_bwd_kernel.h` calls `scale_apply_exp2` directly, not `softmax_rescale_o`, so A-1 does not touch the backward path.

**Verification (proposed):**

1. Forward correctness: run `tests/test_flash_attn.py::test_flash_attn_output` with `Use_rescale_threshold=true` and confirm the relative L∞ of the output vs. reference attention is `<= 1e-2` for fp16 and `<= 5e-3` for bf16 over the standard test seqlens.
2. Performance: `bench/benchmark_flash_attention.py` for hdim ∈ {64, 128, 256}, seqlen ∈ {1024, 4096, 8192}. Expect 1-3% wall-clock improvement on sm_80/90/120, more when seqlen is small enough that the rescale loop is a noticeable fraction of the kernel.
3. Ablation: also build with `Use_rescale_threshold=false` (revert the four call-site lines) and confirm bit-identical output to the pre-A-1 binary.

### 7.2 Plan A-2 detailed design

**Helper added (applied):**

```65:88:d:\USERFILES\fp8e4m3\flash-attention\csrc\flash_attn\src\softmax.h
// Packed FMA helper: computes (d0,d1) = (a0*b0+c0, a1*b1+c1).
// On sm_100+ (Blackwell), emits a single fma.rn.f32x2 instruction.
// Falls back to two plain FMAs on older arches and when UNFUSE_FMA is set.
__forceinline__ __device__ void fma_f32x2(
    float &d0, float &d1,
    float a0, float a1,
    float b0, float b1,
    float c0, float c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
    asm("{\n\t"
        ".reg .b64 ra, rb, rc, rd;\n\t"
        "mov.b64 ra, {%2, %3};\n\t"
        "mov.b64 rb, {%4, %5};\n\t"
        "mov.b64 rc, {%6, %7};\n\t"
        "fma.rn.f32x2 rd, ra, rb, rc;\n\t"
        "mov.b64 {%0, %1}, rd;\n\t"
        "}\n"
        : "=f"(d0), "=f"(d1)
        : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
#else
    d0 = fmaf(a0, b0, c0);
    d1 = fmaf(a1, b1, c1);
#endif
}
```

**Loop rewrite (applied):** `scale_apply_exp2` now branches on `__CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)`. The sm_100+ branch hoists `neg_max_scaled = -max_scaled`, iterates `ni` in steps of 2 with a static `N1 = decltype(size<1>(tensor))::value`, calls the helper, then applies scalar `exp2f` to each result. The trailing odd element (if any) falls back to the original scalar formula. The else-branch is the original code, byte-for-byte unchanged.

**Why the existing `UNFUSE_FMA` guard is preserved:** PyTorch issue #121558 documents a case where the compiler-fused FMA changes numerical results vs. a separated `mul` then `sub`. The fix at the time was a macro setting from the PyTorch side. The new packed path explicitly disables itself when `UNFUSE_FMA` is defined, so any caller building under the PyTorch defaults continues to behave as before; the new path is opt-in by toolchain.

**PTX correctness considerations:**

- `fma.rn.f32x2` requires `.target sm_100` and ptxas from CUDA 12.8 or later. Both are guaranteed in the build matrix (`>=13.0` here).
- The `.b64` packing places lane 0 in the low 32 bits and lane 1 in the high 32 bits; the `mov.b64 ra, {%2, %3}` syntax matches that exactly. The unpacking is symmetric.
- The constraint `"=f"` binds a 32-bit float register; nvcc accepts that for the input and output operands of an asm block that internally uses `.b64`.
- Rounding mode `.rn` matches the rounding of plain `fmaf` (which is round-to-nearest-even by default), so there is no rounding-mode mismatch between the two branches on the sm_100+ path's "even" elements and the trailing scalar element.

**Performance expectation:** the pre-`exp2f` arithmetic in `scale_apply_exp2` is `N_rows * N1` FMAs. Halving the FMA count moves the bottleneck firmly to `MUFU.EX2`, which is what B-1/B-2 then target. On sm_120, the expected wall-clock improvement from A-2 alone is in the 1-5% range depending on hdim (higher for larger hdim because the softmax loop is a larger fraction of the kernel).

**Verification (proposed):**

1. Numeric: compare `scale_apply_exp2` output between `__CUDA_ARCH__` branches. The packed path uses two FMAs vs. two separated `mul`/`sub` (in the unfused case) or two FMAs (in the fused case). Round-off may differ by one ULP relative to `mul`/`sub`, but matches the existing `fma`-fused path. Acceptance: matching to `1e-6` relative on synthetic input.
2. SASS inspection: run `cuobjdump --dump-sass` on the sm_120 binary and confirm `FFMA.X2` (packed) appears in the `scale_apply_exp2` region. If it does not appear, the inline asm has been folded out, which is a regression that must be investigated before claiming the optimization is live.
3. Performance: identical script to A-1, focus on sm_120 cells; rebench sm_80 to confirm no regression.

### 7.3 Plan B-1 detailed design

**Goal:** Provide an opt-in polynomial `exp2f` replacement, used in `scale_apply_exp2` and `max_scale_exp2_sum`.

**Design sketch:**

```cpp
// New file or new section in softmax.h.
// Polynomial approximation of exp2(x) over x in [-1, 0].
__forceinline__ __device__ float exp2_poly_m1_0(float x) {
    // x in [-1, 0]
    // p(x) ~= exp2(x) over this range with abs error <= 2e-5
    // Coefficients (placeholder; to be regenerated with Remez/Sollya at impl time):
    const float c0 = 1.0000000e+00f;
    const float c1 = 6.9314718e-01f;
    const float c2 = 2.4022648e-01f;
    const float c3 = 5.5504076e-02f;
    const float c4 = 9.6181294e-03f;
    // Horner with fma:
    float r = c4;
    r = fmaf(r, x, c3);
    r = fmaf(r, x, c2);
    r = fmaf(r, x, c1);
    r = fmaf(r, x, c0);
    return r;
}

// exp2(y) for arbitrary y via range reduce:
//   y = k + f,  k = floor(y), f = y - k, f in [-1, 0] for k = ceil(y)
//   exp2(y) = ldexp(p(f), k)
__forceinline__ __device__ float exp2f_emu(float y) {
    int k = __float2int_rd(y);            // floor as int
    float f = y - __int2float_rn(k);      // f in [0, 1)
    f -= 1.0f;                            // shift to [-1, 0]
    float p = exp2_poly_m1_0(f);          // p ~= exp2(f) in [0.5, 1]
    return __int_as_float((k + 1 + 127) << 23) * p;  // ldexp(p, k+1)
}
```

**Where to plug in:** `scale_apply_exp2` and `max_scale_exp2_sum` get a new template parameter `template<bool Use_exp2_emu=false, ...>`. When `Use_exp2_emu=true`, `exp2f` is replaced by `exp2f_emu`. The flag flows in from `softmax_rescale_o` and is opt-in at call sites in `flash_fwd_kernel.h`.

**Accuracy gating:** the polynomial coefficients above are placeholders. Before merge, regenerate via Remez fit on `[-1, 0]` for degree 4 (we already see fp16 attention is robust to ~2e-4 relative softmax error). The acceptance threshold for shipping `Use_exp2_emu=true` by default is: per-element relative error of softmax output `<= 1e-3` and end-to-end logits relative error `<= 1e-2` on a fixed reference workload.

**Arch interaction:** the polynomial body is pure FMA. On sm_100/120, the body benefits implicitly from A-2's `fma_f32x2` if we route the polynomial through `fma_f32x2`. That cross-plan synergy is in scope for B-1's implementation.

**Why not always-on:** SFU has dedicated throughput; on memory-bound kernels (small seqlen, large hdim) the SFU is not the bottleneck and emulation is pure loss. The opt-in flag avoids regressing those cases.

### 7.4 Plan B-2 detailed design

**Goal:** Mix `MUFU.EX2` and polynomial within a single row to use both the SFU and FMA pipelines.

**Design sketch:** the inner loop in `scale_apply_exp2` becomes (conceptually):

```cpp
for (int ni = 0; ni < N1; ++ni) {
    if (ni % EmuFreq == 0) {
        tensor(mi, ni) = exp2f(t);              // real MUFU.EX2 every EmuFreq elements
    } else {
        tensor(mi, ni) = exp2f_emu(t);          // polynomial otherwise
    }
}
```

`EmuFreq` is a new `int` template parameter, default 0 (meaning "all SFU, never emulate" = current behavior). Setting `EmuFreq = 1` is equivalent to B-1 (all-emulation). Setting `EmuFreq = 4` issues SFU on every 4th element.

**Interaction with A-2:** When `EmuFreq >= 2`, group the inner loop into chunks of 2 and use `fma_f32x2` for the polynomial FMAs. The SFU-element gets handled scalarly. The cost model is then:

- 1 `MUFU.EX2` per `EmuFreq` elements
- `(EmuFreq - 1) * polynomial_FMAs / 2` packed FMAs per `EmuFreq` elements (only on sm_100+)
- `(EmuFreq - 1)` polynomial FMAs per `EmuFreq` elements (on older arches)

**Tuning surface:** `EmuFreq ∈ {0, 2, 4, 8}` is the search space. Recommended initial defaults on sm_100/120: `EmuFreq = 4` for hdim ≤ 128, `EmuFreq = 2` for hdim ≥ 192. These will be revised after Phase 2 measurement.

**Risk:** if `EmuFreq` is set very low on sm_80 (no `fma.f32x2`), the polynomial cost overrides the SFU savings. Default to `EmuFreq = 0` on sm_80 by guarding the template parameter behind the `__CUDA_ARCH__ >= 1000` selection at call-site instantiation.

### 7.5 Plan C-1 detailed design

**Goal:** Add a Blackwell-specific tuning table in `kernel_traits.h` and `flash_fwd_launch_template.h` that selects `kBlockM`, `kBlockN`, and `kNWarps` based on `hdim` and target arch.

**Mechanism:**

1. In `kernel_traits.h`, introduce a new struct family `Flash_fwd_kernel_traits_blackwell<headdim, ...>` that mirrors `Flash_fwd_kernel_traits<headdim, ...>` but with tuned constants.
2. In `flash_fwd_launch_template.h`, dispatch at the macro level:
   ```cpp
   #if defined(FLASH_FWD_USE_BLACKWELL_TRAITS) && __CUDA_ARCH__ >= 1000
       using Kernel_traits = Flash_fwd_kernel_traits_blackwell</*hdim=*/Headdim, /*...*/>;
   #else
       using Kernel_traits = Flash_fwd_kernel_traits</*hdim=*/Headdim, /*...*/>;
   #endif
   ```
3. The Blackwell traits will deviate from the Ampere traits primarily in `kBlockN` (typically larger to exploit additional smem) and `kNWarps` (often 8 on Hopper-class, depending on hdim).

**Tuning method:** sweep on a single sm_120 device (RTX 5060 Ti), record `bench/benchmark_flash_attention.py` median over 100 runs for each `(hdim, seqlen, dtype, kBlockM, kBlockN, kNWarps)` cell, pick the Pareto-best per `(hdim, seqlen, dtype)`. Compilation cost is bounded because only one new traits family is added (not a Cartesian explosion).

**Risk and mitigation:** compile-time grows by roughly the number of new explicit instantiations. Limit explicit instantiation to hdim ∈ {64, 96, 128, 192, 256} (the existing supported set), and gate the new traits behind a build flag (`FLASH_FWD_USE_BLACKWELL_TRAITS`) initially. Default-on the flag only after C-1 produces a confirmed ≥ 5% improvement on the target hardware.

---

## 8. Validation Plan

### 8.1 Accuracy

| Test | Tool | Metric | Pass threshold |
|---|---|---|---|
| Forward exact-vs-reference | `tests/test_flash_attn.py::test_flash_attn_output` | L∞ relative vs. reference attention | fp16: `<= 1e-2`, bf16: `<= 5e-3` |
| Backward exact-vs-reference | `tests/test_flash_attn.py::test_flash_attn_backward` | L∞ relative vs. reference | fp16: `<= 1.5e-2`, bf16: `<= 7e-3` |
| Endpoint logits (downstream) | Custom script: load a reference LLM, run a fixed prompt, compare attention path outputs | per-token cosine | `>= 0.9995` |
| Ablation A-1 off | Rebuild with `Use_rescale_threshold=false` at all 4 call sites | Bit-identical output vs. pre-A-1 binary | exact |
| Ablation A-2 off | Build with `UNFUSE_FMA` defined | Output relative L∞ within `1e-6` of A-2-on (sm_100/120 only) | `1e-6` |

### 8.2 Performance

| Test | Tool | Cells | Measurement |
|---|---|---|---|
| Forward latency | `bench/benchmark_flash_attention.py` (existing) | hdim ∈ {64, 96, 128, 192, 256}, seqlen ∈ {1024, 2048, 4096, 8192, 16384}, dtype ∈ {fp16, bf16}, causal ∈ {true, false} | median wall-clock over 100 runs after 20 warmup |
| SASS inspection | `cuobjdump --dump-sass` | `_ZN12FLASH_NAMESPACE16scale_apply_exp2*` symbols | manual confirmation of `FFMA.X2` on sm_120, absence on sm_80 |
| Compile time | `time WindowsWhlBuilder_cuda.bat` | full build | no more than 110% of pre-change baseline |
| Wheel size | filesystem | `dist\flash_attn-2.9.0+...whl` | no more than 110% of pre-change baseline |

### 8.3 Hardware matrix

| Arch | Hardware available to operator | Validation status |
|---|---|---|
| sm_80 | None confirmed | Functional regression test (via PTX-JIT or remote rental) |
| sm_86 | None confirmed | Same as sm_80 |
| sm_89 | None confirmed | Same as sm_80 |
| sm_90 | None confirmed | Same as sm_80 |
| sm_100 | None confirmed | Same as sm_80; PTX-JIT verified `fma.rn.f32x2` emission via `nvdisasm` |
| sm_120 | RTX 5060 Ti 16GB | Primary validation target |

The primary measurement device is the operator's RTX 5060 Ti. For sm_80 / sm_90 confirmation, a reduced cross-check is acceptable: build for those arches, JIT-load on the local device through PTX-JIT, run a forward pass, and confirm shape/finiteness. Full performance numbers on those arches are out of scope unless a corresponding device becomes available.

### 8.4 Benchmark scenarios

Each phase exit gate uses the following baseline harness, defined here once so subsequent phases reuse it.

```python
# bench/fa2_baseline.py (to be added in Phase 0)
import torch
from flash_attn import flash_attn_func
from triton.testing import do_bench

def cell(batch, seqlen, nheads, hdim, dtype, causal):
    q = torch.randn(batch, seqlen, nheads, hdim, dtype=dtype, device='cuda')
    k = torch.randn_like(q); v = torch.randn_like(q)
    fn = lambda: flash_attn_func(q, k, v, causal=causal)
    ms = do_bench(fn, warmup=20, rep=100)
    return ms

for hdim in [64, 96, 128, 192, 256]:
    for seqlen in [1024, 2048, 4096, 8192, 16384]:
        for dtype in [torch.float16, torch.bfloat16]:
            for causal in [False, True]:
                print(hdim, seqlen, dtype, causal, cell(2, seqlen, 16, hdim, dtype, causal))
```

This is a placeholder; the Phase 0 task is to commit a finalized version of this script under `bench/` and record its output as the baseline JSON before any of A-1 / A-2 are claimed measured.

---

## 9. Risks and Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | A-2 inline asm fails to compile on some nvcc version | Low | Build break | The `__CUDA_ARCH__` guard ensures the asm is only seen by nvcc when it's actually targeting sm_100+. CUDA 13.0 ptxas accepts `fma.rn.f32x2` for `.target sm_100`. The fallback path is unconditional plain FMA. |
| R2 | A-1 accuracy drop is larger than the threshold suggests on a real workload | Medium | Functional regression | The threshold is `constexpr` and can be tightened (e.g. to `-0.005`) in a follow-up commit. The flag is opt-in per call site, so reverting at any single call site is one-line. |
| R3 | `fma.rn.f32x2` constraint mismatch leaves the asm unscheduled, defeating the optimization silently | Medium | No perf gain on sm_100+, but no correctness issue | The SASS inspection step in §8.2 catches this. Add it as a hard gate before declaring A-2 "live". |
| R4 | Polynomial coefficients in B-1 are tuned for one hdim and degrade on another | Medium | Accuracy regression | Tune coefficients on a representative range covering all supported hdims; validate per-hdim. |
| R5 | C-1 retuning explodes compile time on Windows | Medium | Developer-experience regression | Limit to one new traits family; gate via `FLASH_FWD_USE_BLACKWELL_TRAITS`. |
| R6 | Cursor's `StrReplace` against externally-edited files causes a cache-divergence loss | Medium | Code loss | All large edits use `Read` immediately before `StrReplace`, and pre/post sha256 + byte-size are recorded for every change. Per `no-tool-blaming-and-mixed-editing-caches.mdc`, `launch_utils.py`-style files are not touched here. |
| R7 | A merge from upstream introduces a conflict against the new `Use_rescale_threshold` template parameter | Low-Medium | Manual merge needed | The template parameter has a default value of `false`, so any unchanged upstream call site continues to compile. Only the fork's four call sites need to be re-applied if a merge undoes them. |

---

## 10. Rollback Procedure

This plan's changes are entirely additive at the source level, so rollback is a localized revert per plan.

### 10.1 Roll back A-1

1. In `csrc/flash_attn/src/flash_fwd_kernel.h`, remove `/*Use_rescale_threshold=*/true` from the four `softmax_rescale_o` call sites (lines 344, 407, 918, 985 in the post-A-1 file).
2. In `csrc/flash_attn/src/softmax.h::softmax_rescale_o`, remove the `bool Use_rescale_threshold=false` template parameter and the `if constexpr (Use_rescale_threshold) { ... continue; }` block.
3. Rebuild via `WindowsWhlBuilder_cuda.bat`. The resulting binary is bit-identical to the pre-A-1 binary modulo build timestamp.

### 10.2 Roll back A-2

1. In `csrc/flash_attn/src/softmax.h`, delete the `fma_f32x2` helper (lines 65-88 in the post-A-2 file).
2. Restore the original `scale_apply_exp2` body (single scalar loop with the existing `UNFUSE_FMA` macro split). The pre-A-2 version is preserved in the backup at `D:\USERFILES\_backups\flash-attention_20260513_092006\softmax.h.orig` and corresponds to sha256 `A734A5D857D59D2892BBEEEFD337B880E0BF69DB2C2460C3CFC91A5B0A112D67`.
3. Rebuild.

### 10.3 Roll back the 2.9.0 version bump

1. In `flash_attn/__init__.py`, change `__version__ = "2.9.0"` back to `"2.8.4"`.
2. Rebuild and rename the wheel accordingly.

### 10.4 Roll back B-* and C-1

To be specified at the implementation commit; the same additive pattern is to be followed. B-* changes must be opt-in via template parameter so that rollback is a single-line change at call sites.

---

## 11. Applied Change History

### 11.1 2026-05-13 — Plan A-1 applied

| File | Pre sha256 | Post sha256 | Pre size | Post size | Backup |
|---|---|---|---|---|---|
| `csrc/flash_attn/src/softmax.h` | `A734A5D857D59D2892BBEEEFD337B880E0BF69DB2C2460C3CFC91A5B0A112D67` | `BF7F98F431EC562E66293CEB080CB2C938BC40F48D756CF001A2885A9644B325` | 9653 | 12386 | `D:\USERFILES\_backups\flash-attention_20260513_092006\softmax.h.orig` |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | `94BC41E01E5C223FCB8DD5ABECD4B8CF158F455202FC15E19FD8B75D0DA8E76F` | `31728AD35351A6430A78A13AA2B497E304241B3E5B2571B45556A9A6DA2D8A64` | 78019 | 78147 | `D:\USERFILES\_backups\flash-attention_20260513_092006\flash_fwd_kernel.h.orig` |

Net source delta: 2 files changed, 71 insertions(+), 13 deletions(-).

Call-site changes (Plan A-1): four `Is_first=false` invocations of `softmax_rescale_o` in `flash_fwd_kernel.h` now pass `/*Use_rescale_threshold=*/true`. The two `Is_first=true` invocations are unchanged.

### 11.2 2026-05-13 — Plan A-2 applied

Same files as Plan A-1 (A-1 and A-2 were applied in the same edit session). The `fma_f32x2` helper was added at lines 65-88 of `softmax.h`, and the `scale_apply_exp2` body was rewritten to branch on `__CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)`.

### 11.3 2026-05-13 — Version bump

`flash_attn/__init__.py` line 6: `__version__ = "2.8.4"` → `"2.9.0"`. Pre sha256 `0F7D432F9D94FB89A30967DC2463FCACF683E9F69BA501BAE9D8C1107687D21D`, post sha256 `5F3E25933EF2DE21111F69FE46CA61756C92E3E617F379BA196D5FA2BCCDAECE`. File size unchanged at 470 bytes (both literals are 5 characters). `setup.py::get_package_version()` confirmed to return `"2.9.0"`.

Backup: `D:\USERFILES\_backups\flash-attention_20260513_092006\__init__.py.orig`.

### 11.4 Pending

- Phase 0 measurement harness (`bench/fa2_baseline.py`)
- Plan B-1 commit
- Plan B-2 commit
- Plan C-1 commit
- Validation results table (per §8) populated against the operator's RTX 5060 Ti

---

## 12. References

### 12.1 In-repo

- `csrc/flash_attn/src/softmax.h` — primary edit target for A-* and B-*.
- `csrc/flash_attn/src/flash_fwd_kernel.h` — call sites of `softmax_rescale_o`.
- `csrc/flash_attn/src/flash_bwd_kernel.h` — call site of `scale_apply_exp2` for the backward path.
- `csrc/flash_attn/src/kernel_traits.h` — block-size and warp-count traits (target of C-1).
- `csrc/flash_attn/src/flash_fwd_launch_template.h` — dispatch site (target of C-1).
- `flash_attn/cute/softmax.py` — FA4 reference for A-1, A-2, B-1, B-2 ideas.
- `flash_attn/cute/flash_fwd_sm100.py` — FA4 reference for sm_100-specific kernel structure (background; not directly portable).
- `setup.py` — `cuda_archs()`, `add_cuda_gencodes()`, `get_package_version()` (Phase 0 base).
- `WindowsWhlBuilder_cuda.bat` — build entry.
- `AI/SM90_BLOCK_SIZE_TUNING.md` — adjacent block-size tuning document; methodology reference for C-1.
- `AI/SASS_MMA_ANALYSIS.md` — adjacent SASS-inspection document; methodology reference for §8.2.

### 12.2 PTX / hardware

- PTX ISA, section "Floating-Point Instructions / fma" — `fma.rn.f32x2` operand and target constraints.
- PTX ISA, section "Special Function Instructions" — `MUFU.EX2` (`ex2.approx.f32`) timing background.
- NVIDIA Blackwell Tuning Guide — sm_100 / sm_120 SMEM and tensor-core ratios (used as input to C-1).

### 12.3 External

- `https://github.com/pytorch/pytorch/issues/121558` — origin of the `UNFUSE_FMA` macro; preserved in A-2's guard.
- `https://github.com/Dao-AILab/flash-attention` — upstream FA2/FA3/FA4.
