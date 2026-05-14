# FA2 Kernel Changes in v1.1

This document explains every code change applied to the FA2 (CUDA) path in this fork. The upstream repository has shifted focus to FA3 (Hopper) and FA4 (CuTeDSL); this fork continues FA2 development for sm_80+ on Windows with PyTorch 2.10+ / CUDA 13+.

## Files modified

| File | Purpose |
|---|---|
| `csrc/flash_attn/src/softmax.h` | Core softmax math: rescale logic and `exp2` scaling |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | Kernel dispatch: adds template flags to call sites |
| `flash_attn/__init__.py` | Version marker for the fork feature line |

---

## 1. A-1 — Rescale threshold skip (`softmax_rescale_o`)

### What it does
When a new block’s row-max is virtually identical to the running row-max, the rescale factor `exp2(scaled_diff)` is approximately `1.0`. Skipping the rescale saves one `exp2` and `N` multiplies per row.

### Why it is safe
The threshold (`-0.01f`) corresponds to `scores_scale >= ~0.993`, i.e. a worst-case relative error below `0.7%`. The current block still uses the correct `row_max` for its own `scale_apply_exp2`; only the running `O`/`row_sum` rescale is approximated.

### Changed code

#### `csrc/flash_attn/src/softmax.h` — `Softmax::softmax_rescale_o` signature

```cpp
// Before:
template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>

// After (adds Use_rescale_threshold):
template<bool Is_first, bool Check_inf=false, bool Use_rescale_threshold=false, typename Tensor0, typename Tensor1>
```

#### `csrc/flash_attn/src/softmax.h` — rescale logic body

```cpp
float scaled_diff = (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2;
// Optionally skip the O / row_sum rescale when the new row_max is virtually
// the same as the previous one (scaled_diff is a very small negative number,
// so scores_scale = exp2(scaled_diff) ~= 1.0). The threshold -0.01 corresponds
// to scores_scale >= ~0.993 (worst-case relative error <= 0.7%).
if constexpr (Use_rescale_threshold) {
    constexpr float kRescaleSkipThreshold = -0.01f;
    if (scaled_diff >= kRescaleSkipThreshold) { continue; }
}
float scores_scale = exp2f(scaled_diff);
row_sum(mi) *= scores_scale;
// ... acc_o_rowcol(mi, ni) *= scores_scale;
```

---

## 2. A-2 — Packed FMA via `fma.rn.f32x2` (`scale_apply_exp2`)

### What it does
On `sm_100` and newer (Blackwell), the compiler can emit a single `fma.rn.f32x2` instruction that computes two FMAs in parallel. On older architectures the helper falls back to two plain `fmaf` calls.

### New helper — `fma_f32x2`

```cpp
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

### Changed code — `scale_apply_exp2`

```cpp
// Added near the top of scale_apply_exp2:
constexpr int N1 = decltype(size<1>(tensor))::value;

// Inside the mi loop, after computing max_scaled and neg_max_scaled:
const float neg_max_scaled = -max_scaled;
#pragma unroll
for (int ni = 0; ni + 1 < N1; ni += 2) {
    float t0, t1;
    FLASH_NAMESPACE::fma_f32x2(t0, t1,
        tensor(mi, ni), tensor(mi, ni + 1),
        scale, scale,
        neg_max_scaled, neg_max_scaled);
    tensor(mi, ni)     = exp2f(t0);
    tensor(mi, ni + 1) = exp2f(t1);
}
if constexpr (N1 % 2 != 0) {
    constexpr int ni = N1 - 1;
    tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
}
```

The old single-element loop is preserved under the `#else` branch for `sm_80`/`sm_90` and when `UNFUSE_FMA` is defined.

---

## 3. Kernel dispatch changes (`flash_fwd_kernel.h`)

Four call sites pass the new template argument `/*Use_rescale_threshold=*/true` to `softmax_rescale_o`:

1. `compute_attn_1rowblock`, non-first masking step (causal/local)
2. `compute_attn_1rowblock`, local-only loop
3. `compute_attn_1rowblock_splitkv`, non-first masking step (causal/local/even-MN)
4. `compute_attn_1rowblock_splitkv`, local-only loop

Example (site 1):

```cpp
// Before:
softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(...)

// After:
softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local, /*Use_rescale_threshold=*/true>(...)
```

---

## 4. Version bump

`flash_attn/__init__.py`:

```python
-__version__ = "2.8.4"
+__version__ = "2.9.0"
```

This marks the fork feature line. The upstream package may continue its own numbering; this fork’s `2.9.0` only indicates that A-1 and A-2 are present in this tree.

---

## Build / compatibility notes

- **PyTorch:** `>=2.10` required (ABI header `<torch/extension.h>`).
- **CUDA:** `>=13.0` for native compilation; produced SASS covers `sm_80;90;100;110;120`.
- **Windows:** Supported. FA4 (CuTeDSL) remains unavailable on Windows due to missing `win_amd64` native libraries.
- **Fallback:** Both A-1 and A-2 are guarded by template flags / `__CUDA_ARCH__` so the same binary runs on Ampere, Hopper, and Blackwell.
