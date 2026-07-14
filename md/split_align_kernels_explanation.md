# FlashAttention v2.9.2: split_align Architecture Technical Specification

## 1. Overview

The `split_align` architecture changes introduced in FlashAttention v2.9.2 serve as a structural workaround to resolve **severe compilation delays and memory crashes caused by the NVIDIA compiler (ptxas) serializing massive AST compilations**.

Under the specific condition where sequence splitting is disabled (`num_splits == 1`), a highly optimized block width (`kBlockN_standard`) is required for alignment. Attempting to instantiate this alongside the standard split path within a single dispatch file forces the compiler to generate multiple massive ASTs concurrently, leading to OOM crashes. To prevent this, the `num_splits == 1` instantiations were **extracted into 24 extremely small, independent translation units (`flash_fwd_split_align_*.cu`)**. This forces the build system to compile them in parallel across multiple threads.

However, on Windows (MSVC) environments, linking these dozens of massive object files simultaneously triggers the absolute **2GB size limit (LNK1189)**. Therefore, this project implements a mandatory workaround by dividing the target SM architectures into two separate build configurations: `bat2` (Blackwell family, 3 targets) and `bat3` (Ampere/Hopper family, 3 targets). This perfectly halves the linker payload size, allowing Windows users to bypass the 2GB limit while retaining the performance benefits of the parallelized `split_align` compilation.

## 2. List of Added and Modified Files

Below is the enumerated list of all 50 files involved in this architecture change.

### Core Dispatch & Generator Scripts
1. `csrc\flash_attn\src\generate_kernels.py`
2. `csrc\flash_attn\src\flash_fwd_launch_template.h`

### Newly Added Alignment-Optimized Kernels (24 files)
3. `csrc\flash_attn\src\flash_fwd_split_align_hdim128_bf16_causal_sm80.cu`
4. `csrc\flash_attn\src\flash_fwd_split_align_hdim128_bf16_sm80.cu`
5. `csrc\flash_attn\src\flash_fwd_split_align_hdim128_fp16_causal_sm80.cu`
6. `csrc\flash_attn\src\flash_fwd_split_align_hdim128_fp16_sm80.cu`
7. `csrc\flash_attn\src\flash_fwd_split_align_hdim192_bf16_causal_sm80.cu`
8. `csrc\flash_attn\src\flash_fwd_split_align_hdim192_bf16_sm80.cu`
9. `csrc\flash_attn\src\flash_fwd_split_align_hdim192_fp16_causal_sm80.cu`
10. `csrc\flash_attn\src\flash_fwd_split_align_hdim192_fp16_sm80.cu`
11. `csrc\flash_attn\src\flash_fwd_split_align_hdim256_bf16_causal_sm80.cu`
12. `csrc\flash_attn\src\flash_fwd_split_align_hdim256_bf16_sm80.cu`
13. `csrc\flash_attn\src\flash_fwd_split_align_hdim256_fp16_causal_sm80.cu`
14. `csrc\flash_attn\src\flash_fwd_split_align_hdim256_fp16_sm80.cu`
15. `csrc\flash_attn\src\flash_fwd_split_align_hdim32_bf16_causal_sm80.cu`
16. `csrc\flash_attn\src\flash_fwd_split_align_hdim32_bf16_sm80.cu`
17. `csrc\flash_attn\src\flash_fwd_split_align_hdim32_fp16_causal_sm80.cu`
18. `csrc\flash_attn\src\flash_fwd_split_align_hdim32_fp16_sm80.cu`
19. `csrc\flash_attn\src\flash_fwd_split_align_hdim64_bf16_causal_sm80.cu`
20. `csrc\flash_attn\src\flash_fwd_split_align_hdim64_bf16_sm80.cu`
21. `csrc\flash_attn\src\flash_fwd_split_align_hdim64_fp16_causal_sm80.cu`
22. `csrc\flash_attn\src\flash_fwd_split_align_hdim64_fp16_sm80.cu`
23. `csrc\flash_attn\src\flash_fwd_split_align_hdim96_bf16_causal_sm80.cu`
24. `csrc\flash_attn\src\flash_fwd_split_align_hdim96_bf16_sm80.cu`
25. `csrc\flash_attn\src\flash_fwd_split_align_hdim96_fp16_causal_sm80.cu`
26. `csrc\flash_attn\src\flash_fwd_split_align_hdim96_fp16_sm80.cu`

### Modified Existing Split Kernels (24 files)
27. `csrc\flash_attn\src\flash_fwd_split_hdim128_bf16_causal_sm80.cu`
28. `csrc\flash_attn\src\flash_fwd_split_hdim128_bf16_sm80.cu`
29. `csrc\flash_attn\src\flash_fwd_split_hdim128_fp16_causal_sm80.cu`
30. `csrc\flash_attn\src\flash_fwd_split_hdim128_fp16_sm80.cu`
31. `csrc\flash_attn\src\flash_fwd_split_hdim192_bf16_causal_sm80.cu`
32. `csrc\flash_attn\src\flash_fwd_split_hdim192_bf16_sm80.cu`
33. `csrc\flash_attn\src\flash_fwd_split_hdim192_fp16_causal_sm80.cu`
34. `csrc\flash_attn\src\flash_fwd_split_hdim192_fp16_sm80.cu`
35. `csrc\flash_attn\src\flash_fwd_split_hdim256_bf16_causal_sm80.cu`
36. `csrc\flash_attn\src\flash_fwd_split_hdim256_bf16_sm80.cu`
37. `csrc\flash_attn\src\flash_fwd_split_hdim256_fp16_causal_sm80.cu`
38. `csrc\flash_attn\src\flash_fwd_split_hdim256_fp16_sm80.cu`
39. `csrc\flash_attn\src\flash_fwd_split_hdim32_bf16_causal_sm80.cu`
40. `csrc\flash_attn\src\flash_fwd_split_hdim32_bf16_sm80.cu`
41. `csrc\flash_attn\src\flash_fwd_split_hdim32_fp16_causal_sm80.cu`
42. `csrc\flash_attn\src\flash_fwd_split_hdim32_fp16_sm80.cu`
43. `csrc\flash_attn\src\flash_fwd_split_hdim64_bf16_causal_sm80.cu`
44. `csrc\flash_attn\src\flash_fwd_split_hdim64_bf16_sm80.cu`
45. `csrc\flash_attn\src\flash_fwd_split_hdim64_fp16_causal_sm80.cu`
46. `csrc\flash_attn\src\flash_fwd_split_hdim64_fp16_sm80.cu`
47. `csrc\flash_attn\src\flash_fwd_split_hdim96_bf16_causal_sm80.cu`
48. `csrc\flash_attn\src\flash_fwd_split_hdim96_bf16_sm80.cu`
49. `csrc\flash_attn\src\flash_fwd_split_hdim96_fp16_causal_sm80.cu`
50. `csrc\flash_attn\src\flash_fwd_split_hdim96_fp16_sm80.cu`

---
## 3. Detailed Specifications & Full Source Code

### 3.1 Core Dispatch & Generator Scripts

#### 📄 File 1/50: `csrc\flash_attn\src\generate_kernels.py`
- **Specification**: A Python script that dynamically generates the 24 `fwd_split_align` files. It was updated to also append an `extern` declaration to the existing `fwd_split` files to prevent double compilation.
```python
import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [32, 64, 96, 128, 192, 256]
IS_CAUSAL = ["false", "true"]
NAMESPACE_INCLUDE = '#include "namespace_config.h"\n'

def get_fwd_template() -> str:
    return NAMESPACE_INCLUDE + """#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {{

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}

}} // namespace FLASH_NAMESPACE"""

def get_fwd_split_template() -> str:
    return NAMESPACE_INCLUDE + """#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {{

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream);

}} // namespace FLASH_NAMESPACE"""

def get_fwd_split_align_template() -> str:
    return NAMESPACE_INCLUDE + """#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {{

template void run_mha_fwd_splitkv_align<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream);

}} // namespace FLASH_NAMESPACE"""

def get_bwd_template() -> str:
    return NAMESPACE_INCLUDE + """#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {{

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}

}} // namespace FLASH_NAMESPACE"""

@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    is_causal: bool
    direction: str

    @property
    def template(self) -> str:
        template_funcs = {
            "fwd": get_fwd_template,
            "bwd": get_bwd_template,
            "fwd_split": get_fwd_split_template,
            "fwd_split_align": get_fwd_split_align_template,
        }
        template_func = template_funcs[self.direction]
        return template_func().format(
            DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
            IS_CAUSAL=self.is_causal
        )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_{'causal_' if self.is_causal == 'true' else ''}sm{self.sm}.cu"

def get_all_kernels() -> List[Kernel]:
    for direction in ["fwd", "fwd_split", "fwd_split_align", "bwd"]:
        for dtype, head_dim, is_causal, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, IS_CAUSAL, SM):
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, is_causal=is_causal, direction=direction)

def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n"""
    content = prelude + kernel.template
    (autogen_dir / kernel.filename).write_text(content)

def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)

```

#### 📄 File 2/50: `csrc\flash_attn\src\flash_fwd_launch_template.h`
- **Specification**: The root header file for all kernel calls. It was modified to include a conditional branch for `params.num_splits == 1`, dispatching execution to the alignment-optimized `run_mha_fwd_splitkv_align` kernels.
```cpp
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

namespace FLASH_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
    static_assert(Log_max_splits >= 1);
    FLASH_NAMESPACE::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            // Will only return softmax if dropout, to reduce compilation time.
                            // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                            // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                            // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                            // If Is_local, set Is_causal to false
                            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !Has_alibi && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst && !Has_alibi, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, false, true, true, false>;
                            // printf("IsEvenMNConst = %d, IsEvenKConst = %d, Is_local = %d, Is_causal = %d, ReturnSoftmaxConst = %d, Is_dropout = %d\n", int(IsEvenMNConst), int(IsEvenKConst), int(Is_local), int(Is_causal), int(ReturnSoftmaxConst), int(Is_dropout));
                            // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, true, true, false>;
                            if (smem_size >= 48 * 1024) {
                                C10_CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                            }
                            // int ctas_per_sm;
                            // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                                // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
                                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                                // If Is_local, set Is_causal to false
                                auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && !Has_alibi && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !Has_alibi, Is_softcap, Split, Append_KV>;
                                // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, true, Split, Append_KV>;
                                // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, IsEvenKConst>;
                                if (smem_size >= 48 * 1024) {
                                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                }
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                            });
                        });
                    });
                });
            });
        });
    });
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}

// The num_splits==1 blocksize-aligned splitkv template. If a user specifies
// num_splits=1, we assume they want bitwise identical numerics across the split
// KV and standard kernels so we align kBlockN to match.
// We technically can combine this into one dispatch under run_flash_splitkv_fwd
// but that pathologically slowed down build time by doubling the number of kernels
// in a single file, which made build go from minutes to hours. Thus, it is pulled
// into its own function so it can be explicitly instantiated in a separate file
// (flash_fwd_split_align_*.cu) and compiled in parallel instead of serializing in
// one ptxas invocation.
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_align(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN_standard = Headdim <= 64 ? 128 : 64;
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN_standard, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    constexpr static int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    if (params.num_splits == 1) {
        // Defined in flash_fwd_split_align_*.cu; declared extern in the main
        // flash_fwd_split_*.cu so this call does not re-instantiate the tree here.
        run_mha_fwd_splitkv_align<T, Headdim, Is_causal>(params, stream);
        return;
    }
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
            // Using block size (64 x 256) is 27% slower for seqlen=2k
            // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        // These two are always slower
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 1st ones are good for H100, A100
            // 2nd one is good for A6000 bc we get slightly better occupancy
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For A100, we want to run with 128 x 64 (128KB smem).
        // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // 64 KB
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // 96 KB
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}
}  // namespace FLASH_NAMESPACE

```

### 3.2 Instantiated Kernel Group (48 Files)

#### 📄 File 3/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim32_fp16_causal_sm80.cu`
- **Head Dimension**: `32` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 128` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 4/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim32_fp16_causal_sm80.cu`
- **Head Dimension**: `32` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 5/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim32_fp16_sm80.cu`
- **Head Dimension**: `32` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 128` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 6/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim32_fp16_sm80.cu`
- **Head Dimension**: `32` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 7/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim32_bf16_causal_sm80.cu`
- **Head Dimension**: `32` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 128` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 8/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim32_bf16_causal_sm80.cu`
- **Head Dimension**: `32` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 9/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim32_bf16_sm80.cu`
- **Head Dimension**: `32` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 128` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 10/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim32_bf16_sm80.cu`
- **Head Dimension**: `32` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 11/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim64_fp16_causal_sm80.cu`
- **Head Dimension**: `64` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 128` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 12/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim64_fp16_causal_sm80.cu`
- **Head Dimension**: `64` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 13/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim64_fp16_sm80.cu`
- **Head Dimension**: `64` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 128` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 14/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim64_fp16_sm80.cu`
- **Head Dimension**: `64` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 15/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim64_bf16_causal_sm80.cu`
- **Head Dimension**: `64` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 128` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 16/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim64_bf16_causal_sm80.cu`
- **Head Dimension**: `64` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 17/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim64_bf16_sm80.cu`
- **Head Dimension**: `64` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 128` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 18/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim64_bf16_sm80.cu`
- **Head Dimension**: `64` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 19/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim96_fp16_causal_sm80.cu`
- **Head Dimension**: `96` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 20/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim96_fp16_causal_sm80.cu`
- **Head Dimension**: `96` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 21/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim96_fp16_sm80.cu`
- **Head Dimension**: `96` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 22/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim96_fp16_sm80.cu`
- **Head Dimension**: `96` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 23/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim96_bf16_causal_sm80.cu`
- **Head Dimension**: `96` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 24/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim96_bf16_causal_sm80.cu`
- **Head Dimension**: `96` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 25/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim96_bf16_sm80.cu`
- **Head Dimension**: `96` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 26/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim96_bf16_sm80.cu`
- **Head Dimension**: `96` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 27/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim128_fp16_causal_sm80.cu`
- **Head Dimension**: `128` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 28/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim128_fp16_causal_sm80.cu`
- **Head Dimension**: `128` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 29/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim128_fp16_sm80.cu`
- **Head Dimension**: `128` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 30/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim128_fp16_sm80.cu`
- **Head Dimension**: `128` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 31/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim128_bf16_causal_sm80.cu`
- **Head Dimension**: `128` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 32/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim128_bf16_causal_sm80.cu`
- **Head Dimension**: `128` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 33/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim128_bf16_sm80.cu`
- **Head Dimension**: `128` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 34/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim128_bf16_sm80.cu`
- **Head Dimension**: `128` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 35/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim192_fp16_causal_sm80.cu`
- **Head Dimension**: `192` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 36/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim192_fp16_causal_sm80.cu`
- **Head Dimension**: `192` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 37/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim192_fp16_sm80.cu`
- **Head Dimension**: `192` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 38/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim192_fp16_sm80.cu`
- **Head Dimension**: `192` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 39/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim192_bf16_causal_sm80.cu`
- **Head Dimension**: `192` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 40/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim192_bf16_causal_sm80.cu`
- **Head Dimension**: `192` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 41/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim192_bf16_sm80.cu`
- **Head Dimension**: `192` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 42/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim192_bf16_sm80.cu`
- **Head Dimension**: `192` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 43/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim256_fp16_causal_sm80.cu`
- **Head Dimension**: `256` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 44/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim256_fp16_causal_sm80.cu`
- **Head Dimension**: `256` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 45/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim256_fp16_sm80.cu`
- **Head Dimension**: `256` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::half_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 46/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim256_fp16_sm80.cu`
- **Head Dimension**: `256` | **Data Type**: `fp16` (`cutlass::half_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 47/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim256_bf16_causal_sm80.cu`
- **Head Dimension**: `256` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 48/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim256_bf16_causal_sm80.cu`
- **Head Dimension**: `256` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: Yes (`true`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 49/50: [NEW] `csrc\flash_attn\src\flash_fwd_split_align_hdim256_bf16_sm80.cu`
- **Head Dimension**: `256` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: Alignment-optimized kernel used when `num_splits == 1`. It isolates the instantiation of `run_mha_fwd_splitkv_align` using the optimized block width `kBlockN_standard = 64` into its own translation unit.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

#### 📄 File 50/50: [MODIFIED] `csrc\flash_attn\src\flash_fwd_split_hdim256_bf16_sm80.cu`
- **Head Dimension**: `256` | **Data Type**: `bf16` (`cutlass::bfloat16_t`) | **Causal Mask**: Causal Mask: No (`false`)
- **Specification**: The existing split kernel file. It was modified to include an `extern` declaration for the above alignment-optimized kernel, explicitly preventing the main dispatch tree from re-compiling it and triggering memory crashes.
```cpp
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// The num_splits==1 blocksize-aligned tree is instantiated in its own translation unit
// (flash_fwd_split_align_*.cu) so it compiles in parallel; declare it extern so
// the dispatch below references it instead of re-instantiating.
extern template void run_mha_fwd_splitkv_align<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE
```

