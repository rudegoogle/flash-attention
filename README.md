# FlashAttention

Source fork of [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). Installation, build prerequisites, CUDA and ROCm paths, API usage, benchmarks, tests, and citations are documented in the upstream repository; this file only lists pointers.

## Upstream (authoritative)

- Repository: https://github.com/Dao-AILab/flash-attention  
- README: https://github.com/Dao-AILab/flash-attention/blob/main/README.md  

Paper: https://tridao.me/publications/flash2/flash2.pdf

![FlashAttention-2](assets/flashattention_logo.png)


## Usage

We've been very happy to see FlashAttention being widely adopted in such a short
time after its release. This [page](https://github.com/Dao-AILab/flash-attention/blob/main/usage.md)
contains a partial list of places where FlashAttention is being used.

FlashAttention and FlashAttention-2 are free to use and modify (see LICENSE).
Please cite and credit FlashAttention if you use it.


## FlashAttention-3 beta release
FlashAttention-3 is optimized for Hopper GPUs (e.g. H100). 

Blogpost: https://tridao.me/blog/2024/flash3/

Paper: https://tridao.me/publications/flash3/flash3.pdf

![FlashAttention-3 speedup on H100 80GB SXM5 with FP16](assets/flash3_fp16_fwd.png)

This is a beta release for testing / benchmarking before we integrate that with
the rest of the repo.

Currently released:
- FP16 / BF16 forward and backward, FP8 forward

Requirements: H100 / H800 GPU, CUDA >= 12.3.

We highly recommend CUDA 12.8 for best performance.

To install:
```sh
cd hopper
python setup.py install
```
To run the test:
```sh
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py
```
Once the package is installed, you can import it as follows:
```python
from flash_attn_3 import flash_attn_interface
flash_attn_interface.flash_attn_func()
```

To install using `uv`, in your `pyproject.toml`:

```toml
[project]
dependencies = [
    "flash-attn-3"
]

[tool.uv]
no-build-isolation = true

[tool.uv.sources]
flash-attn-3 = { git = "https://github.com/Dao-AILab/flash-attention", subdirectory = "hopper" }
```

## FlashAttention-4 (CuTeDSL)

FlashAttention-4 is written in CuTeDSL and optimized for Hopper and Blackwell GPUs (e.g. H100, B200).

To install:
```sh
pip install flash-attn-4
```

If you're on CUDA 13, we recommend installing with the `cu13` extra for best performance:
```sh
pip install "flash-attn-4[cu13]"
```

Once installed, you can use it as follows:
```python
from flash_attn.cute import flash_attn_func

out = flash_attn_func(q, k, v, causal=True)
```

## Installation and features
**Requirements:**
- CUDA toolkit or ROCm toolkit
- PyTorch 2.2 and above.
- `packaging` Python package (`pip install packaging`)
- `psutil` Python package (`pip install psutil`)
- `ninja` Python package (`pip install ninja`) *
- Linux. Might work for Windows starting v2.3.2 (we've seen a few positive [reports](https://github.com/Dao-AILab/flash-attention/issues/595)) but Windows compilation still requires more testing. If you have ideas on how to set up prebuilt CUDA wheels for Windows, please reach out via Github issue.

\* Make sure that `ninja` is installed and that it works correctly (e.g. `ninja
--version` then `echo $?` should return exit code 0). If not (sometimes `ninja
--version` then `echo $?` returns a nonzero exit code), uninstall then reinstall
`ninja` (`pip uninstall -y ninja && pip install ninja`). Without `ninja`,
compiling can take a very long time (2h) since it does not use multiple CPU
cores. With `ninja` compiling takes 3-5 minutes on a 64-core machine using CUDA toolkit.

**To install:**
```sh
pip install flash-attn --no-build-isolation
```
Alternatively you can compile from source:
```sh
python setup.py install
```

If your machine has less than 96GB of RAM and lots of CPU cores, `ninja` might
run too many parallel compilation jobs that could exhaust the amount of RAM. To
limit the number of parallel compilation jobs, you can set the environment
variable `MAX_JOBS`:
```sh
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**Interface:** `src/flash_attention_interface.py`

### NVIDIA CUDA Support
**Requirements:**
- CUDA 12.0 and above.

We recommend the
[Pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
container from Nvidia, which has all the required tools to install FlashAttention.

FlashAttention-2 with CUDA currently supports:
1. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). For Turing GPUs (T4, RTX 2080), see the separate [flash-attention-turing](https://github.com/ssiu/flash-attention-turing) repo, which supports a core subset of FlashAttention features on Turing.
2. Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs).
3. All head dimensions up to 256. ~~Head dim > 192 backward requires A100/A800 or H100/H800~~. Head dim 256 backward now works on consumer GPUs (if there's no dropout) as of flash-attn 2.5.5.

### AMD ROCm Support
ROCm version has two backends. There is [composable_kernel](https://github.com/ROCm/composable_kernel) (ck) which is the default backend and a [Triton](https://github.com/triton-lang/triton) backend. They provide an implementation of FlashAttention-2.

**Requirements:**
- ROCm 6.0 and above.

We recommend the
[Pytorch](https://hub.docker.com/r/rocm/pytorch)
container from ROCm, which has all the required tools to install FlashAttention.

#### Composable Kernel Backend
FlashAttention-2 ROCm CK backend currently supports:
1. MI200x, MI250x, MI300x, MI355x, and RDNA 3/4 GPUs.
2. Datatype fp16 and bf16
3. Both forward's and backward's head dimensions up to 256.

#### Triton Backend
The Triton implementation of [Flash Attention](https://tridao.me/publications/flash2/flash2.pdf) supports AMD's CDNA (MI200, MI300) and RDNA GPUs using fp16, bf16, and fp32 datatypes. It provides forward and backward passes with causal masking, variable sequence lengths, arbitrary Q/KV sequence lengths and head sizes, MQA/GQA, dropout, rotary embeddings, ALiBi, paged attention, and FP8 (via the Flash Attention v3 interface). Sliding window attention is currently a work in progress.

The Triton backend kernels are provided by the [aiter](https://github.com/ROCm/aiter) package, included as a git submodule at `third_party/aiter` and automatically installed during setup.

To install, first get PyTorch for ROCm from https://pytorch.org/get-started/locally/, then install Flash Attention:
```sh
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .
```

To use a specific aiter commit (e.g., for testing or development):
```sh
cd flash-attention
cd third_party/aiter && git fetch origin && git checkout <commit-sha> && cd ../..
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .
```

To run the tests (note: full suite takes hours):
```sh
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pytest tests/test_flash_attn_triton_amd.py
```

The Triton backend uses a default kernel configuration optimized for determinism and reasonable performance across workloads. For peak throughput, enable `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"` to search for optimal settings, which incurs a one-time warmup cost.

Alternativly, if _not_ autotuning, `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON` may be used to set a single triton config overriding the hardcoded defaults for `attn_fwd`. E.g.
```sh
FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON='{"BLOCK_M":128,"BLOCK_N":64,"waves_per_eu":1,"PRE_LOAD_V":false,"num_stages":1,"num_warps":8}'
```

For a quick start with Docker:
```dockerfile
FROM rocm/pytorch:latest

WORKDIR /workspace

# build flash attention with triton backend
RUN git clone https://github.com/Dao-AILab/flash-attention &&\ 
    cd flash-attention &&\
    FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .

# set working dir
WORKDIR /workspace/flash-attention

# set env variable to use triton backend
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
```

Build and run:
```sh
docker build -t flash-attn-triton .
docker run -it --network=host --user root --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 16G --device=/dev/kfd --device=/dev/dri flash-attn-triton
```

## How to use FlashAttention

The main functions implement scaled dot product attention (softmax(Q @ K^T *
softmax_scale) @ V):
```python
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
```

```python
flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                          window_size=(-1, -1), alibi_slopes=None, deterministic=False):
"""dropout_p should be set to 0.0 during evaluation
If Q, K, V are already stacked into 1 tensor, this function will be faster than
calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
of the gradients of Q, K, V.
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.
Arguments:
    qkv: (batch_size, seqlen, 3, nheads, headdim)
    dropout_p: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
        the attention score of query i and key j.
    deterministic: bool. Whether to use the deterministic implementation of the backward pass,
        which is slightly slower and uses more memory. The forward pass is always deterministic.
Return:
    out: (batch_size, seqlen, nheads, headdim).
"""
```

```python
flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False):
"""dropout_p should be set to 0.0 during evaluation
Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Arguments:
    q: (batch_size, seqlen, nheads, headdim)
    k: (batch_size, seqlen, nheads_k, headdim)
    v: (batch_size, seqlen, nheads_k, headdim)
    dropout_p: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    deterministic: bool. Whether to use the deterministic implementation of the backward pass,
        which is slightly slower and uses more memory. The forward pass is always deterministic.
Return:
    out: (batch_size, seqlen, nheads, headdim).
"""
```

```python
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
```

To see how these functions are used in a multi-head attention layer (which
includes QKV projection, output projection), see the MHA [implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py).

### Using with 🤗 Kernels

If your hardware environment belongs to any of the above-mentioned, you can also use the [`kernels` library](https://github.com/huggingface/kernels)
to use Flash Attention 2 and 3 right away.

```py
# pip install kernels

from kernels import get_kernel

# FA2
fa_module = get_kernel("kernels-community/flash-attn2", version=1)
flash_attn_func = fa_module.flash_attn_func

# FA3
fa3_module = get_kernel("kernels-community/flash-attn3", version=1)
flash_attn_func = fa3_module.flash_attn_func
```
>>>>>>> upstream/main

## Changelog

- Fork-only release history (not the upstream project changelog): [md/CHANGELOG.md](md/CHANGELOG.md)

## Documentation

- [Explanation of GitHub Actions Workflows Removal](md/FA4_WORKFLOW_REMOVAL_EXPLANATION.md)
