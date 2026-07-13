# FlashAttention

Source fork of [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). Installation, build prerequisites, CUDA and ROCm paths, API usage, benchmarks, tests, and citations are documented in the upstream repository; this file only lists pointers.

## Pre-built Windows Wheels (Recommended)

Pre-compiled Windows `.whl` binaries for various PyTorch and CUDA versions (including the latest `split_align` architectures for Blackwell/Ampere/Hopper) are officially published at:
👉 **[Hugging Face: Flash-Attention-2_for_Windows](https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows)**

### Installation Guideline

You do **not** need to build from source (which can take several hours). Simply locate the wheel that matches your environment (Python, PyTorch, CUDA version) from the Hugging Face repository and install it directly via pip:

```bash
pip install https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/<your_matching_wheel_file>.whl
```

**⚠️ Important Note for v1.6 (fa2.9.2) users:**
If your chosen wheel filename contains `-blackwell-` or `-legacy-` just before the `-cp313` portion, you must **rename the first hyphen to a period (`.`)** before installing, or `pip` will throw an `Invalid build number` error due to strict PEP 427 filename rules.
- ❌ `...abiTRUE-blackwell-cp...whl`
- ✅ `...abiTRUE.blackwell-cp...whl`

*(This filename issue has been corrected in the repository's build scripts for future releases).*

## Upstream (authoritative)

- Repository: https://github.com/Dao-AILab/flash-attention  
- README: https://github.com/Dao-AILab/flash-attention/blob/main/README.md  

## Changelog

- Fork-only release history (not the upstream project changelog): [md/CHANGELOG.md](md/CHANGELOG.md)

## Documentation

- [Explanation of GitHub Actions Workflows Removal](md/FA4_WORKFLOW_REMOVAL_EXPLANATION.md)
