---
license: bsd-3-clause
tags:
- flash-attention
- windows
- pytorch
- cuda
- whl
language:
- en
---

# FlashAttention Windows Wheel

Unofficial Windows-compatible wheel of flash-attention for Windows<br>
**Python 3.11 & 3.12 & 3.13 versions only**.

## Overview

This repository provides **Windows-compatible wheels** for FlashAttention-2 that are not officially distributed.<br>
Pre-built version: **flash_attn 2.9.0** with **Python 3.11 & 3.12 & 3.13** support.

## Key Features

- ✅ Native Windows support (Python 3.11 & 3.12 & 3.13)
- ⚡ FlashAttention-2

## Changelog

- 15.11.2025 Uploaded v2.8.3 based on PyTorch 2.9.1+cu130
- 12.02.2026 Uploaded v2.8.3 based on PyTorch 2.10.0+cu130
- 29.03.2026 Uploaded v2.8.3 based on PyTorch 2.11.0+cu130
- 14.05.2026 Uploaded v2.8.4 based on PyTorch 2.11.0+cu130
- 13.05.2026 Uploaded **v2.9.0** based on PyTorch 2.11.0+cu130 — **unofficial fork-only build** (not an official FlashAttention release). Includes FA2 A-1/A-2 optimizations.

## About v2.9.0

**v2.9.0 is not an official FlashAttention release.**<br>
It is an independent fork build that continues FA2 kernel development while upstream focuses on FA3/F4.

- **Optimization plan (GitHub):** https://github.com/ussoewwin/flash-attention/blob/main/AI/FA2_BACKPORT_FROM_FA4_PLAN.md
- **Kernel change notes (GitHub):** https://github.com/ussoewwin/flash-attention/blob/main/md/FA2_CHANGES_v1.2.md
- **Test and validation results (GitHub):** https://github.com/ussoewwin/flash-attention/blob/main/md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md

## Disclaimer

- **Test scope:** Documented in the [2.9.0 test and validation guide](https://github.com/ussoewwin/flash-attention/blob/main/md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md) (smoke matrices, A-1/A-2, Triton, wheel checks). Not a full multi-environment or production benchmark suite.
- This is an **unofficial fork build**. Use at your own risk.

※Unofficial built version!! It works correctly in my environment, but I am not sure that will work in yours.
