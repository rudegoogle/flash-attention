# CUDA 13.0 → 13.2 Windows wheel build: errors, root causes, and code fixes

This document records how **FlashAttention 2.9.0 (fork v1.2)** was built on **Windows** when moving from a **CUDA 13.0** PyTorch stack to **CUDA 13.2** (`torch 2.12.0+cu132`). It is written for anyone reproducing the build or auditing why `setup.py` and docs changed.

**Successful artifact (reference):**

- Wheel: `dist/flash_attn-2.9.0+cu132torch2.12.0cxx11abiTRUE-cp313-cp313-win_amd64.whl`
- Built: **2026-05-15 19:39:32** (local time)
- GPU tested: **NVIDIA GeForce RTX 5060 Ti** (`sm_120`)
- Post-build tests: `_test_a2.py` (8/8 accuracy OK), `_test_triton.py` (JIT OK) — see `md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md` §10

**Git commits that implement the arch-policy fixes:**

| Commit   | Subject |
|----------|---------|
| `33c55ee` | Drop `sm_110` from default CUDA arch list and build gencodes |
| `cf2f241` | Document and enforce CUDA arch policy `80;90;100;120` |

---

## 1. Environment: what changed from cu130 to cu132

| Item | cu130 path (earlier) | cu132 path (this port) |
|------|----------------------|-------------------------|
| PyTorch | e.g. `2.11.0+cu130` | **`2.12.0+cu132`** |
| CUDA toolkit (`CUDA_HOME`) | `C:\...\CUDA\v13.0` | **`C:\...\CUDA\v13.2`** |
| Wheel filename tag | `cu130torch2.11.0...` | **`cu132torch2.12.0...`** |
| Target GPU (this machine) | — | **RTX 50-series → `sm_120`** |
| Build logs | `_build_cu130*.log` (if any) | `_build_cu132*.log`, `_build_cu132_zcp.log` |

Build command used (venv with cu132 torch):

```bat
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_HOME%\bin;%PATH%
set MAX_JOBS=4
python setup.py build_ext --inplace
```

Canonical Windows wheel script (sets `DISTUTILS_USE_SDK=1`):

```bat
WindowsWhlBuilder_cuda.bat
```

Optional arch override (must match fork policy):

```bat
WindowsWhlBuilder_cuda.bat CUDA_ARCH 80;90;100;120
```

---

## 2. First errors that appeared (verbatim from build logs)

When `CUDA_HOME` pointed at **13.2** but `setup.py` still used the pre-fix default arch list **`80;90;100;110;120`** and Thor gencode logic, the **first failing translation unit** in log `_build_cu132_zcp.log` was `flash_bwd_hdim128_bf16_causal_sm80.cu`. Two independent failures showed up in the same ninja step.

### 2.1 Error A — CCCL / MSVC preprocessor (CUDA 13.2)

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin/../include/cccl\cuda/std/__cccl/preprocessor.h(23): fatal error C1189: #error:  MSVC/cl.exe with traditional preprocessor is used. This may lead to unexpected compilation errors. Please switch to the standard conforming preprocessor by passing `/Zc:preprocessor` to cl.exe. You can define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING to suppress this warning.
```

**When it happens:** `nvcc` compiles a `.cu` file and invokes **MSVC `cl.exe`** for host-side phases. CUDA **13.2** ships stricter **CCCL** headers under `include/cccl/`. Those headers **reject the legacy (“traditional”) MSVC preprocessor**.

**What did *not* fix it:** Ignoring the warning via `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` only hides the diagnostic; the build still needs a conforming preprocessor for real compilation.

### 2.2 Error B — spurious `sm_110` gencode (Thor) on this fork

The same nvcc command line included Thor family gencode (before commit `33c55ee`):

```
-gencode arch=compute_80,code=sm_80
-gencode arch=compute_90,code=sm_90
-gencode arch=compute_100f,code=sm_100
-gencode arch=compute_120f,code=sm_120
-gencode arch=compute_110f,code=sm_110
-gencode arch=compute_120,code=compute_120
```

Follow-on failure:

```
fatal   : Could not open input file C:/Users/ussoe/AppData/Local/Temp/tmpxft_000070dc_00000000-25_flash_bwd_hdim128_bf16_causal_sm80.compute_110f.cpp1.ii
ninja: build stopped: subcommand failed.
RuntimeError: Error compiling objects for extension
```

**Meaning:** The fork’s FA2 CUDA sources are instantiated for **`sm_80` / `sm_90` / `sm_100` / `sm_120`** (Ampere, Hopper, Blackwell consumer/datacenter). There is **no** `flash_*_sm110` (or Thor-specific) kernel variant in this tree. Asking `nvcc` to run the **`sm_80` template path** while also passing **`-gencode ... sm_110`** makes the toolchain generate intermediate files such as `*.compute_110f.cpp1.ii` that **do not correspond to a supported kernel build** → **“Could not open input file”**.

---

## 3. Root cause (essential)

| # | Symptom | Root cause |
|---|---------|------------|
| **A** | `__cccl/preprocessor.h` **C1189** | **CUDA 13.2 + CCCL** requires MSVC **`/Zc:preprocessor`**. PyTorch’s default `nvcc`/`cl` flags on Windows did not pass it. `setup.py` only added `/Zc:__cplusplus` when `DISTUTILS_USE_SDK=1`, which is insufficient for CCCL on 13.2. |
| **B** | `compute_110f.cpp1.ii` missing | Default **`FLASH_ATTN_CUDA_ARCHS`** included **`110`**, and `add_cuda_gencodes()` emitted **`-gencode arch=compute_110f,code=sm_110`** for toolkit ≥ 13.0. This fork **does not build Thor (`sm_101`/`sm_110`)** FA2 kernels; multi-arch wheels should be **`80;90;100;120` only**. |

Both issues are **orthogonal**: fixing only A still leaves B; fixing only B still leaves A. A reproducible **cu132** Windows build needs **both**.

---

## 4. Files modified for the 13.2 port

| File | Role |
|------|------|
| `setup.py` | Arch defaults, Thor stripping, `add_cuda_gencodes()`, Windows MSVC flags |
| `md/FA2_CHANGES_v1.2.md` | Arch policy table + Windows `/Zc:preprocessor` note |
| `md/CHANGELOG.md` | v1.2 summary: default arch list |
| `md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md` | Env table: `FLASH_ATTN_CUDA_ARCHS` |
| `WindowsWhlBuilder_cuda.bat` | Comment/default example `80;90;100;120` |
| `md/CUDA_13.0_TO_13.2_BUILD_FIX.md` | This document |

Kernel / FA2 feature code (`softmax.h`, launch templates, etc.) is **unchanged** for the toolkit bump; only **build policy** and **MSVC host flags** changed.

---

## 5. Code changes (full text of modified sections)

### 5.1 `setup.py` — constants and `cuda_archs()` (after `cf2f241`)

**Purpose:** Single source of truth for which SASS targets this fork emits; strip Thor (`101`, `110`) and unknown tokens with warnings; default env string **`80;90;100;120`**.

```python
# FA2 fork wheel build: SASS for Ampere/Hopper/Blackwell only (see md/FA2_CHANGES_v1.2.md).
FORK_SUPPORTED_CUDA_ARCHS = ("80", "90", "100", "120")
FORK_THOR_CUDA_ARCHS = frozenset({"101", "110"})


@functools.lru_cache(maxsize=None)
def cuda_archs() -> list[str]:
    raw = os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120")
    requested = [a.strip() for a in raw.split(";") if a.strip()]
    dropped_thor = [a for a in requested if a in FORK_THOR_CUDA_ARCHS]
    if dropped_thor:
        warnings.warn(
            "FLASH_ATTN_CUDA_ARCHS includes Thor GPU arch(es) "
            f"{dropped_thor}; this fork FA2 build emits SASS for "
            f"{list(FORK_SUPPORTED_CUDA_ARCHS)} only. Ignoring those entries.",
            stacklevel=2,
        )
    archs = [a for a in requested if a in FORK_SUPPORTED_CUDA_ARCHS]
    unknown = [
        a
        for a in requested
        if a not in FORK_SUPPORTED_CUDA_ARCHS and a not in FORK_THOR_CUDA_ARCHS
    ]
    if unknown:
        warnings.warn(
            f"FLASH_ATTN_CUDA_ARCHS entries ignored (not built by this fork): {unknown}",
            stacklevel=2,
        )
    if not archs:
        warnings.warn(
            "FLASH_ATTN_CUDA_ARCHS has no supported entries after filtering; "
            f"using default {list(FORK_SUPPORTED_CUDA_ARCHS)}.",
            stacklevel=2,
        )
        archs = list(FORK_SUPPORTED_CUDA_ARCHS)
    return archs
```

**Before (`33c55ee` parent):**

```python
@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;110;120").split(";")
```

**Meaning:**

- **`110` in the default list** caused every clean build to request Thor gencode on CUDA 13+ toolchains → **Error B**.
- Returning a **filtered `list[str]`** lets `add_cuda_gencodes()` only see supported arches; user typos and Thor IDs fail **loudly** via `warnings.warn` instead of breaking ninja mid-compile.

---

### 5.2 `setup.py` — `add_cuda_gencodes()` (after `33c55ee` + `cf2f241`)

**Purpose:** Map `80/90/100/120` to correct `-gencode` lines for CUDA 12.8+ / 12.9+ (`100f`, `120f` on ≥12.9). **No Thor branch.**

```python
def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    """
    Adds -gencode flags for this fork's supported CUDA arch list only.

    Requested arch tokens (after cuda_archs() filtering) map to nvcc targets as:
      - 80  -> compute_80, sm_80
      - 90  -> compute_90, sm_90  (CUDA >= 11.8)
      - 100 -> compute_100f, sm_100 on CUDA >= 12.9 else compute_100, sm_100 (CUDA >= 12.8)
      - 120 -> compute_120f, sm_120 on CUDA >= 12.9 else compute_120, sm_120 (CUDA >= 12.8)

    Thor / sm_101 / sm_110 are not built (see FORK_THOR_CUDA_ARCHS in cuda_archs()).
    PTX for the newest numeric arch is embedded for forward-compatible JIT.
    """
    # Always-regular 80
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]

    # Hopper 9.0 needs >= 11.8
    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]

    # Blackwell 10.x requires >= 12.8
    if bare_metal_version >= Version("12.8"):
        if "100" in archs:
            # CUDA 12.9 introduced "family-specific" for Blackwell (100f)
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_100f,code=sm_100"]
            else:
                cc_flag += ["-gencode", "arch=compute_100,code=sm_100"]

        if "120" in archs:
            # sm_120 is supported in CUDA 12.8/12.9+ toolkits
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_120f,code=sm_120"]
            else:
                cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]

    # PTX for newest requested arch (forward-compat)
    numeric = [a for a in archs if a.isdigit()]
    if numeric:
        newest = max(numeric, key=int)
        cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    return cc_flag
```

**Removed block (commit `33c55ee`) — this caused Error B:**

```python
        # Thor rename: 12.9 uses sm_101; 13.0+ uses sm_110
        if "110" in archs:
            if bare_metal_version >= Version("13.0"):
                cc_flag += ["-gencode", "arch=compute_110f,code=sm_110"]
            else:
                # Provide Thor support for CUDA 12.9 via sm_101
                if bare_metal_version >= Version("12.8"):
                    cc_flag += ["-gencode", "arch=compute_101,code=sm_101"]
                # else: no Thor support in older toolkits
```

**Meaning:**

- Thor **`sm_110` / `sm_101`** gencode is appropriate only for **Thor-targeted** native code. This fork’s FA2 instantiations are **`sm_80`–`sm_120`** family only (see A-2 gating `__CUDA_ARCH__ >= 1000` in `softmax.h`, documented in `md/FA2_CHANGES_v1.2.md`).
- With **`110` removed**, nvcc no longer generates orphan `*.compute_110f.cpp1.ii` paths for `*_sm80.cu` sources.

---

### 5.3 `setup.py` — Windows MSVC flags when `DISTUTILS_USE_SDK=1` (CUDA 13.2 fix)

**Purpose:** Pass **`/Zc:preprocessor`** to every MSVC invocation that `nvcc` and `BuildExtension` use, satisfying CCCL on CUDA **13.2** (**Error A**).

```python
    compiler_c17_flag=["-O3", "-std=c++17"]
    # Add Windows-specific flags
    if sys.platform == "win32" and os.getenv('DISTUTILS_USE_SDK') == '1':
        # CUDA 13.2 CCCL headers require MSVC conforming preprocessor (see md/CUDA_13.0_TO_13.2_BUILD_FIX.md).
        nvcc_flags.extend(
            ["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/Zc:preprocessor"]
        )
        compiler_c17_flag = [
            "-O2",
            "/std:c++17",
            "/Zc:__cplusplus",
            "/Zc:preprocessor",
        ]
```

**Before (cu130-era / incomplete for 13.2):**

```python
    if sys.platform == "win32" and os.getenv('DISTUTILS_USE_SDK') == '1':
        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus"])
        compiler_c17_flag=["-O2", "/std:c++17", "/Zc:__cplusplus"]
```

**Meaning:**

| Flag | Effect |
|------|--------|
| `/Zc:__cplusplus` | MSVC reports `__cplusplus` consistent with `-std:c++17` (already required for PyTorch extensions). |
| **`/Zc:preprocessor`** | Enables **standards-conforming preprocessor** required by **CCCL** in CUDA **13.2**; without it, `__cccl/preprocessor.h` issues **C1189** and host-side compile of `.cu` units fails. |
| `-Xcompiler ...` on `nvcc_flags` | Forwards flags to **`cl.exe`** during device compilation host phases. |
| `compiler_c17_flag` | Same flags for **pure `.cpp`** extension sources compiled by MSVC. |

`WindowsWhlBuilder_cuda.bat` sets `DISTUTILS_USE_SDK=1`, so **`bdist_wheel`** builds pick up these flags automatically.

---

### 5.4 Documentation and batch file (commit `cf2f241`)

**`md/FA2_CHANGES_v1.2.md`** — build notes excerpt:

```markdown
- **CUDA:** Toolkit **13.2** for native compilation on cu132 PyTorch builds.
- **Windows:** Supported. ... MSVC host compiles invoked by `nvcc` need `/Zc:preprocessor` (see `setup.py` when `DISTUTILS_USE_SDK=1`).
```

Arch table documents `101`/`110` **not built**, default `80;90;100;120`.

**`md/CHANGELOG.md` v1.2** — adds default arch sentence (multi-arch, no Thor).

**`md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md`** — environment table row:

```markdown
| Multi-arch wheel build | `FLASH_ATTN_CUDA_ARCHS="80;90;100;120"` (see `setup.py`; Thor `101`/`110` not built) |
```

**`WindowsWhlBuilder_cuda.bat`** — comment and example:

```bat
rem Optional args (repeatable): FORCE_CXX11_ABI TRUE|FALSE   CUDA_ARCH 80;90;100;120
rem set FLASH_ATTN_CUDA_ARCHS=80;90;100;120
```

---

## 6. Expected nvcc command line after fixes (cu132, default archs)

After commits `33c55ee` + `cf2f241` and the `/Zc:preprocessor` block above, a representative `nvcc` invocation should include:

- **Gencodes only:** `sm_80`, `sm_90`, `sm_100` (via `100f` on 13.2), `sm_120` (via `120f` on 13.2), plus PTX `compute_120`
- **No** `sm_110` / `compute_110f`
- **MSVC:** `/Zc:preprocessor` (and `/Zc:__cplusplus`) on host compiles

Example fragment (order may vary):

```
... -gencode arch=compute_80,code=sm_80
    -gencode arch=compute_90,code=sm_90
    -gencode arch=compute_100f,code=sm_100
    -gencode arch=compute_120f,code=sm_120
    -gencode arch=compute_120,code=compute_120
    ... -Xcompiler /Zc:__cplusplus -Xcompiler /Zc:preprocessor ...
```

---

## 7. Verification performed on the cu132 wheel

| Check | Result |
|-------|--------|
| Wheel exists | `flash_attn-2.9.0+cu132torch2.12.0cxx11abiTRUE-cp313-cp313-win_amd64.whl` (~355 MiB) |
| `import flash_attn` | OK in `D:\USERFILES\fp8e4m3\venv` |
| `_test_a2.py` | 8/8 accuracy cases OK; backward finite |
| `_test_triton.py` | Triton JIT OK; backward finite |
| Runtime CUDA | Matches wheel tag **cu132** / torch **2.12.0+cu132** |

Functional kernel behavior (A-1, A-2, split-KV) is documented in `md/FA2_CHANGES_v1.2.md` and `md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md`; this document covers **only the toolkit / MSVC / gencode port**.

---

## 8. Reproducible recipe (checklist)

1. Install **PyTorch 2.12+cu132** and a venv; install **CUDA toolkit 13.2**; set `CUDA_HOME` and `PATH`.
2. Use repo at tag **`v1.2`** (or `main` with commits `33c55ee`, `cf2f241`, and `/Zc:preprocessor` in `setup.py`).
3. Do **not** set `FLASH_ATTN_CUDA_ARCHS` to include `110` unless you accept Thor entries being **stripped** with a warning.
4. Build:
   - `python setup.py build_ext --inplace` (dev), or
   - `WindowsWhlBuilder_cuda.bat` (wheel; sets `DISTUTILS_USE_SDK=1`).
5. Confirm ninja log has **no** `C1189` preprocessor error and **no** `compute_110f.cpp1.ii` fatal.
6. Run `_test_a2.py` and `_test_triton.py` on your GPU arch (`sm_120` needs `120` in the arch list).

---

## 9. Summary

| Problem | Fix | Where |
|---------|-----|--------|
| CCCL **C1189** traditional preprocessor | Add **`/Zc:preprocessor`** (+ keep `/Zc:__cplusplus`) for `DISTUTILS_USE_SDK=1` | `setup.py` Windows block |
| **`sm_110` gencode** / missing `compute_110f.cpp1.ii` | Default **`80;90;100;120`**; remove Thor gencode block; filter Thor in `cuda_archs()` | `setup.py` commits `33c55ee`, `cf2f241` |

Together, these changes align the fork’s **multi-arch FA2 wheel** with **CUDA 13.2** toolchains and **Blackwell consumer (`sm_120`)** hardware without claiming unsupported **Thor** SASS in the same binary.
