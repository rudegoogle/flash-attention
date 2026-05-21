# Background and Complete Explanation of GitHub Actions Workflows Removal in the Fork

This document provides a complete explanation and technical rationale for the removal of all GitHub Actions workflows (including FA4-related workflows) from this repository, which is a fork of the official repository (Dao-AILab/flash-attention).

## 1. List of All Deleted Files

To prevent errors and accidents caused by unnecessary cloud CI executions, all of the following workflow files have been completely deleted:

- `.github/workflows/publish-fa4.yml`
- `.github/workflows/publish.yml`
- `.github/workflows/_build.yml`
- `.github/workflows/build.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/pre-commit.yaml`

## 2. Error Details and Root Cause

If the CI workflows from the official repository were left in this fork, the following problems and errors would occur:

1. **Forced Execution of Linux-Specific Builds**:
   The official CI workflows (such as `build.yml`, `_build.yml`, `ci.yml`) are all designed to run on Linux runners (`ubuntu-latest` or NVIDIA Docker environments) and depend on many Linux-specific `.so` libraries (especially CuTeDSL for FA4). Since this fork repository is **focused primarily on native building of the FA2 C++/CUDA kernels in a Windows environment (MSVC)**, executing these CI pipelines would inevitably result in build errors due to architectural mismatches.
2. **Unintended Publish Attempts and Authentication Errors**:
   Workflows like `publish-fa4.yml` and `publish.yml` attempt to automatically publish packages to PyPI. Because this fork does not have the authentication secrets required to upload to the official PyPI repository, these workflows would invariably fail with authentication errors during scheduled runs or tag pushes.
3. **Resource Waste and Noise**:
   Builds would fail on every push or pull request, spamming the fork maintainers with useless error notifications and wasting GitHub Actions computing resources.

The root cause was that "official CI scripts for Linux and official release deployments were mixed into a fork whose sole purpose is local Windows/FA2 builds (managed manually via `.bat` scripts)."

## 3. Full Source Code of All Deleted Files

Below is the complete source code of the files that caused the errors, along with an explanation of what each file did and why it was unnecessary/harmful in this fork.

### .github/workflows/publish-fa4.yml
**Reason for Deletion**: A PyPI publish workflow exclusively for FA4 (CuTeDSL). It cannot be built in this Windows/FA2 fork and was the root cause of PyPI authentication errors and unintended automated deployment risks.

```yaml
name: Publish flash-attn-4 to PyPI

on:
  push:
    tags:
      - 'fa4-v*'
  schedule:
    - cron: '0 8 * * 3'  # Wednesday 08:00 UTC
  workflow_dispatch:

permissions:
  contents: write

jobs:
  prepare-release:
    runs-on: ubuntu-latest
    outputs:
      release_tag: ${{ steps.resolve-tag.outputs.release_tag }}
    steps:
    - name: Require default branch for manual runs
      if: github.event_name == 'workflow_dispatch'
      run: |
        if [ "${{ github.ref_name }}" != "${{ github.event.repository.default_branch }}" ]; then
          echo "::error::Run this workflow from ${{ github.event.repository.default_branch }} only"
          exit 1
        fi
    - uses: actions/checkout@v4
      if: github.event_name != 'push'
      with:
        ref: ${{ github.event.repository.default_branch }}
        fetch-depth: 0
    - uses: actions/setup-python@v5
      if: github.event_name != 'push'
      with:
        python-version: '3.12'
    - name: Bump beta tag
      if: github.event_name != 'push'
      id: bump
      run: python .github/scripts/bump_beta_tag.py --push
    - name: Resolve release tag
      id: resolve-tag
      run: |
        if [ "${{ github.event_name }}" = "push" ]; then
          echo "release_tag=${GITHUB_REF#refs/tags/}" >> "$GITHUB_OUTPUT"
        else
          echo "release_tag=${{ steps.bump.outputs.next_tag }}" >> "$GITHUB_OUTPUT"
        fi

  build:
    needs: prepare-release
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        ref: ${{ needs.prepare-release.outputs.release_tag }}
        fetch-depth: 0
    - uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    - name: Install build dependencies
      run: pip install build twine
    - name: Extract version from tag
      id: strip-prefix
      run: |
        TAG="${{ needs.prepare-release.outputs.release_tag }}"
        echo "version=${TAG#fa4-v}" >> "$GITHUB_OUTPUT"
    - name: Build package
      run: python -m build
      working-directory: flash_attn/cute
      env:
        SETUPTOOLS_SCM_PRETEND_VERSION: ${{ steps.strip-prefix.outputs.version }}
    - name: Check package metadata
      run: twine check dist/*
      working-directory: flash_attn/cute
    - name: Store distribution packages
      uses: actions/upload-artifact@v4
      with:
        name: python-package-distributions
        path: flash_attn/cute/dist/

  github-release:
    needs: [prepare-release, build]
    runs-on: ubuntu-latest
    steps:
    - name: Download distribution packages
      uses: actions/download-artifact@v4
      with:
        name: python-package-distributions
        path: dist/
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v2
      with:
        tag_name: ${{ needs.prepare-release.outputs.release_tag }}
        files: dist/*
        generate_release_notes: true
        prerelease: ${{ contains(needs.prepare-release.outputs.release_tag, '.beta') }}

  publish-to-pypi:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/flash-attn-4
    permissions:
      id-token: write
    steps:
    - name: Download distribution packages
      uses: actions/download-artifact@v4
      with:
        name: python-package-distributions
        path: dist/
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

### .github/workflows/publish.yml
**Reason for Deletion**: A workflow for building and publishing FA2 Linux wheels to PyPI. Since the main purpose of this fork is local Windows builds, this workflow was unnecessary and harmful.

```yaml
# This workflow will:
# - Create a new Github release
# - Build wheels for supported architectures
# - Deploy the wheels to the Github release
# - Release the static code to PyPi
# For more information see: https://help.github.com/en/actions/language-and-framework-guides/using-python-with-github-actions#publishing-to-package-registries

name: Build wheels and deploy

on:
  create:
    tags:
      - v*

jobs:
  setup_release:
    name: Create Release
    runs-on: ubuntu-latest
    outputs:
      release-version: ${{ steps.extract_branch.outputs.branch }}
    steps:
      - name: Get the tag version
        id: extract_branch
        run: echo "branch=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT
        shell: bash
      - name: Create Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: gh release create ${{ steps.extract_branch.outputs.branch }} --repo $GITHUB_REPOSITORY --title ${{ steps.extract_branch.outputs.branch }} --generate-notes
        shell: bash

  build_wheels:
    name: Build Wheel
    needs: setup_release
    strategy:
      fail-fast: false
      matrix:
        # Using ubuntu-22.04 instead of 24.04 for more compatibility (glibc). Ideally we'd use the
        # manylinux docker image, but I haven't figured out how to install CUDA on manylinux.
        os: [ubuntu-22.04, ubuntu-22.04-arm]
        python-version: ["3.10", "3.11", "3.12", "3.13"]
        torch-version: ["2.6.0", "2.7.1", "2.8.0", "2.9.1", "2.10.0"]
        cuda-version: ["12.9.1", "13.0.1"]
        # We need separate wheels that either uses C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI) or not.
        # Pytorch wheels currently don't use it, but nvcr images have Pytorch compiled with C++11 ABI.
        # Without this we get import error (undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs)
        # when building without C++11 ABI and using it on nvcr images.
        cxx11_abi: ["FALSE", "TRUE"]
        exclude:
          # CUDA 13.0 is only supported by PyTorch 2.9+
          - torch-version: "2.6.0"
            cuda-version: "13.0.1"
          - torch-version: "2.7.1"
            cuda-version: "13.0.1"
          - torch-version: "2.8.0"
            cuda-version: "13.0.1"
          # No aarch64 PyTorch wheels for 2.6.0
          - torch-version: "2.6.0"
            os: ubuntu-22.04-arm
          # PyTorch 2.7+ pip wheels use CXX11_ABI=1 by default, no need for FALSE
          - torch-version: "2.7.1"
            cxx11_abi: "FALSE"
          - torch-version: "2.8.0"
            cxx11_abi: "FALSE"
          - torch-version: "2.9.1"
            cxx11_abi: "FALSE"
          - torch-version: "2.10.0"
            cxx11_abi: "FALSE"
    uses: ./.github/workflows/_build.yml
    with:
      runs-on: ${{ matrix.os }}
      python-version: ${{ matrix.python-version }}
      cuda-version: ${{ matrix.cuda-version }}
      torch-version: ${{ matrix.torch-version }}
      cxx11_abi: ${{ matrix.cxx11_abi }}
      release-version: ${{ needs.setup_release.outputs.release-version }}
      upload-to-release: true

  publish_package:
    name: Publish package
    needs: [build_wheels]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: |
          pip install ninja packaging wheel twine
          # Install latest setuptools with support for pypi metadata 2.2 (improved compat w/ uv)
          pip install setuptools==75.8.0
          # We don't want to download anything CUDA-related here
          pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Build core package
        env:
          FLASH_ATTENTION_SKIP_CUDA_BUILD: "TRUE"
        run: |
          python setup.py sdist --dist-dir=dist
      - name: Deploy
        env:
          TWINE_USERNAME: "__token__"
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          python -m twine upload dist/*
```

### .github/workflows/_build.yml
**Reason for Deletion**: A reusable workflow that performs builds and tests inside Linux Docker containers. It is completely incompatible with Windows builds.

```yaml
name: ~Build wheel template

on:
  workflow_call:
    inputs:
      runs-on:
        description: "The runner to use for the build"
        required: true
        type: string
      python-version:
        description: "The Python version to use for the build"
        required: true
        type: string
      cuda-version:
        description: "The CUDA version to use for the build"
        required: true
        type: string
      torch-version:
        description: "The PyTorch version to use for the build"
        required: true
        type: string
      cxx11_abi:
        description: "The C++11 ABI to use for the build"
        required: true
        type: string
      upload-to-release:
        description: "Upload wheel to this release"
        required: false
        type: boolean
        default: false
      release-version:
        description: "Upload wheel to this release"
        required: false
        type: string

defaults:
  run:
    shell: bash -x -e -u -o pipefail {0}

jobs:
  build-wheel:
    runs-on: ${{ inputs.runs-on }}
    name: Build wheel (${{ inputs.release-version }}-${{ inputs.python-version }}-${{ inputs.cuda-version }}-${{ inputs.torch-version }}-${{ inputs.cxx11_abi }})
    steps:
      - name: Checkout
        uses: actions/checkout@v5
        with:
          ref: ${{ inputs.release-version }}
          submodules: recursive

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}

      - name: Set CUDA and PyTorch versions
        run: |
          echo "MATRIX_CUDA_VERSION=$(echo ${{ inputs.cuda-version }} | awk -F \. {'print $1 $2'})" >> $GITHUB_ENV
          echo "MATRIX_TORCH_VERSION=$(echo ${{ inputs.torch-version }} | awk -F \. {'print $1 "." $2'})" >> $GITHUB_ENV
          echo "WHEEL_CUDA_VERSION=$(echo ${{ inputs.cuda-version }} | awk -F \. {'print $1'})" >> $GITHUB_ENV
          echo "MATRIX_PYTHON_VERSION=$(echo ${{ inputs.python-version }} | awk -F \. {'print $1 $2'})" >> $GITHUB_ENV

      - name: Free up disk space
        if: ${{ runner.os == 'Linux' }}
        # https://github.com/easimon/maximize-build-space/blob/master/action.yml
        # https://github.com/easimon/maximize-build-space/tree/test-report
        run: |
          sudo rm -rf /usr/share/dotnet
          sudo rm -rf /opt/ghc
          sudo rm -rf /opt/hostedtoolcache/CodeQL

      - name: Set up swap space
        if: runner.os == 'Linux'
        uses: pierotofy/set-swap-space@v1.0
        with:
          swap-size-gb: 10

      - name: Install CUDA ${{ inputs.cuda-version }}
        if: ${{ inputs.cuda-version != 'cpu' }}
        uses: Jimver/cuda-toolkit@v0.2.30
        id: cuda-toolkit
        with:
          cuda: ${{ inputs.cuda-version }}
          linux-local-args: '["--toolkit"]'
          # default method is "local", and we're hitting some error with caching for CUDA 11.8 and 12.1
          # method: ${{ (inputs.cuda-version == '11.8.0' || inputs.cuda-version == '12.1.0') && 'network' || 'local' }}
          method: "network"
          sub-packages: '["nvcc"]'

      - name: Install PyTorch ${{ inputs.torch-version }}+cu${{ inputs.cuda-version }}
        run: |
          pip install --upgrade pip
          # With python 3.13 and torch 2.5.1, unless we update typing-extensions, we get error
          # AttributeError: attribute '__default__' of 'typing.ParamSpec' objects is not writable
          pip install typing-extensions==4.12.2
          # Pick the highest available PyTorch wheel CUDA version that doesn't exceed system CUDA
          export TORCH_CUDA_VERSION=$(python -c "from os import environ as env; \
            available = { \
              '2.6': [118, 124, 126], \
              '2.7': [118, 126, 128], \
              '2.8': [126, 128, 129], \
              '2.9': [126, 128, 130], \
              '2.10': [126, 128, 130], \
            }[env['MATRIX_TORCH_VERSION']]; \
            sys_cuda = int(env['MATRIX_CUDA_VERSION']); \
            print(max(v for v in available if v <= sys_cuda))" \
          )
          # detect if we're on ARM
          if [ "$(uname -m)" = "aarch64" ] || [ "$(uname -m)" = "arm64" ]; then
              PLAT=linux_aarch64
          else
              PLAT=manylinux_2_27_x86_64.manylinux_2_28_x86_64
          fi
          echo "PLAT=$PLAT" >> $GITHUB_ENV
          if [[ ${{ inputs.torch-version }} == *"dev"* ]]; then
            # pip install --no-cache-dir --pre torch==${{ inputs.torch-version }} --index-url https://download.pytorch.org/whl/nightly/cu${TORCH_CUDA_VERSION}
            # Can't use --no-deps because we need cudnn etc.
            # Hard-coding this version of pytorch-triton for torch 2.9.0.dev20250904
            pip install jinja2
            TRITON_URL=https://download.pytorch.org/whl/nightly/pytorch_triton-3.4.0%2Bgitf7888497-cp${MATRIX_PYTHON_VERSION}-cp${MATRIX_PYTHON_VERSION}-${PLAT}.whl
            TORCH_URL=https://download.pytorch.org/whl/nightly/cu${TORCH_CUDA_VERSION}/torch-${{ inputs.torch-version }}%2Bcu${TORCH_CUDA_VERSION}-cp${MATRIX_PYTHON_VERSION}-cp${MATRIX_PYTHON_VERSION}-manylinux_2_28_$(uname -m).whl
            pip install --no-cache-dir --pre "${TRITON_URL}"
            pip install --no-cache-dir --pre "${TORCH_URL}"
          else
            pip install --no-cache-dir torch==${{ inputs.torch-version }} --index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}
          fi
          nvcc --version
          python --version
          python -c "import torch; print('PyTorch:', torch.__version__)"
          python -c "import torch; print('CUDA:', torch.version.cuda)"
          python -c "from torch.utils import cpp_extension; print (cpp_extension.CUDA_HOME)"

      - name: Restore build cache
        uses: actions/cache/restore@v4
        with:
          path: build.tar
          key: build-${{ inputs.release-version }}-${{ inputs.python-version }}-${{ inputs.cuda-version }}-${{ inputs.torch-version }}-${{ inputs.cxx11_abi }}-${{ github.run_number }}-${{ github.run_attempt }}
          restore-keys: |
            build-${{ inputs.release-version }}-${{ inputs.python-version }}-${{ inputs.cuda-version }}-${{ inputs.torch-version }}-${{ inputs.cxx11_abi }}-

      - name: Unpack build cache
        run: |
          echo ::group::Adjust timestamps
          sudo find / -exec touch -t 197001010000 {} + || true
          echo ::endgroup::

          if [ -f build.tar ]; then
            find . -mindepth 1 -maxdepth 1 ! -name 'build.tar' -exec rm -rf {} +
            tar -xpvf build.tar -C .
          else
            echo "No build.tar found, skipping"
          fi

          ls -al ./
          ls -al build/ || true
          ls -al csrc/ || true

      - name: Build wheel
        id: build_wheel
        run: |
          # We want setuptools >= 49.6.0 otherwise we can't compile the extension if system CUDA version is 11.7 and pytorch cuda version is 11.6
          # https://github.com/pytorch/pytorch/blob/664058fa83f1d8eede5d66418abff6e20bd76ca8/torch/utils/cpp_extension.py#L810
          # However this still fails so I'm using a newer version of setuptools
          pip install setuptools==75.8.0
          pip install ninja packaging wheel
          export PATH=/usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
          export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
          # Limit MAX_JOBS otherwise the github runner goes OOM
          # nvcc 11.8 can compile with 2 jobs, but nvcc 12.3 goes OOM

          export MAX_JOBS=$([ "$MATRIX_CUDA_VERSION" == "129" ] || [ "$MATRIX_CUDA_VERSION" == "130" ] && echo 1 || echo 2)
          export NVCC_THREADS=2
          export FLASH_ATTENTION_FORCE_BUILD="TRUE"
          export FLASH_ATTENTION_FORCE_CXX11_ABI=${{ inputs.cxx11_abi }}

          # 5h timeout since GH allows max 6h and we want some buffer
          EXIT_CODE=0
          timeout 5h python setup.py bdist_wheel --dist-dir=dist || EXIT_CODE=$?

          if [ $EXIT_CODE -eq 0 ]; then
            tmpname=cu${WHEEL_CUDA_VERSION}torch${MATRIX_TORCH_VERSION}cxx11abi${{ inputs.cxx11_abi }}
            wheel_name=$(ls dist/*whl | xargs -n 1 basename | sed "s/-/+$tmpname-/2")
            ls dist/*whl |xargs -I {} mv {} dist/${wheel_name}
            echo "wheel_name=${wheel_name}" >> $GITHUB_ENV
          fi

          # Store exit code in GitHub env for later steps
          echo "build_exit_code=$EXIT_CODE" | tee -a "$GITHUB_OUTPUT"

          # Do not fail the job if timeout killed the build
          exit $EXIT_CODE

      - name: Log build logs after timeout
        if: always() && steps.build_wheel.outputs.build_exit_code == 124
        run: |
          ls -al ./
          tar -cvf build.tar . --atime-preserve=replace

      - name: Save build cache timeout
        if: always() && steps.build_wheel.outputs.build_exit_code == 124
        uses: actions/cache/save@v4
        with:
          key: build-${{ inputs.release-version }}-${{ inputs.python-version }}-${{ inputs.cuda-version }}-${{ inputs.torch-version }}-${{ inputs.cxx11_abi }}-${{ github.run_number }}-${{ github.run_attempt }}
          path: build.tar

      - name: Log Built Wheels
        run: |
          ls dist

      - name: Get Release with tag
        id: get_current_release
        uses: joutvhu/get-release@v1
        with:
          tag_name: ${{ inputs.release-version }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Upload Release Asset
        id: upload_release_asset
        if: inputs.upload-to-release
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ steps.get_current_release.outputs.upload_url }}
          asset_path: ./dist/${{env.wheel_name}}
          asset_name: ${{env.wheel_name}}
          asset_content_type: application/*
```

### .github/workflows/build.yml
**Reason for Deletion**: A matrix build workflow that calls `_build.yml` across various environments. It was a primary source of cloud build errors.

```yaml
name: Build wheels

on:
  workflow_dispatch:
    inputs:
      runs-on:
        description: "The runner to use for the build"
        required: true
        type: string
        default: ubuntu-22.04
      python-version:
        description: "The Python version to use for the build"
        required: true
        type: string
      cuda-version:
        description: "The CUDA version to use for the build"
        required: true
        type: string
      torch-version:
        description: "The PyTorch version to use for the build"
        required: true
        type: string
      cxx11_abi:
        description: "Enable torch flag C++11 ABI (TRUE/FALSE)"
        required: true
        type: string
      upload-to-release:
        description: "Upload wheel to this release"
        required: false
        type: boolean
        default: false
      release-version:
        description: "Upload wheel to this release"
        required: false
        type: string

jobs:
  build-wheels:
    uses: ./.github/workflows/_build.yml
    with:
      runs-on: ${{ inputs.runs-on }}
      python-version: ${{ inputs.python-version }}
      cuda-version: ${{ inputs.cuda-version }}
      torch-version: ${{ inputs.torch-version }}
      cxx11_abi: ${{ inputs.cxx11_abi }}
      upload-to-release: ${{ inputs.upload-to-release }}
      release-version: ${{ inputs.release-version }}
```

### .github/workflows/ci.yml
**Reason for Deletion**: Linux-based testing and linting validations triggered on PRs and pushes. Incompatible with our Windows-specific codebase.

```yaml
name: CI

on:
  push:
    branches: [main, ci-fix]

permissions:
  contents: read

env:
  CI_WORK_DIR: ${{ vars.CI_WORK_DIR || format('/scratch/user/{0}', github.actor) }}
  FA4_TEST_FILTER: "1024-1024-128-True-0-0.0-False-False-False-mha-dtype0 or 1024-1024-128-False-0-0.0-False-False-False-mha-dtype0"

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install ruff
        run: pip install ruff
      - name: Ruff check
        run: ruff check flash_attn/cute/ --extend-exclude "flash_attn/cute/flash_bwd.py,flash_attn/cute/flash_fwd.py,flash_attn/cute/flash_fwd_sm100.py,flash_attn/cute/interface.py"
      - name: Ruff format
        run: ruff format --check flash_attn/cute/ --exclude "flash_attn/cute/flash_bwd.py,flash_attn/cute/flash_fwd.py,flash_attn/cute/flash_fwd_sm100.py,flash_attn/cute/interface.py"

  # Upstream FA4 GPU tests require a self-hosted B200 runner (not available on this fork).
  fa4-correctness-and-benchmark:
    if: false
    strategy:
      fail-fast: false
      matrix:
        gpu: [b200]
    runs-on: [self-hosted, '${{ matrix.gpu }}']
    name: fa4-correctness-and-benchmark (${{ matrix.gpu }})
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: ./.github/actions/gpu-test
        with:
          test-filter: ${{ env.FA4_TEST_FILTER }}
          fa4_image_cu129: "togethercomputer/training-performance:flash-attn-cu12.9-26.03.25@sha256:304a5c3d2b3a75b151cd2a964cd26d444e0d8b5686d63943df13378c9705f943"
          fa4_image_cu130: "togethercomputer/training-performance:flash-attn-cu13.0-26.04.01@sha256:56e50b056eb4d671410846c3483e843ee7bd0f5b13cb45b6f0d7eb8bd27694a5"
```

### .github/workflows/pre-commit.yaml
**Reason for Deletion**: Cloud-based code formatting validation. Removed to prioritize the local Windows development experience.

```yaml
name: Lint

on:
  pull_request:
    paths:
      - 'flash_attn/cute/flash_bwd_sm90.py'
      - 'flash_attn/cute/flash_bwd_preprocess.py'
      - 'flash_attn/cute/flash_bwd_postprocess.py'
      - 'flash_attn/cute/softmax.py'
      - '.pre-commit-config.yaml'
  push:
    branches:
      - main
    paths:
      - 'flash_attn/cute/flash_bwd_sm90.py'
      - 'flash_attn/cute/flash_bwd_preprocess.py'
      - 'flash_attn/cute/flash_bwd_postprocess.py'
      - 'flash_attn/cute/softmax.py'
      - '.pre-commit-config.yaml'

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.11'

      - name: Run pre-commit
        uses: pre-commit/action@v3.0.1
```

## 4. Significance of the Modification and Deletion (Summary)

The complete deletion (`git rm`) of all these files signifies the "purification of responsibilities" for this repository.
To specialize entirely in our unique goal—native building of FA2 (C++/CUDA) in a Windows environment—we stripped away unnecessary cloud CI scripts and consolidated operations into local batch files (`WindowsWhlBuilder_cuda.bat`). This guarantees the complete elimination of useless error notifications, wasted resources, and the risk of accidental deployment accidents.
