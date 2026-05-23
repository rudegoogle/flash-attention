#!/usr/bin/env python3
"""SASS hard-gate checker for Flash-Attention A-1/A-2 optimizations.

Parses ``cuobjdump --dump-sass`` output to verify the presence or absence
of specific SASS instructions within ``scale_apply_exp2`` symbols.

Exit codes:
    0  gate passes
    1  gate fails
    2  usage / environment error

Minimum usage::

    python bench/check_sass_gates.py \\
        --whl dist/flash_attn-*.whl --arch sm_120 --gate a2

Or with a pre-extracted binary::

    python bench/check_sass_gates.py \\
        --pyd flash_attn_2_cuda.cp312-win_amd64.pyd --arch sm_120 --gate a2

Reference: ``AI/FA2_BACKPORT_FROM_FA4_PLAN.md`` §6 (Phase-1 exit criterion),
§8.2 (SASS inspection row), §9 risk R8.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

GATES: dict[str, dict] = {
    "a2": {
        "symbol_re": re.compile(r"scale_apply_exp2"),
        "instruction": "FFMA.X2",
        "expect_present": True,
        "description": (
            "packed FMA (fma.rn.f32x2) must be present in scale_apply_exp2"
        ),
    },
    "b1": {
        "symbol_re": re.compile(r"scale_apply_exp2"),
        "instruction": "MUFU.EX2",
        "expect_present": False,
        "description": (
            "fused exp2 (MUFU.EX2) must NOT appear in scale_apply_exp2"
        ),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUDA_SEARCH_DIRS = [
    # Windows
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    # Linux common
    "/usr/local/cuda",
    "/opt/cuda",
]


def find_cuobjdump() -> str | None:
    """Auto-detect ``cuobjdump`` from PATH or well-known CUDA locations."""
    found = shutil.which("cuobjdump")
    if found:
        return found
    exe = "cuobjdump.exe" if sys.platform == "win32" else "cuobjdump"
    for base in _CUDA_SEARCH_DIRS:
        if not os.path.isdir(base):
            continue
        # Try versioned subdirs (newest first)
        try:
            versions = sorted(os.listdir(base), reverse=True)
        except OSError:
            continue
        for ver in versions:
            candidate = os.path.join(base, ver, "bin", exe)
            if os.path.isfile(candidate):
                return candidate
    return None


def extract_pyd_from_wheel(whl_path: str, tmp_dir: str) -> str:
    """Extract the CUDA extension binary (.pyd / .so) from a wheel."""
    with zipfile.ZipFile(whl_path, "r") as zf:
        candidates = [
            n for n in zf.namelist()
            if n.endswith((".pyd", ".so")) and "cuda" in n.lower()
        ]
        if not candidates:
            candidates = [n for n in zf.namelist() if n.endswith((".pyd", ".so"))]
        if not candidates:
            print("ERROR: no .pyd/.so found in wheel", file=sys.stderr)
            sys.exit(2)
        target = candidates[0]
        zf.extract(target, tmp_dir)
        return os.path.join(tmp_dir, target)


# ---------------------------------------------------------------------------
# SASS parser (streaming)
# ---------------------------------------------------------------------------

_RE_FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)")
_RE_TEXT_SEC = re.compile(r"^\s*\.text\.(\S+)\s*:")


def scan_sass(
    cuobjdump: str,
    binary: str,
    arch: str,
    gate: dict,
) -> list[tuple[str, int]]:
    """Run cuobjdump and stream-parse for instruction hits.

    Returns a list of ``(symbol_name, hit_count)`` for every
    ``scale_apply_exp2`` instantiation found in the specified arch.
    """
    cmd = [cuobjdump, "--dump-sass", "--gpu-architecture", arch, binary]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    assert proc.stdout is not None

    sym_re: re.Pattern = gate["symbol_re"]
    instr: str = gate["instruction"]

    results: list[tuple[str, int]] = []
    cur_sym: str | None = None
    in_target = False
    count = 0

    for line in proc.stdout:
        m = _RE_FUNCTION.match(line) or _RE_TEXT_SEC.match(line)
        if m:
            if in_target and cur_sym is not None:
                results.append((cur_sym, count))
            cur_sym = m.group(1)
            in_target = bool(sym_re.search(cur_sym))
            count = 0
            continue

        if in_target and instr in line:
            count += 1

    if in_target and cur_sym is not None:
        results.append((cur_sym, count))

    proc.wait()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SASS hard-gate checker for FA2 A-1/A-2 optimizations",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--whl", metavar="PATH", help="Path to flash_attn wheel (.whl)")
    src.add_argument("--pyd", metavar="PATH", help="Direct path to .pyd/.so binary")
    ap.add_argument(
        "--arch",
        default="sm_120",
        help="GPU architecture for SASS dump (default: sm_120)",
    )
    ap.add_argument(
        "--gate",
        required=True,
        choices=sorted(GATES),
        help="Gate to check: " + ", ".join(f"{k}" for k in sorted(GATES)),
    )
    ap.add_argument(
        "--cuobjdump",
        default=None,
        metavar="PATH",
        help="Explicit path to cuobjdump (auto-detected if omitted)",
    )
    args = ap.parse_args()

    cuobjdump = args.cuobjdump or find_cuobjdump()
    if not cuobjdump:
        print(
            "ERROR: cuobjdump not found. Install CUDA toolkit or pass --cuobjdump.",
            file=sys.stderr,
        )
        sys.exit(2)

    gate = GATES[args.gate]
    tmp_dir: str | None = None

    try:
        if args.whl:
            tmp_dir = tempfile.mkdtemp(prefix="check_sass_")
            binary = extract_pyd_from_wheel(args.whl, tmp_dir)
        else:
            binary = args.pyd

        if not os.path.isfile(binary):
            print(f"ERROR: binary not found: {binary}", file=sys.stderr)
            sys.exit(2)

        print(f"gate:   --gate {args.gate}  ({gate['description']})")
        print(f"arch:   {args.arch}")
        print(f"binary: {binary}  ({os.path.getsize(binary) / 1e6:.1f} MB)")
        print(f"look:   {gate['instruction']}  in *scale_apply_exp2*")
        print()

        results = scan_sass(cuobjdump, binary, args.arch, gate)

        if not results:
            print(
                f"WARN: no scale_apply_exp2 symbols found for {args.arch}. "
                f"The binary may not contain cubins for this architecture.",
            )
            sys.exit(1)

        total = 0
        for sym, cnt in results:
            tag = "HIT " if cnt > 0 else "MISS"
            print(f"  [{tag}]  {gate['instruction']} x{cnt:3d}  {sym}")
            total += cnt

        print()
        found = total > 0
        if gate["expect_present"]:
            if found:
                print(
                    f"PASS  {gate['instruction']} found "
                    f"({total} total across {len(results)} symbol(s))"
                )
                sys.exit(0)
            else:
                print(
                    f"FAIL  {gate['instruction']} NOT found in any "
                    f"scale_apply_exp2 symbol"
                )
                sys.exit(1)
        else:  # expect absent
            if not found:
                print(f"PASS  {gate['instruction']} absent as expected")
                sys.exit(0)
            else:
                print(
                    f"FAIL  {gate['instruction']} found ({total} total) "
                    f"— expected absent"
                )
                sys.exit(1)

    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
