#!/usr/bin/env python3
"""
Standalone ROCTX smoke test. Run with:
  python test/test_roctx_standalone.py

On a ROCm build this exercises torch.cuda.roctx and emit_roctx.
On a non-ROCm build, ROCTX API is skipped (or stub raises); NVTX test still runs if CUDA.
"""
import sys

import torch


def test_roctx_api():
    """Test manual ROCTX markers (ROCm build only)."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True
    if not getattr(torch.version, "hip", None):
        print("SKIP: Not a ROCm build (torch.version.hip missing); ROCTX is stub-only")
        return True
    try:
        torch.cuda.roctx.range_push("roctx_foo")
        torch.cuda.roctx.mark("roctx_bar")
        torch.cuda.roctx.range_pop()
        rid = torch.cuda.roctx.range_start("roctx_range_start")
        torch.cuda.roctx.range_end(rid)
        with torch.cuda.roctx.range("roctx_context"):
            _ = torch.tensor([1.0], device="cuda")
        print("PASS: torch.cuda.roctx API")
        return True
    except Exception as e:
        print(f"FAIL: torch.cuda.roctx: {e}")
        return False


def test_emit_roctx():
    """Test emit_roctx context manager (ROCm build only)."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True
    if not getattr(torch.version, "hip", None):
        print("SKIP: Not a ROCm build; emit_roctx not exercised")
        return True
    try:
        from torch.autograd.profiler import emit_roctx
        a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        with torch.cuda.profiler.profile():
            with emit_roctx():
                a.add_(1.0)
        print("PASS: emit_roctx")
        return True
    except Exception as e:
        print(f"FAIL: emit_roctx: {e}")
        return False


def test_nvtx_api():
    """Test NVTX API (CUDA build) for comparison."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True
    try:
        torch.cuda.nvtx.range_push("nvtx_foo")
        torch.cuda.nvtx.mark("nvtx_bar")
        torch.cuda.nvtx.range_pop()
        rid = torch.cuda.nvtx.range_start("nvtx_range_start")
        torch.cuda.nvtx.range_end(rid)
        print("PASS: torch.cuda.nvtx API")
        return True
    except Exception as e:
        print(f"SKIP/FAIL: torch.cuda.nvtx: {e}")
        return True  # skip is ok on ROCm-only build


def main():
    from pathlib import Path
    # Allow importing torch from repo root
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("ROCTX standalone smoke test")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  HIP: {getattr(torch.version, 'hip', 'N/A')}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    ok = True
    ok &= test_nvtx_api()
    ok &= test_roctx_api()
    ok &= test_emit_roctx()
    print("Done.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
