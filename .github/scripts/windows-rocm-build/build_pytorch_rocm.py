#!/usr/bin/env python
"""Builds the PyTorch wheel with ROCm on Windows.

Self-contained: it needs only Python (standard library), git, an activated MSVC
environment, and the `rocm` pip packages (the ROCm SDK) from a package index.

Steps (each prints what it runs):
  1. install the ROCm SDK wheels with pip (skipped without --rocm-version)
  2. ask the installed SDK where it is (python -m rocm_sdk path ...)
  3. hipify the PyTorch sources (tools/amd_build/build_amd.py)
  4. set up the build environment for ROCm on Windows
  5. write torch/_rocm_init.py and copy the OpenMP and libuv DLLs into torch/lib
  6. pip install the build requirements
  7. python -m build --wheel --no-isolation
  8. copy the wheel to --output-dir, install it, and import-check it

Use --dry-run to print the environment and the commands without running anything
that changes files.

Example (from a Visual Studio x64 environment):

    python build_pytorch_rocm.py ^
        --source-dir C:/b/pytorch ^
        --rocm-arch "gfx1200;gfx1201" ^
        --rocm-version 10.2.0a20261012 ^
        --index-url https://nightly.repo.amd.com/rocm/whl-next ^
        --output-dir C:/out --max-jobs 8 --clean
"""

import argparse
import ctypes
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

# ROCm libraries that torch/_rocm_init.py preloads on Windows. The names are the
# "shortnames" understood by rocm_sdk.initialize_process().
WINDOWS_PRELOADS = [
    "amd_comgr",
    "amdhip64",
    "hiprtc",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsparse",
    "hipsparselt",
    "hipsolver",
    "hipblaslt",
    "miopen",
    "hipdnn",
    "rocm-openblas",
]

# GPU architectures for which PyTorch's flash attention (aotriton) is built.
AOTRITON_ARCH_PREFIXES = ("gfx90a", "gfx942", "gfx950", "gfx11", "gfx12")

DRY_RUN = False


def run(args, cwd, env=None, check=True):
    """Runs a command, printing it (and the environment overrides) first."""
    args = [str(a) for a in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}", flush=True)
    if env:
        print(":: Env overrides:")
        for key, value in env.items():
            print(f"  {key}={value}")
    if DRY_RUN:
        return ""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(args, cwd=str(cwd), env=full_env, check=check)


def capture(args, cwd=None):
    """Runs a command and returns its stripped output. Fails loudly."""
    args = [str(a) for a in args]
    print(f"++ Capture $ {shlex.join(args)}", flush=True)
    result = subprocess.run(
        args, cwd=str(cwd) if cwd else None, capture_output=True, text=True
    )
    if result.returncode != 0:
        # Show what the command said before failing.
        print(result.stdout + result.stderr)
        raise subprocess.CalledProcessError(result.returncode, args)
    # Only standard output is the answer; warnings on standard error must not leak into it.
    return result.stdout.strip()


def rocm_sdk(*args):
    return capture([sys.executable, "-m", "rocm_sdk", *args])


def parse_args():
    p = argparse.ArgumentParser(description="Build PyTorch with ROCm on Windows.")
    p.add_argument("--source-dir", type=Path, required=True, help="PyTorch checkout")
    p.add_argument("--output-dir", type=Path, required=True, help="where the wheel is copied")
    p.add_argument("--rocm-arch", required=True, help="GPU targets, e.g. 'gfx1200;gfx1201'")
    p.add_argument("--rocm-version", default="", help="rocm SDK version to install (omit to use the installed one)")
    p.add_argument("--index-url", default="", help="pip index that serves the rocm packages")
    p.add_argument("--max-jobs", default="", help="limit parallel compile jobs (MAX_JOBS)")
    p.add_argument("--build-number", default="1")
    p.add_argument("--clean", action="store_true", help="remove build/ and dist/ first")
    p.add_argument("--skip-hipify", action="store_true")
    p.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="print memory and disk use into the log this often during the build (0 = off)",
    )
    p.add_argument("--dry-run", action="store_true", help="print, do not change anything")
    return p.parse_args()


def install_rocm(args):
    if not args.rocm_version:
        print("--- Not installing the ROCm SDK (no --rocm-version); using the installed one")
        return
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--pre"]
    if args.index_url:
        cmd += ["--index-url", args.index_url]
    cmd.append(f"rocm[libraries,devel]=={args.rocm_version}")
    run(cmd, cwd=Path.cwd())


def git_short_sha(source_dir):
    try:
        return capture(
            ["git", "-c", f"safe.directory={source_dir}", "rev-parse", "--short=8", "HEAD"],
            cwd=source_dir,
        )
    except Exception as exc:  # not a git checkout (for example in a dry run)
        print(f"WARNING: could not read the git commit in {source_dir}: {exc}")
        return ""


def compute_version(source_dir, sdk_version):
    base = (source_dir / "version.txt").read_text().strip()
    sha = git_short_sha(source_dir)
    name = "devrocm" if "dev" in sdk_version else "rocm"
    local = f"{name}{sdk_version.replace('+', '-')}"
    if sha:
        local = f"git{sha}.{local}"
    return f"{base}+{local}"


def flash_attention_on(arch_list):
    return any(a.startswith(AOTRITON_ARCH_PREFIXES) for a in arch_list)


def build_env(args, rocm_root, rocm_bin, rocm_cmake, version, sdk_version):
    arch = args.rocm_arch.replace(",", ";")
    llvm_bin = rocm_root / "lib" / "llvm" / "bin"
    flash = "ON" if flash_attention_on(arch.split(";")) else "OFF"
    env = {
        "PYTHONUTF8": "1",
        # --- where ROCm is
        "CMAKE_PREFIX_PATH": str(rocm_cmake),
        "ROCM_HOME": str(rocm_root),
        "ROCM_PATH": str(rocm_root),
        "PATH": str(rocm_bin) + os.pathsep + os.environ.get("PATH", ""),
        # --- what to build
        "PYTORCH_ROCM_ARCH": arch,
        "USE_ROCM": "ON",
        "USE_CUDA": "OFF",
        "USE_MPI": "OFF",
        "USE_NUMA": "OFF",
        "USE_GLOO": "ON",
        "USE_KINETO": os.environ.get("USE_KINETO", "OFF"),
        "USE_FLASH_ATTENTION": flash,
        "USE_MEM_EFF_ATTENTION": flash,
        "BUILD_TEST": "0",
        # --- compiler: ROCm's clang in MSVC-compatible mode, MSVC environment already active
        "DISTUTILS_USE_SDK": "1",
        "HIP_CLANG_PATH": llvm_bin.resolve().as_posix(),
        "CC": str((llvm_bin / "clang-cl.exe").resolve()),
        "CXX": str((llvm_bin / "clang-cl.exe").resolve()),
        # --- wheel metadata
        "PYTORCH_BUILD_VERSION": version,
        "PYTORCH_BUILD_NUMBER": args.build_number,
        "PYTORCH_EXTRA_INSTALL_REQUIREMENTS": f"rocm[libraries]=={sdk_version}",
    }
    if args.max_jobs:
        env["MAX_JOBS"] = str(args.max_jobs)
    # clang must look for the device libraries in the SDK, not guess.
    bitcode = rocm_root / "lib" / "llvm" / "amdgcn" / "bitcode"
    if bitcode.exists():
        env["HIP_DEVICE_LIB_PATH"] = str(bitcode)
    else:
        print(f"WARNING: {bitcode} not found; relying on clang's search")
    # OpenBLAS shipped with the SDK.
    host_math = rocm_root / "lib" / "host-math"
    if host_math.exists():
        env.update({"BLAS": "OpenBLAS", "OpenBLAS_HOME": str(host_math), "OpenBLAS_LIB_NAME": "rocm-openblas"})
    else:
        print(f"WARNING: {host_math} not found; building without OpenBLAS")
    return env


def write_rocm_init(source_dir, sdk_version):
    """torch/__init__.py imports torch._rocm_init if it exists (see upstream torch/__init__.py)."""
    names = ", ".join(f"'{n}'" for n in WINDOWS_PRELOADS)
    text = textwrap.dedent(
        f"""
        def initialize():
            import rocm_sdk
            rocm_sdk.initialize_process(
                preload_shortnames=[{names}],
                check_version='{sdk_version}')
        """
    )
    target = source_dir / "torch" / "_rocm_init.py"
    print(f"--- Writing {target}")
    if not DRY_RUN:
        target.write_text(text)


def copy_dll(src, dest_dir, what):
    print(f"--- Copying {what}: {src} -> {dest_dir}")
    if not DRY_RUN:
        shutil.copy2(src, dest_dir)


def copy_runtime_dlls(source_dir):
    lib = source_dir / "torch" / "lib"
    # OpenMP: torch_cpu.dll needs it. MSVC's copy is used because MSVC is active.
    redist = os.environ.get("VCToolsRedistDir", "")
    if not redist:
        raise RuntimeError("VCToolsRedistDir is not set: run from an activated Visual Studio environment")
    omp = sorted(Path(redist).rglob("libomp140.x86_64.dll"))
    if not omp:
        raise RuntimeError(f"libomp140.x86_64.dll not found under {redist}")
    copy_dll(omp[0], lib, "libomp")
    # libuv: needed at run time by torch.distributed (Gloo). Optional.
    libuv_root = os.environ.get("libuv_ROOT", "")
    if libuv_root:
        uv = Path(libuv_root) / "bin" / "uv.dll"
        if not uv.exists():
            raise RuntimeError(f"libuv_ROOT is set but {uv} does not exist")
        copy_dll(uv, lib, "libuv")


class _MemoryStatus(ctypes.Structure):
    """Windows MEMORYSTATUSEX, filled by GlobalMemoryStatusEx."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_status_gb():
    """Returns (ram_used, ram_total, commit_used, commit_limit) in GB."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    gb = 1024 ** 3
    return (
        (status.ullTotalPhys - status.ullAvailPhys) / gb,
        status.ullTotalPhys / gb,
        (status.ullTotalPageFile - status.ullAvailPageFile) / gb,
        status.ullTotalPageFile / gb,
    )


def heartbeat_line(started, disk_path, peak):
    ram_used, ram_total, commit_used, commit_limit = memory_status_gb()
    free_disk = shutil.disk_usage(disk_path).free / 1024 ** 3
    peak["ram_used"] = max(peak.get("ram_used", 0.0), ram_used)
    peak["commit_used"] = max(peak.get("commit_used", 0.0), commit_used)
    peak["min_free_disk"] = min(peak.get("min_free_disk", free_disk), free_disk)
    minutes = (time.time() - started) / 60
    return (
        f"[heartbeat {time.strftime('%H:%M:%S')}] build {minutes:.1f} min | "
        f"RAM used {ram_used:.1f}/{ram_total:.1f} GB | "
        f"commit {commit_used:.1f}/{commit_limit:.1f} GB | "
        f"{os.path.splitdrive(str(disk_path))[0] or 'disk'} free {free_disk:.1f} GB"
    )


def start_heartbeat(interval, disk_path):
    """Prints memory and disk use into the build output every `interval` seconds.

    Because the lines are part of the build's own log, they reach the CI log live, so the
    last numbers are available even if the machine dies during the build.
    Returns (stop_event, thread, peak) where peak collects the maximum values seen."""
    stop = threading.Event()
    peak = {}
    started = time.time()

    def emit():
        try:
            print(heartbeat_line(started, disk_path, peak), flush=True)
        except Exception as exc:  # never let monitoring break the build
            print(f"[heartbeat] could not read memory/disk: {exc}", flush=True)

    def loop():
        emit()
        while not stop.wait(interval):
            emit()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop, thread, peak


def find_wheel(dist_dir):
    wheels = sorted(dist_dir.glob("torch-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one torch wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def main():
    global DRY_RUN
    args = parse_args()
    DRY_RUN = args.dry_run
    if platform.system() != "Windows":
        sys.exit("This script is for Windows.")
    source_dir = args.source_dir.resolve()
    if not (source_dir / "version.txt").exists():
        sys.exit(f"{source_dir} does not look like a PyTorch checkout (no version.txt)")
    print(f"Python {sys.version.split()[0]}  source={source_dir}  dry_run={DRY_RUN}")

    install_rocm(args)

    sdk_version = rocm_sdk("version")
    rocm_cmake = Path(rocm_sdk("path", "--cmake"))
    rocm_bin = Path(rocm_sdk("path", "--bin"))
    rocm_root = Path(rocm_sdk("path", "--root"))
    print(f"rocm_sdk {sdk_version}\n  root  {rocm_root}\n  bin   {rocm_bin}\n  cmake {rocm_cmake}")

    if not args.skip_hipify:
        run([sys.executable, source_dir / "tools" / "amd_build" / "build_amd.py"], cwd=source_dir)

    version = compute_version(source_dir, sdk_version)
    print(f"PYTORCH_BUILD_VERSION = {version}")
    env = build_env(args, rocm_root, rocm_bin, rocm_cmake, version, sdk_version)

    write_rocm_init(source_dir, sdk_version)
    if DRY_RUN and not os.environ.get("VCToolsRedistDir"):
        print("--- (dry run) VCToolsRedistDir not set; skipping the DLL copy check")
    else:
        copy_runtime_dlls(source_dir)

    py = sys.executable
    run([py, "-m", "pip", "uninstall", "torch", "-y"], cwd=tempfile.gettempdir())
    run([py, "-m", "pip", "install", "-r", source_dir / "requirements.txt"], cwd=source_dir)
    backend = [py, "-m", "pip", "install", "build"]
    if (source_dir / "requirements-build.txt").exists():
        backend += ["-r", source_dir / "requirements-build.txt"]
    run(backend, cwd=source_dir)
    # ninja 1.13.0 has a link.exe response-file bug on Windows (ninja-build/ninja#2616).
    run([py, "-m", "pip", "install", "ninja>=1.13.2"], cwd=source_dir)

    dist = source_dir / "dist"
    if not DRY_RUN:
        shutil.rmtree(dist, ignore_errors=True)
        if args.clean:
            shutil.rmtree(source_dir / "build", ignore_errors=True)
    print("+++ Building PyTorch")
    stop = thread = peak = None
    if not DRY_RUN and args.heartbeat_seconds > 0:
        stop, thread, peak = start_heartbeat(args.heartbeat_seconds, source_dir)
    try:
        run([py, "-m", "build", "--wheel", "--no-isolation"], cwd=source_dir, env=env)
    finally:
        if stop is not None:
            stop.set()
            thread.join(timeout=5)
            print(
                f"[heartbeat] peak RAM used {peak.get('ram_used', 0):.1f} GB, "
                f"peak commit {peak.get('commit_used', 0):.1f} GB, "
                f"minimum free disk {peak.get('min_free_disk', 0):.1f} GB",
                flush=True,
            )
    if DRY_RUN:
        print("--- dry run finished: nothing was built")
        return

    wheel = find_wheel(dist)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, args.output_dir)
    print(f"Wheel: {wheel.name} ({wheel.stat().st_size} bytes) -> {args.output_dir}")

    run([py, "-m", "pip", "install", wheel], cwd=tempfile.gettempdir())
    # Import check. torch.cuda.is_available() is False on a machine without a GPU,
    # which is fine; the import itself must work.
    check = "import torch; print(torch.__version__); print('hip', torch.version.hip); print('available', torch.cuda.is_available())"
    result = subprocess.run([py, "-c", check], cwd=tempfile.gettempdir(), capture_output=True, text=True)
    print(result.stdout + result.stderr)
    if result.returncode != 0:
        sys.exit("import check of the built torch failed")


if __name__ == "__main__":
    main()
