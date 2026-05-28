#!/usr/bin/env python3
"""
install_pytorch_wheels.py

Installs PyTorch wheels from a pip index URL.

Usage (from repo root):
    python .github/scripts/install_pytorch_wheels.py --index-url <URL> --amdgpu-family <FAMILY> [OPTIONS]

Examples:
    # Install latest versions
    python .github/scripts/install_pytorch_wheels.py \
        --index-url <BASE_INDEX_URL>/whl \
        --amdgpu-family gfx1250

    # Install specific versions (matching ROCm builds)
    python .github/scripts/install_pytorch_wheels.py \
        --index-url <BASE_INDEX_URL>/whl \
        --amdgpu-family gfx1250 \
        --torch-version "2.10.0+devrocm7.12.0.dev0.849eec43b..." \
        --torchaudio-version "2.11.0a0+devrocm7.12.0.dev0.849eec43b..." \
        --torchvision-version "0.25.0a0+devrocm7.12.0.dev0.849eec43b..."
"""

import argparse
import re
import subprocess
import sys
import urllib.parse
import urllib.request


# Package configuration: (name, always_install)
PACKAGES = {
    "torch": True,
    "torchaudio": True,
    "torchvision": True,
    "triton": False,
    "rocm[devel]": True,
}
PYTORCH_PKGS = ["torch", "torchaudio", "torchvision", "triton"]


def print_banner(title: str) -> None:
    """Print a formatted banner."""
    print("=" * 50)
    print(title)
    print("=" * 50)


def build_package_spec(name: str, version: str | None) -> str:
    """Build a pip package spec (e.g., 'torch==2.10.0' or 'torch')."""
    return f"{name}=={version}" if version else name

def get_latest_package_version_for_rocm(
    index_url: str, package_name: str, rocm_version: str, required: bool = True,
    version_prefix: str | None = None,
) -> str | None:
    """Return latest package version containing rocm_version by parsing the index HTML.

    If version_prefix is set (e.g. "2.9"), only versions whose base part starts
    with that prefix are considered.
    """

    # Build the URL for this package's index page (e.g. .../gfx1250/torch/).
    rocm_tag = f"rocm{rocm_version}"
    url = f"{index_url.rstrip('/')}/{package_name}/"
    # Fetch the package index page; on failure (e.g. 404, timeout) fail if always_install, else return None.
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error: failed to fetch index for {package_name}: {e}", file=sys.stderr)
        sys.exit(1)
    # Parse wheel links: format is package-VERSION-...whl (e.g. torch-0.26.0a0+rocm7.12...-cp312-....whl).
    # Version can contain dots and + (URL-encoded as %2B), so we capture everything up to .whl.
    pattern = re.compile(
        re.escape(package_name) + r"-(.+?)\.whl",
        re.IGNORECASE,
    )
    all_suffixes = [m.group(1).strip() for m in pattern.finditer(html)]
    # Keep only wheels whose version string contains the requested ROCm tag (e.g. rocm7.12.0a20260224).
    # Version is the first segment before "-" in the suffix; decode %2B to + for comparison.
    matching = []
    for s in all_suffixes:
        ver = s.split("-")[0]
        if rocm_tag in ver:
            matching.append(urllib.parse.unquote(ver))
    # Filter by version prefix (e.g. "2.9" matches "2.9.0+...", "2.9.1+...").
    if version_prefix and matching:
        matching = [v for v in matching if v.split("+")[0].startswith(version_prefix)]
    # No matching wheels: if required (always_install), fail; otherwise return None (package will be skipped).
    if not matching:
        if required:
            msg = f"Error: no wheel found for {package_name} with ROCm {rocm_version}"
            if version_prefix:
                msg += f" and version prefix {version_prefix}"
            print(msg, file=sys.stderr)
            sys.exit(1)
        return None
    # Pick the latest version by comparing all numeric parts including the ROCm date.
    def _key(v: str) -> tuple[int, ...]:
        try:
            return tuple(int(x) for x in re.split(r"[.\-a+]", v) if x.isdigit())
        except (ValueError, AttributeError):
            return (0,)
    return max(matching, key=_key)


def run_pip_install(
    index_url: str,
    packages: list[str],
    break_system_packages: bool = True,
    extra_index_url: str | None = None,
) -> None:
    """Run pip install with the given packages."""
    cmd = [sys.executable, "-m", "pip", "install", "--index-url", index_url]

    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url])

    if break_system_packages:
        cmd.append("--break-system-packages")

    cmd.extend(packages)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"Error: pip install failed with return code {result.returncode}")
        sys.exit(result.returncode)


def check_package(name: str) -> tuple[bool, str | None]:
    """Check if a package is installed and return (installed, version)."""
    try:
        module = __import__(name)
        return True, getattr(module, "__version__", "unknown")
    except ImportError:
        return False, None


def verify_installation() -> bool:
    """Verify PyTorch installation and print version info."""
    print_banner("Verifying Installation")

    # Check torch separately for ROCm info
    try:
        import torch as _torch

        version = getattr(_torch, "__version__", "unknown")
    except ImportError as e:
        print(f"Error: torch import failed ({e!r}). If wheels are installed, run rocm-sdk init first.")
        return False

    print(f"torch: {version}")

    hip_version = _torch.version.hip
    print(f"ROCm/HIP: {hip_version or 'not available'}")
    print(f"Built with ROCm: {hip_version is not None}")

    # Check other packages
    for name in ["torchaudio", "torchvision", "triton", "rocm"]:
        installed, version = check_package(name)
        status = version if installed else "not installed"
        print(f"{name}: {status}")

    return True


def list_installed_packages() -> None:
    """List installed torch-related packages."""
    print("\nInstalled PyTorch packages:")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        keywords = ["torch", "triton", "rocm"]
        for line in result.stdout.splitlines():
            if any(kw in line.lower() for kw in keywords):
                print(f"  {line}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Install PyTorch wheels from a pip index URL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--index-url", required=True, help="Base URL for PyTorch wheels index"
    )
    parser.add_argument(
        "--amdgpu-family", required=True, help="AMD GPU family (e.g., gfx1250)"
    )
    parser.add_argument(
        "--rocm-package-index-url",
        help="Optional base URL for ROCm Python packages when different from --index-url",
    )
    parser.add_argument(
        "--rocm-version",
        help="Optional. ROCm version (e.g. 7.12.0a20260126). When set without --torch-version: discovers and installs latest torch/torchaudio/torchvision/triton built for this ROCm. ",
    )
    parser.add_argument(
        "--torch-version", help="Specific torch version (default: latest)"
    )
    parser.add_argument(
        "--torch-version-prefix",
        help="Torch version prefix for discovery (e.g. '2.9' matches 2.9.x). "
             "Only used in auto-discovery mode (--rocm-version without --torch-version).",
    )
    parser.add_argument(
        "--torchaudio-version", help="Specific torchaudio version (default: latest)"
    )
    parser.add_argument(
        "--torchvision-version", help="Specific torchvision version (default: latest)"
    )
    parser.add_argument(
        "--triton-version",
        help="Specific triton version (default: from torch dependency)",
    )
    parser.add_argument(
        "--no-break-system-packages",
        action="store_true",
        help="Don't use --break-system-packages",
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip verification step"
    )

    args = parser.parse_args()

    # Build the full index URLs. TheRock full-suite can use separate indexes
    # for the torch wheel and ROCm packages, so mirror that shape here.
    index_url = f"{args.index_url.rstrip('/')}/{args.amdgpu_family}/"
    rocm_index_url = (
        f"{args.rocm_package_index_url.rstrip('/')}/{args.amdgpu_family}/"
        if args.rocm_package_index_url
        else index_url
    )
    extra_index_url = rocm_index_url if rocm_index_url != index_url else None

    rocm = args.rocm_version
    rocm_only = bool(rocm and not args.torch_version)
    torch_prefix = args.torch_version_prefix if rocm_only else None
    break_sys = not args.no_break_system_packages

    if rocm_only:
        # Three-pass install:
        #   Pass 1: rocm[devel] (pinned) from the ROCm package index
        #   Pass 2: torch (pinned) from the torch wheel index
        #   Pass 3: torchaudio, torchvision, triton (unpinned; pip resolves compatibility)
        torch_version = get_latest_package_version_for_rocm(
            index_url, "torch", rocm, required=True, version_prefix=torch_prefix,
        )

        print_banner("PyTorch Wheels Installation")
        print(f"Index URL:      {index_url}")
        print(f"ROCm Index URL: {rocm_index_url}")
        print(f"AMDGPU Family:  {args.amdgpu_family}")
        print(f"Python:         {sys.version_info.major}.{sys.version_info.minor}")
        print(f"torch:          {torch_version}")
        print(f"rocm[devel]:    {rocm}")
        print(f"torchaudio:     (pip resolves)")
        print(f"torchvision:    (pip resolves)")
        print(f"triton:         (torch dependency)")
        print("=" * 50)

        print_banner("Pass 1: rocm[devel]")
        rocm_package = build_package_spec("rocm[devel]", rocm)
        print(f"Installing: {rocm_package}")
        run_pip_install(rocm_index_url, [rocm_package], break_sys)

        torch_package = build_package_spec("torch", torch_version)
        print_banner("Pass 2: torch")
        print(f"Installing: {torch_package}")
        run_pip_install(index_url, [torch_package], break_sys, extra_index_url)

        # Install torchaudio/torchvision without pinning; pip picks versions
        # compatible with the torch that's already installed.
        companions = ["torchaudio", "torchvision"]
        print_banner("Pass 3: torchaudio, torchvision (unpinned)")
        print(f"Installing: {', '.join(companions)}")
        run_pip_install(index_url, companions, break_sys, extra_index_url)
    else:
        # Explicit versions mode — install everything in one shot
        arg_attrs = ["torch_version", "torchaudio_version", "torchvision_version", "triton_version"]
        versions = {p: getattr(args, a) for p, a in zip(PYTORCH_PKGS, arg_attrs)}
        versions["rocm[devel]"] = rocm if rocm else None

        print_banner("PyTorch Wheels Installation")
        print(f"Index URL:      {index_url}")
        print(f"ROCm Index URL: {rocm_index_url}")
        print(f"AMDGPU Family:  {args.amdgpu_family}")
        print(f"Python:         {sys.version_info.major}.{sys.version_info.minor}")
        for name, version in versions.items():
            print(f"{name:14}: {version or 'latest'}")
        print("=" * 50)

        packages = []
        for name, always_install in PACKAGES.items():
            version = versions.get(name)
            if always_install or version:
                packages.append(build_package_spec(name, version))

        rocm_package = build_package_spec("rocm[devel]", rocm) if rocm else None
        if rocm_package:
            print_banner("Pass 1: rocm[devel]")
            print(f"Installing: {rocm_package}")
            run_pip_install(rocm_index_url, [rocm_package], break_sys)
            packages = [p for p in packages if not p.startswith("rocm[devel]")]

        print_banner("Pass 2: PyTorch packages")
        print(f"Installing: {', '.join(packages)}")
        run_pip_install(index_url, packages, break_sys, extra_index_url)

    # Verify
    if not args.skip_verify and not verify_installation():
        return 1

    list_installed_packages()
    print_banner("Installation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
