import sys
import platform
import subprocess
import warnings
from pathlib import Path


def ensure_swig():
    """Ensure swig is available on Linux or macOS. Installs if missing."""
    system = platform.system()

    try:
        subprocess.run(
            ["swig", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("✅ swig already installed")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ swig not found, attempting to install...")

    if system == "Linux":
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "swig"], check=True)
    elif system == "Darwin":  # macOS
        subprocess.run(["brew", "install", "swig"], check=True)
    else:
        raise RuntimeError(f"❌ Unsupported OS for automatic swig install: {system}")

    print("✅ swig installed successfully")


def setup_ignore_warnings():
    """Silence noisy deprecation/user warnings from dependencies."""

    # "pkg_resources is deprecated" warning from Pygame
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame")
    warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

    # "Deprecated call to `pkg_resources.declare_namespace`" warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

    # "Jupyter is migrating its paths to use standard platformdirs" warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="jupyter_client.connect")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="jupyter_core.paths")

    # Optional: silence FutureWarnings too if they pop up often
    # warnings.filterwarnings("ignore", category=FutureWarning)

    print("🔇 Warnings filtered")


def _run_pip(args, allow_uv=True):
    """
    Run pip (or uv pip if available).
    Args should be a list of arguments, e.g. ["install", "-r", "requirements.txt"].
    If `allow_uv` is False, always use pip.
    """
    if allow_uv:
        try:
            subprocess.run(
                ["uv", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            cmd = ["uv", "pip"] + args
        except (FileNotFoundError, subprocess.CalledProcessError):
            cmd = [sys.executable, "-m", "pip"] + args
    else:
        cmd = [sys.executable, "-m", "pip"] + args

    subprocess.run(cmd, check=True)


def setup_env():
    """
    Bootstrap environment:
      1. Ensure swig is available
      2. Uninstall old gym
      3. Install requirements from local requirements.txt (repo root)
      4. Ignore warnings
    """
    print("🔧 Bootstrapping environment...")

    # Step 1: Ensure swig
    ensure_swig()

    # Step 2: Forcefully remove the old 'gym' package to avoid conflicts in Colab
    #         (pip only, since uv uninstall doesn’t support -y)
    _run_pip(["uninstall", "-y", "gym"], allow_uv=False)

    # Step 3: Install requirements from repo root
    repo_root = Path(__file__).resolve().parents[2]  # go up from src/rlcourse/
    req_file = repo_root / "requirements.txt"
    if not req_file.exists():
        raise FileNotFoundError(f"❌ requirements.txt not found at {req_file}")
    _run_pip(["install", "-q", "-r", str(req_file)])
    print("📦 Packages installed")

    # Step 4: Filter warnings
    setup_ignore_warnings()

    print("✅ Environment ready")
