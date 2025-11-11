from huggingface_hub import snapshot_download
from pathlib import Path
import sys
import sysconfig

def _installed_package_dir() -> Path:
    purelib = Path(sysconfig.get_paths()["purelib"])
    return purelib / "squidly"

def _user_data_dir() -> Path:
    try:
        from appdirs import user_data_dir
        return Path(user_data_dir("squidly", ""))
    except Exception:
        return Path.home() / ".local" / "share" / "squidly"

def download_models() -> Path:
    pkg_dir = _installed_package_dir()
    models_path = pkg_dir / "models"

    # Try writing into the installed package (site-packages/squidly)
    try:
        print(f"Attempting to download HuggingFace snapshot into package dir: {pkg_dir}")
        pkg_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id="WillRieger/Squidly", local_dir=str(pkg_dir))
        print(f"Models downloaded to: {models_path}")
        return models_path
    except (PermissionError, OSError) as e:
        # Fallback to per-user data directory
        user_dir = _user_data_dir()
        print(f"Warning: cannot write to package directory ({e}). Falling back to user data directory: {user_dir}", file=sys.stderr)
        user_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id="WillRieger/Squidly", local_dir=str(user_dir))
        fallback = user_dir / "models"
        print(f"Models downloaded to: {fallback}")
        return fallback

if __name__ == "__main__":
    download_models()
