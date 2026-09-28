"""Download and verify the prebuilt ONNX Runtime release for this platform
(CLAUDE.md 7.1: "prebuilt release fetched via CMake FetchContent or a
documented download script" -- this is the documented-download-script
option; CMake's own FetchContent step in engine/CMakeLists.txt calls this
script rather than re-implementing the download logic in CMake).

CPU execution provider only (default). Extracts to .onnxruntime_cache/
(gitignored) at the repo root; engine/CMakeLists.txt points its include/lib
paths there.

Usage: python scripts/fetch_onnxruntime.py
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".onnxruntime_cache"
VERSION = "1.20.1"

_RELEASES = {
    "windows": {
        "url": f"https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/"
               f"onnxruntime-win-x64-{VERSION}.zip",
        "sha256": "78d447051e48bd2e1e778bba378bec4ece11191c9e538cf7b2c4a4565e8f5581",
        "archive_name": f"onnxruntime-win-x64-{VERSION}.zip",
        "extracted_dir": f"onnxruntime-win-x64-{VERSION}",
    },
    "linux": {
        "url": f"https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/"
               f"onnxruntime-linux-x64-{VERSION}.tgz",
        "sha256": "67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105",
        "archive_name": f"onnxruntime-linux-x64-{VERSION}.tgz",
        "extracted_dir": f"onnxruntime-linux-x64-{VERSION}",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _host_platform() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    raise RuntimeError(
        f"no ONNX Runtime release configured for platform {system!r}. "
        "Add an entry to _RELEASES in this script (with a real SHA256 you "
        "computed yourself, not guessed)."
    )


def fetch(force: bool = False) -> Path:
    info = _RELEASES[_host_platform()]
    CACHE_DIR.mkdir(exist_ok=True)
    final_dir = CACHE_DIR / info["extracted_dir"]

    if final_dir.exists() and not force:
        print(f"already extracted: {final_dir}")
        return final_dir

    archive_path = CACHE_DIR / info["archive_name"]
    if not archive_path.exists() or _sha256(archive_path) != info["sha256"]:
        print(f"downloading {info['url']}")
        urllib.request.urlretrieve(info["url"], archive_path)

    actual = _sha256(archive_path)
    if actual != info["sha256"]:
        archive_path.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {info['archive_name']}: "
            f"expected {info['sha256']}, got {actual}"
        )

    print(f"extracting {archive_path}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(CACHE_DIR)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(CACHE_DIR)  # noqa: S202 -- trusted, hash-verified archive

    if not final_dir.exists():
        raise RuntimeError(f"expected {final_dir} after extraction, not found")
    print(f"ready: {final_dir}")
    return final_dir


def clean():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


if __name__ == "__main__":
    fetch()
