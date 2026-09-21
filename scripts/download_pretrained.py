"""Download the frozen SwinIR teacher's pretrained weights.

Weights are never committed to the repo (CLAUDE.md 11: "Never commit data,
checkpoints or large binaries"). This script fetches the official
Apache-2.0-licensed classical SR x2 checkpoint from the SwinIR GitHub
release and caches it locally at .weights_cache/ (gitignored).

Usage: python scripts/download_pretrained.py
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".weights_cache"

# 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth: SwinIR-M classical SR, trained
# on DF2K (DIV2K + Flickr2K), x2. Apache-2.0, see THIRD_PARTY.md.
SWINIR_X2_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
)
SWINIR_X2_SHA256 = "2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac"
SWINIR_X2_FILENAME = "swinir_classicalSR_x2.pth"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_swinir_x2(force: bool = False) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / SWINIR_X2_FILENAME
    if out_path.exists() and not force:
        if _sha256(out_path) == SWINIR_X2_SHA256:
            print(f"already cached and verified: {out_path}")
            return out_path
        print("cached file hash mismatch, re-downloading")

    print(f"downloading {SWINIR_X2_URL} -> {out_path}")
    urllib.request.urlretrieve(SWINIR_X2_URL, out_path)

    actual = _sha256(out_path)
    if actual != SWINIR_X2_SHA256:
        out_path.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {SWINIR_X2_FILENAME}: expected {SWINIR_X2_SHA256}, got {actual}"
        )
    print("download verified")
    return out_path


if __name__ == "__main__":
    download_swinir_x2()
