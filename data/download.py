"""Download and extract the EMBER 2018 (v2) feature files into data/raw/ember2018/.

    python data/download.py

The archive (~1.6 GB) holds only pre-extracted JSON features, never executables.
Extracted size is ~9 GB. Safe to re-run: skips steps that are already done.
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RAW_ROOT = config.RAW_DIR.parent                     # data/raw/
ARCHIVE = RAW_ROOT / "ember_dataset_2018_2.tar.bz2"


def _progress(blocks, block_size, total):
    done = blocks * block_size
    if total > 0 and blocks % 2000 == 0:
        print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)


def main():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {config.EMBER_URL} -> {ARCHIVE}")
        tmp = ARCHIVE.with_suffix(".part")
        urllib.request.urlretrieve(config.EMBER_URL, tmp, _progress)
        tmp.rename(ARCHIVE)
        print()
    if not any(config.RAW_DIR.glob("*.jsonl")):
        print(f"Extracting {ARCHIVE.name} -> {RAW_ROOT}")
        with tarfile.open(ARCHIVE, "r:bz2") as tar:
            tar.extractall(RAW_ROOT, filter="data")   # rejects absolute paths / '..' members
    print("Files:", *sorted(p.name for p in config.RAW_DIR.iterdir()), sep="\n  ")


if __name__ == "__main__":
    main()
