"""Optional: download the full dataset into data/full/."""
from pathlib import Path

FULL_DIR = Path(__file__).resolve().parent / "full"


def main():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: download full dataset into FULL_DIR
    print(f"Download full data into {FULL_DIR}")


if __name__ == "__main__":
    main()
