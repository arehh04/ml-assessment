"""
Download the AI governance documents dataset from Kaggle.

Prerequisites:
  1. Create a Kaggle account at kaggle.com
  2. Go to Account > API > Create New Token > download kaggle.json
  3. Place kaggle.json at:
     - Linux/Mac: ~/.kaggle/kaggle.json
     - Windows: C:\\Users\\<user>\\.kaggle\\kaggle.json
  4. kaggle package is already in requirements.txt

Usage:
    python scripts/download_data.py
"""
import os
import sys
from pathlib import Path

DATASET = "umerhaddii/ai-governance-documents-data"
OUTPUT_DIR = Path("data/documents")


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        kaggle.api.authenticate()
        print(f"Downloading {DATASET}...")
        kaggle.api.dataset_download_files(DATASET, path=str(OUTPUT_DIR), unzip=True)
        print(f"Dataset downloaded to {OUTPUT_DIR}")
        print(f"Files: {list(OUTPUT_DIR.iterdir())}")
    except Exception as e:
        print(f"Kaggle download failed: {e}", file=sys.stderr)
        print("\nAlternative — use kagglehub:", file=sys.stderr)
        print("  pip install kagglehub", file=sys.stderr)
        print("  python -c \"import kagglehub; print(kagglehub.dataset_download('umerhaddii/ai-governance-documents-data'))\"", file=sys.stderr)
        print("Then copy the files to data/documents/", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download()
