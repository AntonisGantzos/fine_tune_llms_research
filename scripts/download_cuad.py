from huggingface_hub import snapshot_download
from pathlib import Path
import argparse
import shutil
import sys
import dotenv
import os

dotenv.load_dotenv()
# Your specific path
target_dir = Path(os.getenv("DATA_DIR"))
print(f"Target directory for dataset: {target_dir}")

def main(force: bool = False) -> None:
    if target_dir.exists():
        if not force:
            print(f"Dataset already exists at {target_dir}. Skipping download.")
            return
        else:
            print(f"--force: removing existing dataset at {target_dir} before download...")
            shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Download directly to the folder
    # local_dir_use_symlinks=False prevents the nested cache structure
    snapshot_download(
        repo_id="theatticusproject/cuad",
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )

    print(f"Dataset downloaded directly to {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download theatticusproject/cuad snapshot to local folder")
    parser.add_argument("--force", action="store_true", help="Remove existing target folder and re-download")
    args = parser.parse_args()
    try:
        main(force=args.force)
    except Exception as e:
        print("Error during download:", e, file=sys.stderr)
        raise