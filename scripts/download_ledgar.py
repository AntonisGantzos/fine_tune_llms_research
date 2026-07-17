#!/usr/bin/env python3
"""
Download the LexGLUE LEDGAR subset (Task 3) to a local folder.

Mirrors scripts/download_cuad.py, but pulls the research-standard version of
LEDGAR from LexGLUE (`coastalcph/lex_glue`, config `ledgar`): ~80k provisions,
the 100 most frequent labels, single-label, with a ready-made
train/validation/test split. See docs/task_3/TASK3_PLAN.md, Step 1.

Each example is {text, label} where `label` is an int 0-99. The dataset's
`ClassLabel` metadata maps those ids to human-readable names; we resolve them
once and write a `label_name` column plus a `labels.json` id->name mapping so
everything downstream reads plain files (same idea as CUAD's CSV).

Usage:
  python scripts/download_ledgar.py
  python scripts/download_ledgar.py --out data/LEDGAR --force
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import dotenv

dotenv.load_dotenv()

# LEDGAR gets its own folder, sibling to CUAD_v1. DATA_DIR in this repo points
# at the CUAD root in some contexts, so default to a plain `data/LEDGAR` unless
# overridden by --out or the LEDGAR_DIR env var.
DEFAULT_OUT = os.getenv("LEDGAR_DIR", "data/LEDGAR")

SPLITS = ("train", "validation", "test")


def main(out_dir: Path, force: bool = False) -> None:
    if out_dir.exists():
        if not force:
            print(f"Dataset already exists at {out_dir}. Skipping download.")
            return
        print(f"--force: removing existing dataset at {out_dir} before download...")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so `--help` works without the heavy dependency installed.
    from datasets import load_dataset

    print("Loading coastalcph/lex_glue (config 'ledgar') from Hugging Face...")
    ds = load_dataset("coastalcph/lex_glue", "ledgar")

    # Resolve the id -> name mapping once from the ClassLabel feature.
    label_feature = ds["train"].features["label"]
    label_names = list(label_feature.names)
    print(f"{len(label_names)} labels; {len(ds['train'])} train / "
          f"{len(ds['validation'])} validation / {len(ds['test'])} test examples.")

    labels_path = out_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump({i: name for i, name in enumerate(label_names)}, fh,
                  ensure_ascii=False, indent=2)
    print(f"Wrote label map -> {labels_path}")

    for split in SPLITS:
        split_ds = ds[split]
        # Add the human-readable name so downstream never re-resolves ids.
        split_ds = split_ds.map(
            lambda ex: {"label_name": label_names[ex["label"]]},
            desc=f"resolving {split} label names",
        )
        csv_path = out_dir / f"ledgar_{split}.csv"
        split_ds.to_csv(str(csv_path), index=False)
        print(f"Wrote {len(split_ds):>6} rows -> {csv_path}")

    print(f"Done. LEDGAR saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the LexGLUE LEDGAR subset to a local folder")
    parser.add_argument("--out", "-o", default=DEFAULT_OUT,
                        help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--force", action="store_true",
                        help="Remove existing target folder and re-download")
    args = parser.parse_args()
    try:
        main(out_dir=Path(args.out), force=args.force)
    except Exception as e:
        print("Error during download:", e, file=sys.stderr)
        raise
