#!/usr/bin/env python3
"""
Sample a reproducible random subset from master_clauses_cleaned.csv.

Usage:
  python scripts/sample_master_clauses.py --input data/CUAD_v1/master_clauses_cleaned.csv \
      --output data/CUAD_v1/master_clauses_sampled.csv --seed 42 --n 20
"""
import argparse
import csv
import os
import random
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Sample rows from a CSV reproducibly")
    p.add_argument("--input", "-i", default="data/CUAD_v1/master_clauses_cleaned.csv")
    p.add_argument("--output", "-o", default="data/CUAD_v1/master_clauses_cleaned_sampled.csv")
    p.add_argument("--seed", "-s", type=int, default=42)
    p.add_argument("--n", "-n", type=int, default=100, help="number of rows to sample")
    p.add_argument("--format", "-f", choices=["csv", "json", "both"], default="csv",
                   help="output format: csv (default), json (jsonlines), or both")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    with open(args.input, newline="", encoding="utf-8") as fh:
        dict_reader = csv.DictReader(fh)
        rows = list(dict_reader)

    if not rows:
        print("Input CSV is empty or has no data rows", file=sys.stderr)
        sys.exit(1)

    # header fields (column names)
    header = dict_reader.fieldnames if hasattr(dict_reader, 'fieldnames') else None

    n = args.n
    if n <= 0:
        print("Sample size must be > 0", file=sys.stderr)
        sys.exit(2)

    if n > len(rows):
        print(f"Requested sample size {n} is larger than available rows ({len(rows)}); using {len(rows)}")
        n = len(rows)

    rnd = random.Random(args.seed)
    sampled = rnd.sample(rows, n)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write CSV if requested
    if args.format in ("csv", "both"):
        csv_out = args.output if args.output.lower().endswith('.csv') else args.output + '.csv'
        with open(csv_out, "w", newline="", encoding="utf-8") as outfh:
            writer = csv.DictWriter(outfh, fieldnames=header)
            writer.writeheader()
            writer.writerows(sampled)
        print(f"Wrote {n} sampled rows to {csv_out}")

    # Write JSON lines if requested (datasets.load_dataset with 'json' accepts jsonlines)
    if args.format in ("json", "both"):
        json_out = args.output if args.output.lower().endswith('.json') else args.output + '.json'
        import json as _json
        with open(json_out, 'w', encoding='utf-8') as jfh:
            for row in sampled:
                jfh.write(_json.dumps(row, ensure_ascii=False) + '\n')
        print(f"Wrote {n} sampled rows (jsonlines) to {json_out}")


if __name__ == "__main__":
    main()
