# Stage the raw LEDGAR split files the Task 3 pipeline reads (~a few MB).
# The T3 notebook builds the JSONL on Kaggle from these CSVs + labels.json,
# so we stage the raw splits, not the generated JSONL.
#
# Files are staged FLAT (no subfolder) alongside dataset-metadata.json so they
# resolve directly at the mount root: the notebook reads
#   DATA_DIR / "labels.json" | "ledgar_train.csv" | "ledgar_validation.csv"
# with DATA_DIR = /kaggle/input/ledgar-lexglue (no fallback subfolder search),
# which also sidesteps the `--dir-mode zip` subfolder-flattening quirk.
#
# Run from anywhere; paths are resolved relative to this script.
#
# NOTE: staging only copies files locally. The Kaggle dataset must then be
# uploaded, or the kernel push fails with "not valid dataset sources"
# (['antonisgantzos/ledgar-lexglue']). From the kaggle\ dir, use a SLASH-FREE
# -p path — a forward slash makes the 2.2.1 CLI build a broken temp upload path:
#   First time:  python -m kaggle datasets create  -p dataset_payload_ledgar
#   Updates:     python -m kaggle datasets version -p dataset_payload_ledgar -m "msg"
$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$source   = Join-Path $repoRoot "data\LEDGAR"
$payload  = Join-Path $PSScriptRoot "dataset_payload_ledgar"

New-Item -ItemType Directory -Force $payload | Out-Null
Copy-Item (Join-Path $source "labels.json")            $payload -Force
Copy-Item (Join-Path $source "ledgar_train.csv")       $payload -Force
Copy-Item (Join-Path $source "ledgar_validation.csv")  $payload -Force
# test CSV is held out (unused for now); uncomment to include it:
# Copy-Item (Join-Path $source "ledgar_test.csv")      $payload -Force

Write-Host "Staged LEDGAR files into $payload"
Get-ChildItem $payload -File | Format-Table Name, Length
