# Part 2.1 — stage ONLY the cleaned CSVs the pipeline reads (~5 MB), not the PDFs/txt.
# Run from anywhere; paths are resolved relative to this script.
$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$payload  = Join-Path $PSScriptRoot "dataset_payload\CUAD_v1"

New-Item -ItemType Directory -Force $payload | Out-Null
Copy-Item (Join-Path $repoRoot "data\CUAD_v1\master_clauses_cleaned.csv")          $payload -Force
Copy-Item (Join-Path $repoRoot "data\CUAD_v1\master_clauses_cleaned_sampled.csv")  $payload -Force

Write-Host "Staged CSVs into $payload"
Get-ChildItem $payload | Format-Table Name, Length
