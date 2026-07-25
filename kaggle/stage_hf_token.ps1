# Stage the HF token into a PRIVATE Kaggle dataset so the Task 3 kernel can read it
# from a mounted FILE — the fully headless alternative to the Add-ons -> Secrets web step.
#
# WHY: kernel-metadata.json can declare datasets but NOT secrets, and the Kaggle API
# cannot attach a secret. So people add the HF_TOKEN secret in the web editor and click
# "Save Version" — which commits the editor's inputs and SILENTLY empties dataset_sources,
# detaching the data. Two surfaces clobber each other. Putting the token in a private
# dataset makes it just another dataset_source: a plain `kernels push` carries BOTH the
# LEDGAR data and the token, and the web UI is never touched, so nothing gets clobbered.
#
# The token is read from $env:HF_TOKEN (or .env at the repo root) and written to
# hf_token.txt, which is git-ignored and MUST NOT be committed.
#
# First time:  .\kaggle\stage_hf_token.ps1
#              python -m kaggle datasets create  -p kaggle\dataset_payload_hf_token
# Rotate:      .\kaggle\stage_hf_token.ps1
#              python -m kaggle datasets version -p kaggle\dataset_payload_hf_token -m "rotate token"
$ErrorActionPreference = "Stop"
$payload = Join-Path $PSScriptRoot "dataset_payload_hf_token"

# Resolve the token: prefer the shell env var, else parse .env at the repo root.
$token = $env:HF_TOKEN
if (-not $token) {
    $envFile = Join-Path $PSScriptRoot "..\.env"
    if (Test-Path $envFile) {
        $line = Get-Content $envFile | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1
        if ($line) { $token = ($line -replace '^\s*HF_TOKEN\s*=\s*', '').Trim().Trim('"').Trim("'") }
    }
}
if (-not $token) {
    throw "HF_TOKEN not found. Set it first:  `$env:HF_TOKEN = 'hf_...'  (or add it to .env)"
}

New-Item -ItemType Directory -Force $payload | Out-Null
# Plain UTF-8, no BOM/newline; the notebook .strip()s it anyway.
[System.IO.File]::WriteAllText((Join-Path $payload "hf_token.txt"), $token.Trim())
Write-Host "Wrote hf_token.txt into $payload  (token length = $($token.Trim().Length))"
Get-ChildItem $payload -File | Format-Table Name, Length
