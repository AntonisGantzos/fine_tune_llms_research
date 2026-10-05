# Part 4.3 — push the kernel, optionally poll to completion and pull the adapter back.
#   .\kaggle\run.ps1            # push only
#   .\kaggle\run.ps1 -Wait      # push, poll status, then download outputs when complete
#   .\kaggle\run.ps1 -KernelDir kaggle\extension -OutDir kaggle_output_extension -Wait
#                               # the legal-model extension runner (its own kernel folder)
param(
    [switch]$Wait,
    [string]$KernelDir = $PSScriptRoot,   # folder holding kernel-metadata.json
    [string]$OutDir = "kaggle_output"     # download target, relative to the repo root
)
$ErrorActionPreference = "Stop"
$KernelDir = (Resolve-Path $KernelDir).Path
$repoRoot  = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Invoke the CLI as a Python module — the kaggle.exe shim is blocked by Windows
# Application Control on this machine, but `python -m kaggle` runs fine.
$kaggle = @($(if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }), "-m", "kaggle")

# Derive the kernel id from kernel-metadata.json so the username lives in exactly one place.
$meta     = Get-Content (Join-Path $KernelDir "kernel-metadata.json") -Raw | ConvertFrom-Json
$kernelId = $meta.id
if ($kernelId -like "YOUR_KAGGLE_USERNAME/*") {
    throw "Edit kaggle\kernel-metadata.json: replace YOUR_KAGGLE_USERNAME with your Kaggle username."
}

Write-Host "Pushing kernel $kernelId ..."
& $kaggle[0] $kaggle[1..2] kernels push -p $KernelDir

# --- Dataset-source guard ---------------------------------------------------
# The Kaggle CLI (2.2.1) SILENTLY DROPS a dataset_source that does not yet exist
# at push time instead of failing: the kernel then runs with NOTHING mounted and
# dies on the first `/kaggle/input/...` read — after wasting the whole GPU run.
# So if the LOCAL metadata declares any dataset_sources, pull the pushed kernel's
# metadata back and confirm every one of them actually attached; abort loudly if not.
$localSources = @($meta.dataset_sources)
if ($localSources.Count -gt 0) {
    $verifyDir = Join-Path $env:TEMP ("kverify_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force $verifyDir | Out-Null
    try {
        & $kaggle[0] $kaggle[1..2] kernels pull $kernelId -p $verifyDir -m | Out-Null
        $pushed        = Get-Content (Join-Path $verifyDir "kernel-metadata.json") -Raw | ConvertFrom-Json
        $pushedSources = @($pushed.dataset_sources)
        $missing       = $localSources | Where-Object { $pushedSources -notcontains $_ }
        if ($missing) {
            throw ("Dataset source(s) were DROPPED on push (not attached to the kernel): " +
                   ($missing -join ", ") + ". The Kaggle dataset must exist and be 'ready' BEFORE pushing. " +
                   "Upload it (see stage_data.ps1), confirm with " +
                   "``python -m kaggle datasets status <slug>``, then re-run this script.")
        }
        Write-Host ("Dataset source(s) attached OK: " + ($pushedSources -join ", "))
    }
    finally {
        Remove-Item $verifyDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
# ---------------------------------------------------------------------------

if ($Wait) {
    do {
        Start-Sleep 30
        $s = & $kaggle[0] $kaggle[1..2] kernels status $kernelId
        $s
    } while ($s -notmatch "complete|error|KernelWorkerError")

    $out = Join-Path $repoRoot $OutDir
    New-Item -ItemType Directory -Force $out | Out-Null
    & $kaggle[0] $kaggle[1..2] kernels output $kernelId -p $out
    Write-Host "Outputs downloaded to $out"
}
