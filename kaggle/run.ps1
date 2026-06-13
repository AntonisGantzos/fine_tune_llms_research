# Part 4.3 — push the kernel, optionally poll to completion and pull the adapter back.
#   .\kaggle\run.ps1            # push only
#   .\kaggle\run.ps1 -Wait      # push, poll status, then download outputs when complete
param([switch]$Wait)
$ErrorActionPreference = "Stop"

# Invoke the CLI as a Python module — the kaggle.exe shim is blocked by Windows
# Application Control on this machine, but `python -m kaggle` runs fine.
$kaggle = @($(if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }), "-m", "kaggle")

# Derive the kernel id from kernel-metadata.json so the username lives in exactly one place.
$meta     = Get-Content (Join-Path $PSScriptRoot "kernel-metadata.json") -Raw | ConvertFrom-Json
$kernelId = $meta.id
if ($kernelId -like "antonisgantzos/*") {
    throw "Edit kaggle\kernel-metadata.json: replace antonisgantzos with your Kaggle username."
}

Write-Host "Pushing kernel $kernelId ..."
& $kaggle[0] $kaggle[1..2] kernels push -p $PSScriptRoot

if ($Wait) {
    do {
        Start-Sleep 30
        $s = & $kaggle[0] $kaggle[1..2] kernels status $kernelId
        $s
    } while ($s -notmatch "complete|error|KernelWorkerError")

    $out = Join-Path $PSScriptRoot "..\kaggle_output"
    New-Item -ItemType Directory -Force $out | Out-Null
    & $kaggle[0] $kaggle[1..2] kernels output $kernelId -p $out
    Write-Host "Outputs downloaded to $out"
}
