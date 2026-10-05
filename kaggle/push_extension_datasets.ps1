# Stage and upload the three private Kaggle datasets used by legal_model_extension_runner.ipynb.
#
#   .\kaggle\push_extension_datasets.ps1 -StageOnly           # copy + verify locally, upload nothing
#   .\kaggle\push_extension_datasets.ps1 -Message "harness v2" # stage, then create-or-version all three
#   .\kaggle\push_extension_datasets.ps1 -Only code -Message "fix"   # just one of: code, data, adapters
#
# Payloads (staged FLAT - the CLI skips or flattens subfolders, see kaggle/README.md):
#   dataset_payload_ext_code     -> legal-extension-code : scripts/legal_model_extension.py,
#                                   scripts/task*_metrics.py, requirements-kaggle.txt
#   dataset_payload_ext_data     -> legal-extension-data : the six canonical JSONL + LEDGAR labels.json
#   dataset_payload_ext_adapters -> llama-adapters       : taskN__adapter_config.json and
#                                   taskN__adapter_model.safetensors per Llama adapter (no checkpoints);
#                                   the runner rebuilds each adapter folder from the prefix.
#
# The JSONL are copied from the kaggle_output_task*_fine_tuned/ copies (LF line endings, exactly what
# the Llama runs trained on) and every file is checked against the SHA-256 table in
# docs/extension/DESIGN.md before anything is uploaded.
#
# New datasets are created PRIVATE (the CLI default). Always invoked as `python -m kaggle`
# (the kaggle.exe shim is blocked on this machine), from kaggle\ with slash-free -p paths.
param(
    [switch]$StageOnly,
    [string]$Message = "update",
    [ValidateSet("all", "code", "data", "adapters")][string]$Only = "all"
)
$ErrorActionPreference = "Stop"
$repo   = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }

function Reset-Payload($dir) {
    # Remove previously staged files but keep the tracked dataset-metadata.json.
    Get-ChildItem $dir -File | Where-Object { $_.Name -ne "dataset-metadata.json" } | Remove-Item -Force
}

# ---- code
$codeDir = Join-Path $PSScriptRoot "dataset_payload_ext_code"
if ($Only -in "all", "code") {
    Reset-Payload $codeDir
    Copy-Item (Join-Path $repo "scripts\legal_model_extension.py") $codeDir
    Copy-Item (Join-Path $repo "scripts\task*_metrics.py") $codeDir
    Copy-Item (Join-Path $repo "requirements-kaggle.txt") $codeDir
}

# ---- data (expected hashes = docs/extension/DESIGN.md, "Canonical data")
$dataDir = Join-Path $PSScriptRoot "dataset_payload_ext_data"
$data = @(
    @{ src = "kaggle_output_task1_fine_tuned\cuad\train\cuad_train.jsonl";                  sha = "5385dc55281636077f2b3ae0198487598a6b948a7c2efc3542720e9fd4ed06b1" },
    @{ src = "kaggle_output_task1_fine_tuned\cuad\validation\cuad_validation.jsonl";        sha = "97294400c4fa6a2e6997803397aba0dbebc6260b93127f006c1f473f989efbd0" },
    @{ src = "kaggle_output_task2_fine_tuned\cuad\train\cuad_task2_train.jsonl";            sha = "88f1092397015e47925ef35947689e4700471c59554e6d14bed1063c0b7390d2" },
    @{ src = "kaggle_output_task2_fine_tuned\cuad\validation\cuad_task2_validation.jsonl";  sha = "0a9c203065c15c6352f56cae430313ec561201d3e939da97877a3a6aa079c775" },
    @{ src = "kaggle_output_task3_fine_tuned\ledgar\train\ledgar_task3_train.jsonl";        sha = "9fc6b58fdb898fc697bd094c9badd99182544eac081f8303e9837a17679f1190" },
    @{ src = "kaggle_output_task3_fine_tuned\ledgar\validation\ledgar_task3_validation.jsonl"; sha = "dbfb450a695dccb82a2eba871afb15bad05b360de898e4b4564482d6121af4d1" }
)
if ($Only -in "all", "data") {
    Reset-Payload $dataDir
    foreach ($f in $data) {
        $src = Join-Path $repo $f.src
        $sha = (Get-FileHash $src -Algorithm SHA256).Hash.ToLower()
        if ($sha -ne $f.sha) {
            throw "$($f.src): SHA-256 $sha does not match DESIGN.md ($($f.sha)). Refusing to stage."
        }
        Copy-Item $src $dataDir
    }
    Copy-Item (Join-Path $repo "data\LEDGAR\labels.json") $dataDir
}

# ---- adapters
$adapterDir = Join-Path $PSScriptRoot "dataset_payload_ext_adapters"
$adapters = @(
    @{ task = 1; dir = "kaggle_output_task1_fine_tuned\llama-3.1-8B-cuad-task1" },
    @{ task = 2; dir = "kaggle_output_task2_fine_tuned\llama-3.1-8B-cuad-task2" },
    @{ task = 3; dir = "kaggle_output_task3_fine_tuned\llama-3.1-8B-ledgar-task3" }
)
if ($Only -in "all", "adapters") {
    Reset-Payload $adapterDir
    foreach ($a in $adapters) {
        foreach ($name in "adapter_config.json", "adapter_model.safetensors") {
            Copy-Item (Join-Path $repo (Join-Path $a.dir $name)) (Join-Path $adapterDir ("task{0}__{1}" -f $a.task, $name))
        }
    }
}

$targets = @(
    @{ key = "code";     dir = "dataset_payload_ext_code" },
    @{ key = "data";     dir = "dataset_payload_ext_data" },
    @{ key = "adapters"; dir = "dataset_payload_ext_adapters" }
) | Where-Object { $Only -in "all", $_.key }

foreach ($t in $targets) {
    Write-Host "`n== $($t.dir)"
    Get-ChildItem (Join-Path $PSScriptRoot $t.dir) -File | Format-Table Name, Length -AutoSize | Out-String | Write-Host
}
if ($StageOnly) { Write-Host "Staged only - nothing uploaded."; return }

Push-Location $PSScriptRoot
try {
    foreach ($t in $targets) {
        $id = (Get-Content (Join-Path $t.dir "dataset-metadata.json") -Raw | ConvertFrom-Json).id
        # `datasets status` fails for a dataset that does not exist yet -> create it instead of versioning.
        $ErrorActionPreference = "Continue"
        & $python -m kaggle datasets status $id *> $null
        $exists = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = "Stop"
        if ($exists) {
            Write-Host "Versioning $id ..."
            & $python -m kaggle datasets version -p $t.dir -m $Message
        } else {
            Write-Host "Creating PRIVATE dataset $id ..."
            & $python -m kaggle datasets create -p $t.dir
        }
        if ($LASTEXITCODE -ne 0) { throw "Upload of $id failed (exit $LASTEXITCODE)." }
    }
} finally { Pop-Location }
Write-Host "`nDone. Wait until each shows 'ready' before pushing the kernel:"
foreach ($t in $targets) {
    $id = (Get-Content (Join-Path $PSScriptRoot (Join-Path $t.dir "dataset-metadata.json")) -Raw | ConvertFrom-Json).id
    Write-Host "  python -m kaggle datasets status $id"
}
