# Kaggle scripts walkthrough — `stage_data.ps1` and `run.ps1`

This document explains, step by step, how to run the two PowerShell scripts under
[`kaggle/`](../../kaggle/) and exactly what each line does. Together they let you run a
notebook on Kaggle's free GPU without leaving your machine.

- [`kaggle/stage_data.ps1`](../../kaggle/stage_data.ps1) — copies the cleaned CSVs into an
  upload payload so they can be published as a Kaggle **Dataset**.
- [`kaggle/run.ps1`](../../kaggle/run.ps1) — pushes the notebook as a Kaggle **Kernel**,
  optionally waits for it to finish, and downloads the results.

They rely on two metadata files and one payload folder:

| File | Role |
|---|---|
| [`kaggle/dataset_payload/dataset-metadata.json`](../../kaggle/dataset_payload/dataset-metadata.json) | Identifies the Dataset (`id`, `title`, license) that `stage_data` fills and you upload. |
| [`kaggle/dataset_payload/CUAD_v1/`](../../kaggle/dataset_payload/) | The staged CSVs — created/refreshed by `stage_data.ps1`. |
| [`kaggle/kernel-metadata.json`](../../kaggle/kernel-metadata.json) | Identifies the Kernel (`id`, which notebook `code_file` runs, GPU, attached datasets) that `run.ps1` pushes. |

---

## Prerequisites (one-time)

1. **Activate the venv** — the Kaggle CLI lives there:
   ```powershell
   .\env\Scripts\Activate.ps1
   ```

2. **Authenticate.** On this machine the `kaggle.exe` shim is blocked by Windows
   Application Control, so the CLI is always invoked as **`python -m kaggle`**. Save your
   API token:
   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
   Set-Content "$env:USERPROFILE\.kaggle\access_token" "KGAT_your_new_token" -NoNewline -Encoding ascii
   python -m kaggle kernels list --mine        # auth check — should list your kernels
   ```

3. **Replace the username placeholder** (only if the metadata still says
   `YOUR_KAGGLE_USERNAME`). In this repo the `id`s are already set to
   `antonisgantzos/…`, so this is only relevant on a fresh clone.

4. **Hugging Face access.** The notebook downloads a gated Llama model. On
   kaggle.com → your notebook → **Add-ons → Secrets**, add a secret named exactly
   `HF_TOKEN`, and accept the model license on huggingface.co with the same account.

---

## `stage_data.ps1` — build the dataset payload

### What it's for
The notebook reads `master_clauses_cleaned.csv` (and the sampled variant). Kaggle can't
read your local disk, so those CSVs must be published as a Kaggle **Dataset** and mounted
into the kernel. This script copies **only the ~5 MB CSVs** into an upload folder — never
the multi-GB PDFs/txt — so the upload stays tiny.

### Line-by-line breakdown
```powershell
$ErrorActionPreference = "Stop"                              # (1) abort on the first error
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")      # (2) repo root = one level up from kaggle/
$payload  = Join-Path $PSScriptRoot "dataset_payload\CUAD_v1" # (3) destination folder for the CSVs

New-Item -ItemType Directory -Force $payload | Out-Null       # (4) create the payload dir (no-op if it exists)
Copy-Item ...\master_clauses_cleaned.csv          $payload -Force   # (5) copy the full cleaned CSV
Copy-Item ...\master_clauses_cleaned_sampled.csv  $payload -Force   # (6) copy the sampled CSV

Write-Host "Staged CSVs into $payload"                       # (7) confirm where files landed
Get-ChildItem $payload | Format-Table Name, Length           # (8) list them with sizes as a sanity check
```

1. `$ErrorActionPreference = "Stop"` — if any copy fails, the script stops instead of
   silently uploading a half-built payload.
2. `$PSScriptRoot` is the folder the script lives in (`kaggle/`); `..` resolves to the repo
   root, so the script works **run from anywhere**.
3. Builds the destination path `kaggle/dataset_payload/CUAD_v1/`.
4. Ensures that folder exists (`-Force` makes it idempotent — no error if already present).
5–6. Copies the two CSVs the pipeline actually reads. `-Force` overwrites older copies.
7–8. Prints the destination and a table of `Name, Length` so you can confirm both files
   are present and non-empty before uploading.

### How to run
```powershell
.\kaggle\stage_data.ps1
```
Expected output: a two-row table listing `master_clauses_cleaned.csv` and
`master_clauses_cleaned_sampled.csv` with their byte sizes.

### Then publish the payload to Kaggle
`stage_data.ps1` only *stages* files locally — it does **not** upload. Follow it with the
Kaggle CLI:

- **First time (creates the Dataset):**
  ```powershell
  python -m kaggle datasets create -p kaggle\dataset_payload --dir-mode zip
  ```
- **Every later refresh (new version):**
  ```powershell
  .\kaggle\stage_data.ps1
  python -m kaggle datasets version -p kaggle\dataset_payload -m "update csvs" --dir-mode zip
  ```

The `id` in `dataset_payload/dataset-metadata.json`
(`antonisgantzos/cuad-master-clauses-cleaned`) must match the `dataset_sources` entry in
`kernel-metadata.json`, so the kernel mounts the dataset you just uploaded.

---

## `run.ps1` — push the kernel and (optionally) fetch results

### What it's for
Uploads the notebook to Kaggle as a batch kernel and runs it on a T4 GPU. With `-Wait` it
also polls until the run finishes and downloads the outputs.

### Signature
```powershell
param([switch]$Wait)
```
- **No flag** → push only (fire-and-forget; watch progress on kaggle.com).
- **`-Wait`** → push, poll status every 30 s, then download outputs when complete.

### Line-by-line breakdown
```powershell
$ErrorActionPreference = "Stop"                              # (1) abort on first error

$kaggle = @($(if ($env:VIRTUAL_ENV) {                        # (2) build the CLI invocation:
    Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"          #     venv python if a venv is active,
  } else { "python" }), "-m", "kaggle")                      #     else system python — always `-m kaggle`

$meta     = Get-Content (Join-Path $PSScriptRoot "kernel-metadata.json") -Raw | ConvertFrom-Json  # (3) read metadata
$kernelId = $meta.id                                          # (4) kernel id lives in exactly one place
if ($kernelId -like "YOUR_KAGGLE_USERNAME/*") { throw ... }  # (5) guard: fail if placeholder not replaced

Write-Host "Pushing kernel $kernelId ..."
& $kaggle[0] $kaggle[1..2] kernels push -p $PSScriptRoot     # (6) upload everything in kaggle/ as the kernel

if ($Wait) {
    do {
        Start-Sleep 30                                        # (7) poll every 30 seconds
        $s = & $kaggle[0] $kaggle[1..2] kernels status $kernelId
        $s
    } while ($s -notmatch "complete|error|KernelWorkerError")# (8) stop when finished or errored

    $out = Join-Path $PSScriptRoot "..\kaggle_output"         # (9) local download folder
    New-Item -ItemType Directory -Force $out | Out-Null
    & $kaggle[0] $kaggle[1..2] kernels output $kernelId -p $out  # (10) pull the kernel's /kaggle/working files
    Write-Host "Outputs downloaded to $out"
}
```

1. Stop on first error — a failed push shouldn't fall through to polling.
2. Chooses the interpreter: the venv's `python.exe` if a venv is active, otherwise plain
   `python`. Always calls the CLI as `python -m kaggle` (the blocked-`.exe` workaround).
   `$kaggle` becomes an array like `python.exe -m kaggle`, spread into later calls.
3–4. Reads `kernel-metadata.json` and pulls out `id`. Keeping the id in the metadata means
   the script never hard-codes your username.
5. Safety guard: if you cloned fresh and forgot to replace `YOUR_KAGGLE_USERNAME`, it
   throws a clear message instead of pushing to an invalid id.
6. `kernels push -p $PSScriptRoot` uploads the **entire `kaggle/` folder** as the kernel.
   Kaggle reads `kernel-metadata.json` there to know which notebook (`code_file`), which
   GPU, and which datasets to attach.
7–8. When `-Wait` is set, poll `kernels status` every 30 s and print each status line. The
   loop exits once the status text matches `complete`, `error`, or `KernelWorkerError`.
9–10. Create `..\kaggle_output` and run `kernels output`, which downloads **all files the
   kernel wrote to `/kaggle/working`** into that folder.

> **Which notebook runs?** Whatever `code_file` points to in `kernel-metadata.json`
> (currently `../llm_fine_tuning_LORA_task1_v2.ipynb`). To run a different notebook, change
> `code_file` — and if you don't want to overwrite the existing kernel and its stored
> outputs, also change `id` to a new slug. See the note below.

### How to run
```powershell
# Push only — then watch on kaggle.com:
.\kaggle\run.ps1

# Push, wait for completion, and download results into ..\kaggle_output:
.\kaggle\run.ps1 -Wait
```

### What lands locally
`kernels output` downloads everything under the kernel's `/kaggle/working` into
`kaggle_output/`. For the fine-tuning notebook that's the adapter
(`adapter_model.safetensors`, `adapter_config.json`), `eval_metrics.json`,
`eval_report.txt`, `train_metrics.json`, and the generated `cuad/` JSONL.

> **Note — output overwrite behavior.** `kernels output` overwrites files of the *same
> name* and does not delete others (it never prunes). Running the no-fine-tune baseline
> notebook, which writes everything under a `no_finetune_baseline/` subfolder, therefore
> won't overwrite the fine-tuned artifacts. But pushing a different notebook under the
> **same kernel `id`** replaces the kernel *on Kaggle* — use a new `id` for a separate run.

---

## End-to-end sequence (first run)

```powershell
.\env\Scripts\Activate.ps1                                          # 0. venv

# 1. Stage + publish the CSV dataset
.\kaggle\stage_data.ps1
python -m kaggle datasets create -p kaggle\dataset_payload --dir-mode zip

# 2. Push the kernel (creates it)
.\kaggle\run.ps1

# 3. On kaggle.com: Add-ons -> Secrets -> add HF_TOKEN, enable it; accept model license.

# 4. Run and fetch results
.\kaggle\run.ps1 -Wait
```

Subsequent runs are just steps 1 (only if the CSVs changed, using `datasets version`) and 4.

---

## Quick reference

| I want to… | Command |
|---|---|
| Copy the CSVs into the upload folder | `.\kaggle\stage_data.ps1` |
| Publish the dataset (first time) | `python -m kaggle datasets create -p kaggle\dataset_payload --dir-mode zip` |
| Update the dataset (new version) | `python -m kaggle datasets version -p kaggle\dataset_payload -m "msg" --dir-mode zip` |
| Push the kernel only | `.\kaggle\run.ps1` |
| Push, wait, download results | `.\kaggle\run.ps1 -Wait` |
| Check status manually | `python -m kaggle kernels status antonisgantzos/cuad-task1-finetune` |
| Download outputs manually | `python -m kaggle kernels output <id> -p .\kaggle_output` |
