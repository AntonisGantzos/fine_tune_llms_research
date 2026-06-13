# Running the Fine-Tuning Pipeline on Kaggle's GPU (Option A — API "push kernel")

**Goal:** Run [`llm_fine_tuning_LORA_task1.ipynb`](../../llm_fine_tuning_LORA_task1.ipynb) on Kaggle's free GPU (T4×2 / P100),
driven entirely from this local machine via the Kaggle CLI — **without** opening the Kaggle web editor.

**Model of execution:** You edit code here → push it as a Kaggle *kernel* → Kaggle runs it **in batch
(non-interactive)** on its GPU → you pull the trained adapter back here. No live/interactive session, no SSH.

**Design principle:** make the notebook *environment-aware* so the **same file runs both locally and on Kaggle**.
You will not maintain two copies.

---

## Constraints to keep in mind (they shape the changes below)

| Kaggle fact | Consequence for this repo |
|---|---|
| `/kaggle/input/...` is **read-only** | The notebook currently writes `train/` and `validation/` JSONL *under* `CUAD_PATH`. On Kaggle that path is read-only → writes must go to `/kaggle/working/`. **This is the #1 change.** |
| `.env` files are not uploaded | `HF_TOKEN` must come from **Kaggle Secrets**, not `dotenv`. |
| Only `/kaggle/working/` persists as output | The saved adapter must land in `/kaggle/working/`. |
| Batch run, max ~12 h, 30 GPU-h/week | Fine for a smoke test / single training run; not for interactive debugging. |
| Internet is off by default | Must enable internet in kernel metadata (needed to download the model from HF). |
| Base image already has torch/transformers | But `trl` must be 1.x (`SFTConfig`, `completion_only_loss`) — pin it in a setup cell. |

---

## Overview of the work

1. **Repo changes** — add a `kaggle/` folder (token config, kernel metadata, push/pull scripts).
2. **Notebook changes** — 4 small, surgical edits to make it environment-aware.
3. **One-time setup** — Kaggle API token, upload data as a Kaggle Dataset, add HF token as a Secret.
4. **Run loop** — `push` → poll `status` → `output` (pull adapter back).

---

## Part 1 — One-time account / CLI setup

### 1.1 Install and authenticate the Kaggle CLI (local)

```powershell
pip install pip install kaggle
```

- Go to **kaggle.com → Account → Settings → API → "Create New Token"**. This downloads `kaggle.json`.
- Move it to `C:\Users\anton\.kaggle\kaggle.json` (the CLI looks here by default).

```powershell
# verify
kaggle --version
kaggle kernels list --mine    # should authenticate without error
```

### 1.2 Add `HF_TOKEN` as a Kaggle Secret

Kaggle Secrets are attached **per-notebook**. The cleanest one-time path:

1. Create the kernel once (Part 4 first push will create it), then on **kaggle.com → your notebook →
   "Add-ons" → "Secrets"**, add a secret named exactly `HF_TOKEN` with your Hugging Face token value.
2. Enable that secret for the notebook.

> This is the *only* unavoidable visit to the website (a ~30 s settings toggle, not editing code).
> Alternatively, you can hardcode a token in the dataset, but Secrets is the correct approach.

> **Reminder:** request access to the gated model on Hugging Face with the account the token belongs to.
> The notebook currently uses `meta-llama/Llama-3.2-1B` (the smoke-test model).

---

## Part 2 — Upload the data as a Kaggle Dataset

The training notebook only reads the **cleaned CSVs** (~5 MB), *not* the 161 MB of PDFs/txt. Upload only
what the pipeline needs.

### 2.1 Stage the files

```powershell
# from repo root
New-Item -ItemType Directory -Force kaggle\dataset_payload\CUAD_v1 | Out-Null
Copy-Item data\CUAD_v1\master_clauses_cleaned.csv          kaggle\dataset_payload\CUAD_v1\
Copy-Item data\CUAD_v1\master_clauses_cleaned_sampled.csv  kaggle\dataset_payload\CUAD_v1\
```

> **Gotcha (caused a run failure):** the `CUAD_v1/` subfolder does **not** survive the upload.
> `kaggle datasets create ... --dir-mode zip` zips the subdirectory and Kaggle extracts it
> into the dataset **root**, so the CSVs mount at
> `/kaggle/input/cuad-master-clauses-cleaned/master_clauses_cleaned_sampled.csv` — **without**
> the `CUAD_v1/` segment. Verify with `kaggle datasets files YOUR_USERNAME/cuad-master-clauses-cleaned`.
> The notebook's Kaggle `DATA_DIR` (Edit 1) is therefore set to the dataset root, no `CUAD_v1`.

### 2.2 Create the dataset metadata

`kaggle/dataset_payload/dataset-metadata.json`:

```json
{
  "title": "cuad-master-clauses-cleaned",
  "id": "YOUR_KAGGLE_USERNAME/cuad-master-clauses-cleaned",
  "licenses": [{ "name": "CC0-1.0" }]
}
```

Replace `YOUR_KAGGLE_USERNAME`. The `id` slug is how the data mounts: `/kaggle/input/cuad-master-clauses-cleaned/`.

### 2.3 Push it

```powershell
kaggle datasets create -p kaggle\dataset_payload --dir-mode zip
# later updates:  kaggle datasets version -p kaggle\dataset_payload -m "update csvs" --dir-mode zip
```

Resulting mount path on Kaggle: `/kaggle/input/cuad-master-clauses-cleaned/CUAD_v1/`.

---

## Part 3 — Notebook changes (4 surgical edits)

All edits make the notebook detect whether it runs on Kaggle and pick paths/token source accordingly.
Locally, behavior is unchanged.

### Edit 1 — Add an environment-config cell (NEW cell, right after the imports in cell index 3)

```python
# --- Environment config: run unchanged locally OR on Kaggle ---
import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()

if ON_KAGGLE:
    # Read-only mounted dataset. NOTE: --dir-mode zip flattens the CUAD_v1/ subfolder into
    # the dataset root, so the CSVs live directly under the slug (no CUAD_v1 segment).
    DATA_DIR = Path("/kaggle/input/cuad-master-clauses-cleaned")
    # Only this dir is writable AND persisted as kernel output:
    WORK_DIR = Path("/kaggle/working")
else:
    DATA_DIR = Path(os.getenv("DATA_DIR", "data")) / "CUAD_v1"
    WORK_DIR = Path(".")

print(f"ON_KAGGLE={ON_KAGGLE} | DATA_DIR={DATA_DIR} | WORK_DIR={WORK_DIR}")
```

### Edit 2 — Cell index 4: use `DATA_DIR` for reads

Replace:
```python
CUAD_PATH = Path('data/CUAD_v1')
```
with:
```python
CUAD_PATH = DATA_DIR   # set by the environment-config cell
```
The rest of cell 4 (which reads `master_clauses_cleaned_sampled.csv`) stays as-is.

### Edit 3 — Cell index 20: write outputs to `WORK_DIR` (the read-only fix)

Replace:
```python
CUAD_TRAIN_PATH = CUAD_PATH/'train'
CUAD_VALIDATION_PATH = CUAD_PATH/'validation'
```
with:
```python
# On Kaggle, CUAD_PATH is read-only — generated JSONL must go to the writable WORK_DIR.
CUAD_TRAIN_PATH = WORK_DIR/'cuad'/'train'
CUAD_VALIDATION_PATH = WORK_DIR/'cuad'/'validation'
```
The existing `.mkdir(parents=True, exist_ok=True)` lines below already handle creation.
Cells 21 and 25 reference these same variables, so they update automatically.

### Edit 4 — Cells 24 & 25: get `HF_TOKEN` from Kaggle Secrets

In **both** cells, replace this block:
```python
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
assert hf_token, "HF_TOKEN not found — check your .env file"
```
with:
```python
def get_hf_token():
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("HF_TOKEN")

hf_token = get_hf_token()
assert hf_token, "HF_TOKEN not found (Kaggle Secret or local .env)"
```

### Edit 5 (optional but recommended) — pin `trl` at the very top

Add a NEW first code cell so the Kaggle base image has the versions this notebook expects
(`SFTConfig`, `completion_only_loss`, `processing_class`):

```python
if Path("/kaggle").exists():
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "trl>=0.12", "peft>=0.13", "transformers>=4.45",
                    "bitsandbytes>=0.44", "accelerate>=1.0", "datasets"], check=True)
```
> Put this *before* the `import sys` UTF-8 cell, or merge it in. Locally it is a no-op.

### Note on the saved adapter (cell 28 — no edit strictly needed)

`new_model_name = "llama-3.2-1B-cuad-task1-smoke-test"` is a **relative** path. On Kaggle the working
directory is `/kaggle/working/`, so the adapter is saved there and becomes downloadable output —
no change required. (If you prefer to be explicit, change the save target to
`WORK_DIR / new_model_name`.)

---

## Part 4 — Package the notebook as a kernel and run it

### 4.1 Kernel metadata

`kaggle/kernel-metadata.json`:

```json
{
  "id": "YOUR_KAGGLE_USERNAME/cuad-task1-finetune",
  "title": "CUAD Task1 Finetune",
  "code_file": "../llm_fine_tuning_LORA_task1.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["YOUR_KAGGLE_USERNAME/cuad-master-clauses-cleaned"],
  "competition_sources": [],
  "kernel_sources": []
}
```

- `enable_gpu` → the whole point.
- `enable_internet` → required to pull the model from Hugging Face.
- `dataset_sources` → mounts the dataset from Part 2 at `/kaggle/input/...`.
- `code_file` is relative to this metadata file (kept in `kaggle/`, notebook one level up).

### 4.2 Push, poll, pull

```powershell
# push (first push creates the kernel; then go enable the HF_TOKEN secret once — Part 1.2)
kaggle kernels push -p kaggle

# poll until complete
kaggle kernels status YOUR_KAGGLE_USERNAME/cuad-task1-finetune

# when "complete", download outputs (the saved adapter folder + rendered notebook + logs)
kaggle kernels output YOUR_KAGGLE_USERNAME/cuad-task1-finetune -p .\kaggle_output
```

The trained adapter (`adapter_model.safetensors`, `adapter_config.json`) lands under `.\kaggle_output\`.

### 4.3 Optional helper script

`kaggle/run.ps1` to wrap the loop:

```powershell
param([switch]$Wait)
kaggle kernels push -p $PSScriptRoot
if ($Wait) {
  do {
    Start-Sleep 30
    $s = kaggle kernels status YOUR_KAGGLE_USERNAME/cuad-task1-finetune
    $s
  } while ($s -notmatch "complete|error")
  kaggle kernels output YOUR_KAGGLE_USERNAME/cuad-task1-finetune -p (Join-Path $PSScriptRoot "..\kaggle_output")
}
```

---

## Part 5 — `.gitignore` additions

```
/kaggle/dataset_payload/
/kaggle_output/
```
Keep `kaggle/kernel-metadata.json` and `kaggle/dataset_payload/dataset-metadata.json` tracked; ignore the
copied data payload and downloaded outputs.

---

## Final checklist

- [ ] `pip install kaggle`; `kaggle.json` at `C:\Users\anton\.kaggle\`
- [ ] Data uploaded as Kaggle Dataset (Part 2); note the mount path
- [ ] Notebook Edits 1–5 applied
- [ ] `kaggle/kernel-metadata.json` created with your username + GPU + internet on
- [ ] First `kaggle kernels push`, then add `HF_TOKEN` secret on the website once and enable it
- [ ] HF account has access to the gated model
- [ ] `kaggle kernels push` → `status` → `output`

## What you cannot do this way (so expectations are set)

- **No interactive/cell-by-cell execution** on Kaggle's GPU — it's batch. Debug logic locally on CPU
  (or the 1B model), push to Kaggle only to *run* the training.
- **No live VS Code attach** — that's Option B (SSH tunnel), which is fragile and gray-area.
- Each run is a fresh container: anything not under `/kaggle/working/` is gone after the run.
