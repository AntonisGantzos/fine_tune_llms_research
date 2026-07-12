# Guide: Running a New Notebook on Kaggle's GPU with `stage_data.ps1` + `run.ps1`

This is the step-by-step recipe for taking a **new local notebook** (e.g. the Task 2
pipeline), pushing its **data** and then the **pipeline itself** to Kaggle, and running it
on the free GPU. It breaks down what each script does and what you must edit between runs.

Related docs:
- [kaggle/README.md](../../kaggle/README.md) — quick-reference version of this flow
- [pipeline_state_and_kaggle_interaction.md](pipeline_state_and_kaggle_interaction.md) — single source of truth for what currently works
- [scripts_walkthrough.md](scripts_walkthrough.md) — line-by-line script explanations

---

## The mental model

There is **no interactive remote session**. The loop is always:

```
edit notebook locally  →  push to Kaggle  →  Kaggle runs it top-to-bottom  →  download outputs
```

Two Kaggle objects are involved, each driven by its own metadata file in `kaggle/`:

| Kaggle object | Metadata file | Pushed by | Contains |
|---|---|---|---|
| **Dataset** (`cuad-master-clauses-cleaned`) | `kaggle/dataset_payload/dataset-metadata.json` | `python -m kaggle datasets create/version` | the cleaned CSVs (~5 MB) |
| **Kernel** (the notebook) | `kaggle/kernel-metadata.json` | `kaggle/run.ps1` | one notebook, run as a batch job |

The dataset mounts read-only at `/kaggle/input/cuad-master-clauses-cleaned` inside the
kernel. The notebook writes everything to `/kaggle/working`, which is what gets downloaded
back as outputs.

> **CLI rule for this machine:** always invoke the CLI as `python -m kaggle`.
> The `kaggle.exe` shim is blocked by Windows Application Control.

---

## Step 0 — One-time prerequisites (skip if already done)

1. Activate the venv:
   ```powershell
   .\env\Scripts\Activate.ps1
   ```
2. Kaggle auth — save an API token (kaggle.com → Account → Settings → API → Create New Token):
   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
   Set-Content "$env:USERPROFILE\.kaggle\access_token" "KGAT_your_token" -NoNewline -Encoding ascii
   python -m kaggle kernels list --mine    # auth check — should list your kernels
   ```
3. Hugging Face: the account behind your `HF_TOKEN` must have accepted the license for
   the gated model (`meta-llama/Meta-Llama-3.1-8B`) on huggingface.co.

---

## Step 1 — Prepare the notebook locally

1. Make sure the notebook is **environment-aware** (the existing T1/T2 notebooks already
   follow this pattern — keep it):
   - a config cell sets `ON_KAGGLE = Path("/kaggle").exists()`
   - reads go through `DATA_DIR` (→ `/kaggle/input/cuad-master-clauses-cleaned` on Kaggle)
   - writes go through `WORK_DIR` (→ `/kaggle/working` on Kaggle — the **only** writable dir)
   - the HF token comes from Kaggle Secrets (`HF_TOKEN`) when `ON_KAGGLE`, from `.env` locally
2. Remember the dataset mounts with `--dir-mode zip`, which **flattens the `CUAD_v1/`
   subfolder** — the notebook needs the CSV fallback search the existing notebooks have.
3. **Save the .ipynb file.** Local unsaved edits are invisible to the push.

---

## Step 2 — Stage and push the data (`stage_data.ps1`)

### What `stage_data.ps1` does

It copies exactly two files from `data/CUAD_v1/` into the upload payload folder
`kaggle/dataset_payload/CUAD_v1/`:

- `master_clauses_cleaned.csv` (the full cleaned dataset)
- `master_clauses_cleaned_sampled.csv` (small sample for smoke tests)

That's all — it stages the ~5 MB the pipeline actually reads, not the PDFs/txt. It does
**not** upload anything; the upload is a separate CLI call.

### Run it

```powershell
.\kaggle\stage_data.ps1
```

It prints the staged files. Then upload:

```powershell
# First time ever (dataset doesn't exist yet on Kaggle):
python -m kaggle datasets create -p kaggle\dataset_payload --dir-mode zip

# Every later time the CSVs changed (creates a new dataset version):
python -m kaggle datasets version -p kaggle\dataset_payload -m "describe the change" --dir-mode zip
```

**When do you need this step?** Only when the CSVs changed (e.g. after re-running
`scripts/preprocess_values_cuad.py`). If your new notebook reads the same cleaned CSVs
that are already uploaded — as the T2 notebook does — **skip this step entirely**; the
existing dataset version is reused.

> Note: the JSONL training files are **not** part of the dataset. Each notebook
> regenerates its JSONL from the CSV at the start of the run, on Kaggle itself.

---

## Step 3 — Point the kernel at your new notebook (`kernel-metadata.json`)

`run.ps1` always pushes whatever `kaggle/kernel-metadata.json` says. To run a **new**
notebook you edit three fields:

```json
{
  "id": "antonisgantzos/<new-kernel-slug>",          // ← NEW slug = NEW kernel on Kaggle
  "title": "<Human Readable Title>",                  // ← should match the slug
  "code_file": "../<your_new_notebook>.ipynb",        // ← which notebook gets pushed
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "dataset_sources": ["antonisgantzos/cuad-master-clauses-cleaned"],
  "competition_sources": [],
  "kernel_sources": []
}
```

Key decisions:

- **New `id` → Kaggle creates a brand-new kernel** on first push. Keeping an existing
  `id` but changing `code_file` would *overwrite* that kernel's notebook instead. For a
  new task pipeline, use a new id (e.g. `antonisgantzos/cuad-task2-finetune`) so each
  run's history stays separate.
- `code_file` is relative to the `kaggle/` folder, hence the `../` prefix (notebooks live
  at the repo root).
- `dataset_sources` stays as-is unless the notebook needs a different/new dataset.
- Keep `enable_gpu` and `enable_internet` `true` (internet is needed to pull the model
  from Hugging Face).

---

## Step 4 — First push: create the kernel (`run.ps1` without `-Wait`)

### What `run.ps1` does

1. Reads the kernel id from `kernel-metadata.json` (guards against the
   `YOUR_KAGGLE_USERNAME` placeholder).
2. Runs `python -m kaggle kernels push -p kaggle\` — uploads the notebook + metadata,
   and Kaggle **immediately queues a full top-to-bottom run**.
3. With `-Wait`: polls `kernels status` every 30 s until the run reaches
   `complete`/`error`, then downloads all outputs into `kaggle_output\` at the repo root.

### Run the first push *without* `-Wait`

```powershell
.\kaggle\run.ps1
```

Why without `-Wait`? A brand-new kernel is missing two things that can only be set in the
Kaggle web UI, so its very first run will fail at model download anyway — that's expected.

---

## Step 5 — One-time web-UI setup for the new kernel

On kaggle.com, open the new notebook (Your Work → the kernel you just created):

1. **Secret:** Add-ons → Secrets → attach a secret named exactly **`HF_TOKEN`** with your
   Hugging Face token, and toggle it **on for this notebook**. Secrets are per-notebook —
   a new kernel does not inherit the T1 kernel's secret.
2. **Accelerator:** in the notebook's settings pick **GPU T4×2**. `kernel-metadata.json`
   only carries `enable_gpu: true`; the specific T4×2 machine type is selected here. The
   notebooks mask GPU 1, which is the combination the successful runs used.

---

## Step 6 — The real run (`run.ps1 -Wait`)

```powershell
.\kaggle\run.ps1 -Wait
```

This pushes the (saved!) notebook again, polls every 30 s, and when the kernel finishes
downloads everything from `/kaggle/working` into `kaggle_output\` (git-ignored):

- the LoRA adapter folder (e.g. `llama-3.1-8B-cuad-task1/` — `adapter_model.safetensors`, `adapter_config.json`)
- `eval_metrics.json`, `eval_report.txt`, `train_metrics.json`
- the generated JSONL and run logs

Tips while it runs:
- Runtime for a full fine-tune is hours; you can close the terminal and later just re-check
  with `python -m kaggle kernels status <id>` + `python -m kaggle kernels output <id> -p kaggle_output`.
- Watch progress live in the kernel's page on kaggle.com (Logs tab).
- For a cheap smoke test first, switch the notebook's model config to
  `meta-llama/Llama-3.2-1B` and/or the `_sampled` CSV, push, verify it completes, then
  switch back for the production run.

---

## Step 7 — Inspect results locally

Point [kaggle_results_visualization.ipynb](../../kaggle_results_visualization.ipynb) at
`kaggle_output/` (set `RESULTS_DIR`), or for T1 use
[finetune_vs_baseline_comparison.ipynb](../../finetune_vs_baseline_comparison.ipynb).

---

## Cheat sheet

```powershell
.\env\Scripts\Activate.ps1                                     # always work in the venv

# Data changed? (skip otherwise)
.\kaggle\stage_data.ps1
python -m kaggle datasets version -p kaggle\dataset_payload -m "msg" --dir-mode zip

# New notebook? edit kaggle\kernel-metadata.json: id, title, code_file
.\kaggle\run.ps1                                               # first push creates the kernel
# → on kaggle.com: add HF_TOKEN secret + select GPU T4×2 (one time per kernel)

.\kaggle\run.ps1 -Wait                                         # push + poll + download outputs
# results land in kaggle_output\
```

## Common pitfalls

| Symptom | Cause / fix |
|---|---|
| `kaggle.exe` won't run | Blocked by Windows Application Control — use `python -m kaggle` |
| Push succeeded but old code ran | Notebook file wasn't saved before `run.ps1`, or `code_file` points at the wrong notebook |
| 401/403 downloading the model | `HF_TOKEN` secret missing/not enabled on **this** kernel, or license not accepted on that HF account |
| CSV not found on Kaggle | `--dir-mode zip` flattened `CUAD_v1/` — the notebook's fallback search handles it; keep that cell |
| Writes fail on Kaggle | Only `/kaggle/working` is writable — route all writes through `WORK_DIR` |
| Overwrote the wrong kernel | Pushed with an existing `id` while `code_file` pointed at a different notebook — use a fresh `id` per pipeline |
