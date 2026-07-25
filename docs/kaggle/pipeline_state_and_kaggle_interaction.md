# Pipeline State & Kaggle Interaction — Task 1 (CUAD Risk-Clause Recognition)

**Notebook:** [`llm_fine_tuning_LORA_task1_v2.ipynb`](../../llm_fine_tuning_LORA_task1_v2.ipynb)
**Operational guide:** [kaggle_connection_guide.md](kaggle_connection_guide.md)
**Last verified:** 2026-06-13 — **full end-to-end run succeeded on Kaggle (T4×2).**

This document describes **exactly where the pipeline stands today**, how it interacts with Kaggle, and
the concrete differences between running it **locally** versus **on Kaggle's GPU**. It is the
single source of truth for "what works, what doesn't, and why."

> **Note (setup has since evolved).** For the current operational steps, read
> **[kaggle_connection_guide.md](kaggle_connection_guide.md)** — it supersedes the procedure here. Two
> things changed after this record was written: (1) the **HF token is now delivered as a private
> `hf-token` dataset**, not a web Secret (§6/§7-A below describe the older Secret approach; both work,
> the dataset is preferred and fully headless); (2) the `run.ps1` username-guard bug noted in §7 is
> **fixed** — `run.ps1` works and also verifies every declared dataset actually attaches.

> **Status: GREEN.** After attaching the `HF_TOKEN` secret and selecting the **GPU T4×2** accelerator,
> the notebook trained end-to-end on Kaggle: final training loss **0.3035**, validation accuracy
> **0.91**, adapter saved to `/kaggle/working/` and downloadable. See §2 and §7.

---

## 1. The goal in one sentence

Drive the fine-tuning notebook on **Kaggle's free GPU from VS Code**, via the Kaggle CLI — edit code
locally, push it, let Kaggle run the training in batch, pull the trained adapter back — **without
opening the Kaggle web editor** to run cells.

---

## 2. Current state — what works and what doesn't

| Stage | Status | Evidence |
|---|---|---|
| CLI auth (`python -m kaggle`) | ✅ Works | `kernels list --mine` authenticates |
| Data uploaded as a Kaggle Dataset | ✅ Works | mounts at `/kaggle/input/cuad-master-clauses-cleaned` |
| Notebook is environment-aware (`ON_KAGGLE`) | ✅ Works | log shows `ON_KAGGLE=True \| DATA_DIR=… \| WORK_DIR=/kaggle/working` |
| Push → run on GPU container → pull output | ✅ Works | `kaggle_output/` contains pulled JSONL + log |
| Data-prep half (load, clean, label, write JSONL) | ✅ Runs on Kaggle | log: `Saved 1282 training samples and 448 validation samples.` |
| HF gated-model access via `HF_TOKEN` secret | ✅ Works | `Token belongs to: AntonisGantzos123`; `Access granted to meta-llama/Llama-3.2-1B` |
| **Model load + QLoRA training (cells 18+)** | ✅ **Works on Kaggle GPU** | `Training finished — final training loss: 0.3035`; saved to `/kaggle/working/llama-3.2-1B-cuad-task1-smoke-test/` |
| Validation evaluation | ✅ Works | accuracy **0.91** (Yes F1 0.85 / No F1 0.93), macro-F1 0.89 |
| Adapter as downloadable output | ✅ Works | `adapter_model.safetensors`, `adapter_config.json`, `README.md` in `/kaggle/working/` → pull with `kernels output` |

**Bottom line:** the pipeline now runs **end-to-end on Kaggle** — push from VS Code, train on the T4×2
GPU, pull the adapter back to `kaggle_output/`. The two former blockers (HF secret, GPU type) are
resolved; see §7 for what they were and the prerequisites that must stay in place.

---

## 3. Mental model — how local and Kaggle relate

This is **batch remote execution**, not a live/interactive GPU session. There is no SSH, no
cell-by-cell execution against the remote GPU, no VS Code attach.

```
  VS Code (local, CPU)                 Kaggle (cloud, T4 GPU)
  ────────────────────                 ──────────────────────
  edit notebook  ──────[push]────────▶ stores a COPY of the notebook
  edit CSVs      ──[datasets version]▶ stores a COPY of the data
                                       runs the notebook top-to-bottom (papermill)
  kaggle_output/ ◀─────[output]─────── /kaggle/working/  (the only persisted dir)
```

Three rules that follow from this model:

1. **local = draft, push = publish, run = execute the last published version.** Edits made locally are
   invisible to Kaggle until the next `push`. Same for data: a CSV edit needs its own `datasets version`.
2. **Save before push.** Unsaved Jupyter cells are not on disk; `push` uploads the file on disk.
3. **Each run is a fresh container.** Anything not written under `/kaggle/working/` is gone after the run.

---

## 4. The environment-aware notebook (one file, two environments)

The same notebook runs locally and on Kaggle. A config cell near the top detects the environment and
sets paths/token source accordingly:

```python
ON_KAGGLE = Path("/kaggle").exists()
if ON_KAGGLE:
    DATA_DIR = Path("/kaggle/input/cuad-master-clauses-cleaned")  # read-only mounted dataset
    WORK_DIR = Path("/kaggle/working")                            # only writable + persisted dir
else:
    DATA_DIR = Path(os.getenv("DATA_DIR", "data")) / "CUAD_v1"
    WORK_DIR = Path(".")                                          # repo root
```

Everything downstream keys off `ON_KAGGLE`, `DATA_DIR`, and `WORK_DIR`:

- **Reads** use `DATA_DIR` (`CUAD_PATH = DATA_DIR`).
- **Writes** (generated JSONL, saved adapter) go to `WORK_DIR` — critical because `/kaggle/input` is
  **read-only** on Kaggle, so writes must land in `/kaggle/working`.
- A **CSV fallback search** handles a Kaggle mount quirk: `--dir-mode zip` flattens the `CUAD_v1/`
  subfolder into the dataset root, so if the expected path is missing the notebook searches
  `/kaggle/input` for the CSV (log: `Resolved CSV via fallback search: …`).
- The **HF token** comes from Kaggle Secrets on Kaggle, and from `.env` / an env var locally (see §6).

---

## 5. The GPU wall — why cells 18+ cannot run locally

The local machine has **no accelerator**. Verified:

```
torch 2.12.0+cpu   cuda_available = False   device_count = 0
```

The model-load cell (execution_count 19) is **GPU-only by construction**:

```python
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)   # 4-bit quant = bitsandbytes = CUDA-only
model = AutoModelForCausalLM.from_pretrained(..., device_map={"": 0})  # "put model on GPU 0"
```

Running it locally raises:

> `RuntimeError: Cannot access accelerator device when none is available.`

This is **expected, not a bug** — it is the exact reason the Kaggle integration exists. `device_map={"": 0}`
demands accelerator device 0, and 4-bit `bitsandbytes` has no CPU path. The training cell, inference
cell, and the pre-flight `torch.cuda` checks are all in the same GPU-only region.

**Consequence — split the notebook by where each half can run:**

| Cells | What they do | Where they run |
|---|---|---|
| ~1–17 | load CSV, clean columns, build Yes/No labels, balance, write JSONL | **Locally (CPU)** or Kaggle — no GPU needed |
| 18 | HF token + gated-model access check | both (token source differs, see §6) |
| 19+ | 4-bit model load → QLoRA `SFTTrainer` train → save adapter → inference | **Kaggle GPU only** |

Iterate on data/logic locally with cells 1–17; push to Kaggle to actually train.

---

## 6. Secrets / HF token — the one place behaviour diverges hard

The notebook needs an `HF_TOKEN` because the base model `meta-llama/Llama-3.2-1B` is **gated**.

```python
def get_hf_token():
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")   # Kaggle Secrets vault
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("HF_TOKEN")                            # local .env / shell env var
```

- **Locally:** set it in the shell before running — `$env:HF_TOKEN = "hf_..."` — or put it in a `.env`
  file at the repo root. (`.env` is **not** uploaded to Kaggle.)
- **On Kaggle:** the token must exist as a **Kaggle Secret named exactly `HF_TOKEN`, attached and enabled
  on the notebook.** The Kaggle API **cannot create or attach secrets** — this is the one unavoidable
  ~30-second Kaggle-website visit (Add-ons → Secrets), unless you use the private-dataset workaround in §8.
- The HF account that owns the token must have **accepted the gated-model license** on huggingface.co.

---

## 7. Prerequisites & resolved blockers

Two things had to be in place before training would run. **Both are now satisfied** — keep them in place
for every future run.

### Prerequisite A — `HF_TOKEN` Kaggle Secret must be attached *(RESOLVED)*

An earlier run died at cell `In[18]` with:

```
UserSecretsClient().get_secret("HF_TOKEN")
→ HTTPError: HTTP Error 400: Bad Request   (secret not found / not attached)
```

After adding the secret (Add-ons → Secrets, named exactly `HF_TOKEN`, enabled), the successful run logs:

```
Token belongs to: AntonisGantzos123  (type: user)
✅ Access granted to meta-llama/Llama-3.2-1B — you can run the load cell below.
```

Keep this secret attached and enabled on the notebook (and keep the HF gated-model license accepted).

### Prerequisite B — accelerator MUST be set to **GPU T4×2** *(REQUIRED before every run)*

**Before executing the notebook on Kaggle, set the accelerator to “GPU T4×2”** (notebook **Settings →
Accelerator → GPU T4 x2**). This is required — with the wrong accelerator the run hits device conflicts.

The notebook is written for the T4×2 machine: it exposes **two** GPUs, and the very first code cell
masks the second one so training stays on a single device:

```python
# Kaggle "GPU T4 x2" exposes 2 GPUs. device_map="auto" would shard across cuda:0/cuda:1 and
# Trainer's DataParallel needs everything on cuda:0 -> "...found one on cuda:1". Hide GPU 1.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

So the contract is: **select T4×2 + the notebook masks GPU 1 → no DataParallel / sharding conflict.**
The successful run used exactly this and trained cleanly. (Note: `kernel-metadata.json` only carries
`enable_gpu: true`; the *specific* T4×2 machine type is chosen in the notebook's accelerator setting.)

### `run.ps1` placeholder guard bug — *FIXED*

The guard in [`kaggle/run.ps1`](../../kaggle/run.ps1) now correctly tests for the literal
`YOUR_KAGGLE_USERNAME` placeholder (not the real `antonisgantzos/` username), so `.\kaggle\run.ps1`
runs cleanly. It additionally pulls the pushed kernel's metadata back and **aborts loudly if any
declared `dataset_source` failed to attach** (the CLI otherwise silently drops a missing dataset and
the run dies on the first `/kaggle/input/...` read).

---

## 7a. Successful run — results (reference)

From the green run (`cuad-task1-finetune` log, ~27 min wall-clock total):

| Metric | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B` (4-bit QLoRA, smoke-test config) |
| Training data | 1282 balanced examples (641 Yes / 641 No), 1 epoch |
| Training time | `Starting training...` @ 79s → `Training finished` @ 1490s (~23.5 min) |
| Final training loss | **0.3035** |
| Saved artifacts | `/kaggle/working/llama-3.2-1B-cuad-task1-smoke-test/` → `adapter_config.json`, `adapter_model.safetensors`, `README.md` |
| Validation accuracy | **0.91** (448 examples) |
| Per-class | Yes: P 0.78 / R 0.93 / F1 0.85 (n=125) · No: P 0.97 / R 0.90 / F1 0.93 (n=323) |
| Confusion matrix | `[[116, 9], [32, 291]]` (rows=true [Yes,No], cols=pred [Yes,No]) |

Dependency-resolver `ERROR:` lines about `numba-cuda` / `cudf` / `dask-cuda` early in the log are
**harmless** — they come from the Kaggle base image's preinstalled RAPIDS stack, not from this notebook,
and do not affect training.

---

## 8. The Kaggle integration files

All under [`kaggle/`](../../kaggle/):

| File | Role |
|---|---|
| `kernel-metadata.json` | kernel id, `code_file` (points one level up at the notebook), `enable_gpu: true`, `enable_internet: true`, `dataset_sources` |
| `dataset_payload/dataset-metadata.json` | dataset id/title/license for the uploaded CSVs |
| `dataset_payload/CUAD_v1/*.csv` | the staged cleaned CSVs (the only data Kaggle needs, ~5 MB) |
| `stage_data.ps1` | copies the cleaned CSVs from `data/` into `dataset_payload/` |
| `run.ps1` | convenience wrapper: `push` → poll `status` → `output` (currently aborts on the username-guard bug, §7 — push manually for now) |
| `README.md` | the operational run-book |

> Note: on this machine the `kaggle.exe` shim is blocked by Windows Application Control, so the CLI is
> always invoked as **`python -m kaggle`** (the venv has the package installed).

`run.ps1` is **optional sugar** — it just wraps three plain CLI commands. You can always run them by hand:

```powershell
python -m kaggle kernels push   -p kaggle
python -m kaggle kernels status  antonisgantzos/cuad-task1-finetune
python -m kaggle kernels output  antonisgantzos/cuad-task1-finetune -p kaggle_output
```

**Fully-headless token alternative (avoids the one UI visit in §6):** put the HF token in a 1-line
**private** Kaggle dataset, mount it like the CSVs, and read it from the file instead of `get_secret`.
Trade-off: the token lives in a private dataset rather than the secrets vault.

---

## 9. Local vs Kaggle — side-by-side

| Aspect | Local (VS Code) | Kaggle (pushed kernel) |
|---|---|---|
| Accelerator | none — `torch 2.12.0+cpu`, `cuda_available=False` | **GPU T4×2** (must be selected, §7-B); notebook masks GPU 1 via `CUDA_VISIBLE_DEVICES="0"` to avoid DataParallel sharding |
| `ON_KAGGLE` | `False` | `True` |
| `DATA_DIR` | `data/CUAD_v1` (or `$DATA_DIR`) | `/kaggle/input/cuad-master-clauses-cleaned` (read-only) |
| `WORK_DIR` | repo root `.` | `/kaggle/working` (only persisted dir) |
| HF token source | `.env` / `$env:HF_TOKEN` | Kaggle Secret `HF_TOKEN` (attached + enabled) |
| Cells 1–17 (data prep) | ✅ run | ✅ run |
| Cells 18+ (load + train) | ❌ `Cannot access accelerator device…` | ✅ run (HF secret + T4×2 in place) |
| Saved adapter | `WORK_DIR/<new_model_name>` = repo root (won't get here without GPU) | `/kaggle/working/<new_model_name>` → downloadable output |
| Execution style | interactive, cell-by-cell | batch, top-to-bottom (papermill), ~12 h cap, 30 GPU-h/week |
| Dependencies | CPU torch; `dotenv` for token | base image has torch/transformers; `trl` 1.x etc. pinned in a setup cell |

Key training config (cell ~ execution_count 19, on Kaggle): `MODEL_ID="meta-llama/Llama-3.2-1B"`,
`new_model_name="llama-3.2-1B-cuad-task1-smoke-test"`, 4-bit NF4 + `bf16` compute, LoRA targets
`["q_proj","k_proj","v_proj","o_proj"]`, `num_train_epochs=1`, `per_device_train_batch_size=4`,
`max_length=1024`. (`trl` 1.x API: `SFTTrainer` takes an `SFTConfig`, tokenizer via `processing_class=`.)

---

## 10. The run loop (working)

```powershell
.\env\Scripts\Activate.ps1

# 0. ONE-TIME / per-run on Kaggle: HF_TOKEN secret attached (§7-A) + accelerator = GPU T4×2 (§7-B)

# 1. (only if CSVs changed) re-stage + version the dataset
.\kaggle\stage_data.ps1
python -m kaggle datasets version -p kaggle\dataset_payload -m "update csvs" --dir-mode zip

# 2. push the notebook (uploads the on-disk file — save it first)
python -m kaggle kernels push -p kaggle

# 3. poll until complete
python -m kaggle kernels status antonisgantzos/cuad-task1-finetune

# 4. pull outputs (rendered notebook, logs, saved adapter) into kaggle_output/
python -m kaggle kernels output antonisgantzos/cuad-task1-finetune -p kaggle_output
```

A successful run ends with `adapter_model.safetensors` + `adapter_config.json` under `kaggle_output/`.

---

## 11. Next steps / improvements

1. **Persist evaluation + training results as files in `/kaggle/working/`** so they come back in
   `kaggle_output/` instead of living only in the run log. Today the metrics
   (accuracy / precision / recall / F1, confusion matrix) and the final training loss are only
   `print`ed — they show in the Kaggle log but are **not** downloadable artifacts. Have the notebook
   write them to `WORK_DIR` next to the adapter, e.g.:
   - `WORK_DIR / "eval_metrics.json"` — the classification report as a dict (`classification_report(..., output_dict=True)`), plus the confusion matrix and overall accuracy.
   - `WORK_DIR / "eval_report.txt"` — the human-readable `classification_report` string.
   - `WORK_DIR / "train_metrics.json"` — final training loss + key `SFTConfig` hyper-params (so each run is self-describing).

   After this, `kaggle kernels output` pulls the trained adapter **and** the eval/training results into
   `kaggle_output/` in one step, giving a complete, reproducible record per run.
2. Fix the `run.ps1` username guard (§7) so the wrapper works instead of pushing manually.
3. *(Optional)* add a fast-fail guard at the top of the GPU cells so a local run prints
   "run this on Kaggle" instead of the cryptic accelerator error.
4. *(Optional)* scale past the smoke test: larger sample / more epochs, or swap to the 8B base model.
