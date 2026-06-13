# Pipeline State & Kaggle Interaction — Task 1 (CUAD Risk-Clause Recognition)

**Notebook:** [`llm_fine_tuning_LORA_task1_v2.ipynb`](../../llm_fine_tuning_LORA_task1_v2.ipynb)
**Companion design doc:** [option_a_kaggle_remote_gpu.md](option_a_kaggle_remote_gpu.md)
**Last verified:** 2026-06-13

This document describes **exactly where the pipeline stands today**, how it interacts with Kaggle, and
the concrete differences between running it **locally** versus **on Kaggle's GPU**. It is the
single source of truth for "what works, what doesn't, and why."

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
| **Model load + QLoRA training (cells 18+)** | ❌ **Never executed successfully** | run died at cell `In[18]` |
| Adapter pulled back to local disk | ⛔ Blocked | depends on training succeeding first |

**Bottom line:** the *plumbing* is complete and proven — a notebook really does execute on Kaggle's
GPU and the output comes back to the local `kaggle_output/` folder. **But no fine-tuning has happened
yet.** The run aborts before the training cell because of the two blockers in §7.

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

## 7. Known blockers (why the last run failed before training)

**Blocker 1 — `HF_TOKEN` Kaggle Secret is not attached.**
The last Kaggle run died at cell `In[18]`:

```
UserSecretsClient().get_secret("HF_TOKEN")
→ HTTPError: HTTP Error 400: Bad Request
→ ConnectionError: Connection error trying to communicate with service.
```

A 400 from the secrets service means the secret labelled `HF_TOKEN` is **not found / not attached** to
the notebook. The run aborted here, **before** the model download and training cell ever ran.
*Fix:* attach + enable the `HF_TOKEN` secret (§6), then re-push.

**Blocker 2 — `run.ps1` placeholder guard fires on the real username.**
[`kaggle/run.ps1`](../../kaggle/run.ps1) throws if the kernel id starts with `antonisgantzos/` — but that
**is** the real username, so `.\kaggle\run.ps1` aborts immediately every time. (The guard was meant to
catch the literal `YOUR_KAGGLE_USERNAME` placeholder.) The existing Kaggle run was pushed manually with
`python -m kaggle kernels push -p kaggle`. *Fix:* change the guard to test for the literal placeholder
string, not the username.

---

## 8. The Kaggle integration files

All under [`kaggle/`](../../kaggle/):

| File | Role |
|---|---|
| `kernel-metadata.json` | kernel id, `code_file` (points one level up at the notebook), `enable_gpu: true`, `enable_internet: true`, `dataset_sources` |
| `dataset_payload/dataset-metadata.json` | dataset id/title/license for the uploaded CSVs |
| `dataset_payload/CUAD_v1/*.csv` | the staged cleaned CSVs (the only data Kaggle needs, ~5 MB) |
| `stage_data.ps1` | copies the cleaned CSVs from `data/` into `dataset_payload/` |
| `run.ps1` | convenience wrapper: `push` → poll `status` → `output` (currently broken — Blocker 2) |
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
| Accelerator | none — `torch 2.12.0+cpu`, `cuda_available=False` | T4 GPU (single GPU; `CUDA_VISIBLE_DEVICES="0"` hides the 2nd to avoid DataParallel sharding) |
| `ON_KAGGLE` | `False` | `True` |
| `DATA_DIR` | `data/CUAD_v1` (or `$DATA_DIR`) | `/kaggle/input/cuad-master-clauses-cleaned` (read-only) |
| `WORK_DIR` | repo root `.` | `/kaggle/working` (only persisted dir) |
| HF token source | `.env` / `$env:HF_TOKEN` | Kaggle Secret `HF_TOKEN` (attached + enabled) |
| Cells 1–17 (data prep) | ✅ run | ✅ run |
| Cells 18+ (load + train) | ❌ `Cannot access accelerator device…` | ✅ run (once token blocker cleared) |
| Saved adapter | `WORK_DIR/<new_model_name>` = repo root (won't get here without GPU) | `/kaggle/working/<new_model_name>` → downloadable output |
| Execution style | interactive, cell-by-cell | batch, top-to-bottom (papermill), ~12 h cap, 30 GPU-h/week |
| Dependencies | CPU torch; `dotenv` for token | base image has torch/transformers; `trl` 1.x etc. pinned in a setup cell |

Key training config (cell ~ execution_count 19, on Kaggle): `MODEL_ID="meta-llama/Llama-3.2-1B"`,
`new_model_name="llama-3.2-1B-cuad-task1-smoke-test"`, 4-bit NF4 + `bf16` compute, LoRA targets
`["q_proj","k_proj","v_proj","o_proj"]`, `num_train_epochs=1`, `per_device_train_batch_size=4`,
`max_length=1024`. (`trl` 1.x API: `SFTTrainer` takes an `SFTConfig`, tokenizer via `processing_class=`.)

---

## 10. The run loop (once blockers are cleared)

```powershell
.\env\Scripts\Activate.ps1

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

## 11. Next steps to a green end-to-end run

1. Attach + enable the `HF_TOKEN` Kaggle Secret (or wire the private-dataset token of §8).
2. Fix the `run.ps1` username guard (Blocker 2).
3. Re-push; confirm the log gets **past cell 18** into the `SFTTrainer` loop.
4. Confirm `adapter_model.safetensors` lands in `kaggle_output/`.
5. *(Optional)* add a fast-fail guard at the top of the GPU cells so a local run prints
   "run this on Kaggle" instead of the cryptic accelerator error.
