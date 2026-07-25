# Kaggle Connection Guide

How this repo talks to Kaggle: deploy data, push code, run on the free GPU, pull results.
Short and operational. Deeper rationale lives in
[pipeline_state_and_kaggle_interaction.md](pipeline_state_and_kaggle_interaction.md).

> **Always call the CLI as `python -m kaggle`.** The `kaggle.exe` shim is blocked by Windows
> Application Control on this machine.

---

## 1. How the connection works (mental model)

Batch remote execution — **not** an interactive/SSH session. You edit locally, push a copy,
Kaggle runs the notebook top-to-bottom on its GPU, you pull the outputs back.

```
  VS Code (local, CPU)              Kaggle (cloud, T4 GPU)
  ──────────────────                ──────────────────────
  notebook  ──[kernels push]──────▶ stores a COPY, runs it top-to-bottom
  data      ──[datasets version]──▶ mounts read-only at /kaggle/input/<slug>
  kaggle_output/ ◀─[kernels output] /kaggle/working/  (only writable + persisted dir)
```

Three rules:
1. **Local edits are invisible until pushed.** Notebook → `kernels push`; data → `datasets version`.
2. **Save the notebook file before pushing** — push uploads the file on disk, not unsaved cells.
3. **Each run is a fresh container.** Only files written to `/kaggle/working/` survive and come back.

### The three moving parts

| Part | What it is | CLI verb |
|---|---|---|
| **Kernel** | the notebook + its settings, from `kaggle/kernel-metadata.json` | `kernels push` |
| **Datasets** | inputs mounted read-only at `/kaggle/input/<slug>` | `datasets create` / `version` |
| **Output** | everything the run wrote to `/kaggle/working/` | `kernels output` |

### Datasets used

| Dataset slug | Holds | Payload folder | Mount |
|---|---|---|---|
| `antonisgantzos/cuad-master-clauses-cleaned` | T1/T2 CSVs | `kaggle/dataset_payload` | `/kaggle/input/cuad-master-clauses-cleaned` |
| `antonisgantzos/ledgar-lexglue` | T3 CSVs + labels.json | `kaggle/dataset_payload_ledgar` | `/kaggle/input/ledgar-lexglue` |
| `antonisgantzos/hf-token` | 1-line `hf_token.txt` | `kaggle/dataset_payload_hf_token` | `/kaggle/input/hf-token` |

The HF token rides as a **private dataset**, not a web Secret — a plain `kernels push` then carries
data *and* token. Never click **Save Version** in the Kaggle web editor: it silently empties
`dataset_sources` and detaches every input.

### Kernel settings (`kaggle/kernel-metadata.json`)

| Field | Value | Meaning |
|---|---|---|
| `code_file` | `../llm_fine_tuning_LORA_task3.ipynb` | **which notebook gets pushed** — edit to switch tasks |
| `enable_gpu` | `true` | request a GPU |
| `enable_internet` | `true` | needed to download the base model from HF |
| `is_private` | `true` | kernel not public |
| `dataset_sources` | data + token slugs | mounted read-only under `/kaggle/input` |

### GPU spec for the run

- **Accelerator: GPU T4×2** — set it in the notebook's **Settings → Accelerator** (the metadata only
  says `enable_gpu: true`; the exact machine is chosen in the web UI once per kernel and sticks).
- The notebook's first cell masks the second GPU (`CUDA_VISIBLE_DEVICES="0"`) so `Trainer` stays on
  one device.
- Limits: ~12 h max runtime per session, ~30 GPU-hours/week (Kaggle free tier).

---

## 2. GUIDE — run a repo notebook on Kaggle (do this)

### A. One-time setup

```powershell
.\env\Scripts\Activate.ps1

# 1. Auth: Kaggle.com → Settings → API → Create New Token, save the KGAT_ token.
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Set-Content "$env:USERPROFILE\.kaggle\access_token" "KGAT_your_token" -NoNewline -Encoding ascii
python -m kaggle kernels list --mine        # should list your kernels = auth OK

# 2. Upload the HF token as a private dataset (once).
$env:HF_TOKEN = "hf_..."                     # or put HF_TOKEN=... in .env at repo root
.\kaggle\stage_hf_token.ps1
python -m kaggle datasets create -p kaggle\dataset_payload_hf_token

# 3. Accept the gated model's license on huggingface.co with that token's account.
```

### B. Upload the data (once per task, re-run when the data changes)

**Task 1 / Task 2 (CUAD):**
```powershell
.\kaggle\stage_data.ps1
python -m kaggle datasets create  -p kaggle\dataset_payload --dir-mode zip   # first time
python -m kaggle datasets version -p kaggle\dataset_payload --dir-mode zip -m "update csvs"  # updates
```

**Task 3 (LEDGAR):** — note: **no** `--dir-mode zip` (files are staged flat), and use a slash-free
`-p` path.
```powershell
.\kaggle\stage_data.ps1  # LEDGAR variant
python -m kaggle datasets create  -p dataset_payload_ledgar                      # first time (run from kaggle\)
python -m kaggle datasets version -p dataset_payload_ledgar -m "update ledgar"   # updates
```

Confirm it's ready before pushing a kernel:
```powershell
python -m kaggle datasets status antonisgantzos/<slug>
```

### C. Pick the notebook to run

Edit [`kaggle/kernel-metadata.json`](../../kaggle/kernel-metadata.json):
- set `code_file` to the notebook you want (e.g. `../llm_fine_tuning_LORA_task1_v2.ipynb`)
- set `dataset_sources` to the datasets that notebook needs (its data slug + `antonisgantzos/hf-token`)
- keep the same `id` to overwrite the existing kernel and its run history; give it a **new `id`** to
  create a separate kernel (each `id` is one kernel on Kaggle)

**Save the notebook file** in VS Code.

### D. Set the GPU (once per kernel)

After the first push, open the kernel on kaggle.com → **Settings → Accelerator → GPU T4 x2**.

### E. Push, wait, pull

```powershell
.\kaggle\run.ps1 -Wait
```
This pushes the kernel, verifies every declared dataset actually attached (aborts loudly if not),
polls status until `complete`/`error`, then downloads outputs to `kaggle_output\`.

Prefer manual control? The three underlying commands:
```powershell
python -m kaggle kernels push   -p kaggle
python -m kaggle kernels status antonisgantzos/<kernel-id>
python -m kaggle kernels output antonisgantzos/<kernel-id> -p kaggle_output
```

### F. Results

Everything the run wrote to `/kaggle/working/` lands in `kaggle_output\`: the trained adapter
(`adapter_model.safetensors`, `adapter_config.json`), `eval_metrics.json`, `eval_report.txt`,
`train_metrics.json`, the rendered notebook, and the log.

---

## 3. Fast troubleshooting

| Symptom | Cause / fix |
|---|---|
| `kaggle.exe` not found / blocked | Use `python -m kaggle`, never the shim. |
| Kernel runs but dies on first `/kaggle/input/...` read | Dataset wasn't uploaded/`ready` before push — the CLI silently drops missing sources. Upload it, check `datasets status`, re-push. `run.ps1` guards against this. |
| `not valid dataset sources` on push | Same — the dataset slug in `dataset_sources` doesn't exist yet. Upload it first. |
| Inputs detached after a run | Someone clicked **Save Version** in the web editor. Re-push from the CLI; never save from the web UI. |
| HF 401 / gated model error | Token missing/invalid, or the license isn't accepted on that HF account. |
| Device / DataParallel error on `cuda:1` | Accelerator isn't GPU T4×2, or the GPU-masking first cell didn't run. |
| Task 3 dataset upload builds a broken path | Don't use a forward slash in `-p` for the LEDGAR payload; run from `kaggle\` with `-p dataset_payload_ledgar`. |
