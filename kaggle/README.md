# Kaggle remote-GPU runner

Runs a repo notebook on Kaggle's free **GPU T4×2** as a batch kernel: push from VS Code, Kaggle runs
it top-to-bottom, pull outputs back to `kaggle_output/`. Which notebook runs is set by `code_file` in
[`kernel-metadata.json`](kernel-metadata.json) (currently
[`../llm_fine_tuning_LORA_task3.ipynb`](../llm_fine_tuning_LORA_task3.ipynb)).

**Full step-by-step guide:** [../docs/kaggle/kaggle_connection_guide.md](../docs/kaggle/kaggle_connection_guide.md).
T1 run record / evidence: [../docs/kaggle/pipeline_state_and_kaggle_interaction.md](../docs/kaggle/pipeline_state_and_kaggle_interaction.md).

> **Always invoke the CLI as `python -m kaggle`.** The `kaggle.exe` shim is blocked by Windows
> Application Control on this machine (the venv has the package installed).

## The pieces

| File | Role |
|---|---|
| `kernel-metadata.json` | kernel `id`, `code_file` (which notebook), `enable_gpu`, `enable_internet`, `dataset_sources` |
| `run.ps1` | push kernel → verify datasets attached → poll status → download outputs (`-Wait`) |
| `stage_data.ps1` | copy the LEDGAR splits into `dataset_payload_ledgar/` (Task 3 data) |
| `stage_hf_token.ps1` | write the HF token into `dataset_payload_hf_token/hf_token.txt` |
| `dataset_payload/` | CUAD CSVs (T1/T2) → dataset `cuad-master-clauses-cleaned` |
| `dataset_payload_ledgar/` | LEDGAR CSVs + labels.json (T3) → dataset `ledgar-lexglue` |
| `dataset_payload_hf_token/` | 1-line `hf_token.txt` → **private** dataset `hf-token` |

Datasets mount read-only at `/kaggle/input/<slug>`; the notebook writes to `/kaggle/working/`, which is
what `run.ps1` downloads.

## Before first run — replace the placeholder

On a fresh clone the metadata may say `YOUR_KAGGLE_USERNAME`. Replace it with your Kaggle username in
`kernel-metadata.json` (`id`, `dataset_sources`) and each `dataset_payload*/dataset-metadata.json` (`id`).

## One-time setup

```powershell
.\env\Scripts\Activate.ps1

# Auth: save the KGAT_... token (kaggle.com -> Settings -> API -> Create New Token).
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Set-Content "$env:USERPROFILE\.kaggle\access_token" "KGAT_your_new_token" -NoNewline -Encoding ascii
python -m kaggle kernels list --mine        # auth check

# 1. Upload the HF token as a PRIVATE dataset (headless — no web Secret; see below).
$env:HF_TOKEN = "hf_..."            # or have it in .env at the repo root
.\kaggle\stage_hf_token.ps1
python -m kaggle datasets create -p kaggle\dataset_payload_hf_token   # -p has no trailing slash

# 2. Upload the data the notebook needs (only its own task's dataset):
#    CUAD (T1/T2) — subfolder is flattened, so use --dir-mode zip:
python -m kaggle datasets create -p kaggle\dataset_payload --dir-mode zip
#    LEDGAR (T3) — staged flat, NO --dir-mode zip, slash-free -p (run from kaggle\):
.\kaggle\stage_data.ps1
python -m kaggle datasets create -p dataset_payload_ledgar

# 3. First kernel push (run.ps1 verifies every declared dataset attaches).
.\kaggle\run.ps1
#    Then on kaggle.com open the kernel -> Settings -> Accelerator -> GPU T4 x2 (once per kernel).
#    Accept the gated model license on huggingface.co with the token's account.
```

### HF token: private dataset, not a web Secret

`kernel-metadata.json` can declare **datasets** but **not secrets**, and the Kaggle API cannot attach a
secret. If you instead add the `HF_TOKEN` secret in the web editor and click **Save Version**, that commit
takes the editor's inputs and **silently empties `dataset_sources`** — every input detaches.

So the token rides along as a second **private** `dataset_source` (`antonisgantzos/hf-token`, a 1-line
`hf_token.txt`). A plain `kernels push` then carries the data **and** the token; the web UI is never
touched. The notebook's `get_hf_token()` reads `/kaggle/input/*/hf_token.txt` first, falling back to the
Secrets vault, then local `.env`.

> Never click **Save Version** / **Save & Run All** in the Kaggle web editor for this kernel — it empties
> `dataset_sources`. Drive every run from the CLI (`run.ps1`); its guard aborts loudly if any declared
> dataset fails to attach.

Rotate the token later:

```powershell
$env:HF_TOKEN = "hf_new_..."
.\kaggle\stage_hf_token.ps1
python -m kaggle datasets version -p kaggle\dataset_payload_hf_token -m "rotate token"
```

## Run loop

```powershell
.\kaggle\run.ps1 -Wait     # push, verify datasets, poll status, download outputs to ..\kaggle_output
```

Update a dataset after its files change:

```powershell
# CUAD
.\kaggle\stage_data.ps1  # (edit for CUAD) — or re-copy the CSVs into dataset_payload\CUAD_v1
python -m kaggle datasets version -p kaggle\dataset_payload -m "update csvs" --dir-mode zip
# LEDGAR
.\kaggle\stage_data.ps1
python -m kaggle datasets version -p dataset_payload_ledgar -m "update ledgar"
```

Outputs (adapter `adapter_model.safetensors` + `adapter_config.json`, `eval_metrics.json`,
`eval_report.txt`, `train_metrics.json`, logs) land under `..\kaggle_output\`.
