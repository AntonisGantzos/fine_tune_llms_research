# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for fine-tuning LLMs on legal contract review tasks using QLoRA. Work is done in Jupyter notebooks, edited locally (VS Code, CPU-only machine) and **trained remotely on Kaggle's free GPU** via the Kaggle CLI — see "Kaggle Workflow" below.

**Three tasks:**
- **T1 – Risk Clause Recognition:** Binary Yes/No clause identification from CUAD (all clause categories not used by T2). **Implemented and evaluated** — fine-tuned vs. baseline comparison done.
- **T2 – Structured Entity Extraction:** Extract dates/names/terms as valid JSON (9 categories: Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period To Terminate Renewal, Governing Law, Warranty Duration) from CUAD. **In progress** — notebook and JSONL generation exist; training run pending.
- **T3 – Jurisdiction Identification:** Provision classification from LEDGAR. **In progress** — notebook [llm_fine_tuning_LORA_task3.ipynb](llm_fine_tuning_LORA_task3.ipynb) exists, LEDGAR data staged as a Kaggle dataset, and `kaggle/kernel-metadata.json` currently targets this notebook; training run pending.

## Environment Setup

```powershell
# Activate the local venv
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Notes:
- `nbstripout` is configured — notebook outputs are stripped on commit.
- The local machine has **no GPU** (`torch` is CPU-only). Data-prep cells run locally; model-loading/training cells only run on Kaggle.
- HF token: from `.env` / env var locally, from Kaggle Secrets (`HF_TOKEN`) on Kaggle. Gated Llama models require an accepted license on the same HF account.

## Data Layout

The `DATA_DIR` environment variable controls the data root (defaults to `data`). Raw CUAD lives at `data/CUAD_v1/` (git-ignored):
- `data/CUAD_v1/master_clauses.csv` — primary source (545 contracts, 41 clause categories)
- `data/CUAD_v1/master_clauses_cleaned.csv` — preprocessed version used by the training notebooks (negative placeholders replaced with real non-related text)
- `data/CUAD_v1/master_clauses_cleaned_sampled.csv` — small sample for quick iteration
- `data/CUAD_v1/CUAD_v1.json` — SQuAD-format JSON (EDA only, not for training)

Generated JSONL (written by the notebooks to `WORK_DIR/cuad/`, which locally is the repo root):
- `cuad/train/cuad_train.jsonl` + `cuad/validation/cuad_validation.jsonl` — T1
- `cuad/train/cuad_task2_train.jsonl` + `cuad/validation/cuad_task2_validation.jsonl` — T2

Helper scripts in `scripts/`: `download_cuad.py`, `preprocess_values_cuad.py` (produces the cleaned CSV), `sample_master_clauses.py`.

## Notebooks

- [CUAD_dataset_exploration.ipynb](CUAD_dataset_exploration.ipynb) — EDA: class imbalance, context lengths, instruction-format preview
- [llm_fine_tuning_LORA_task1_v2.ipynb](llm_fine_tuning_LORA_task1_v2.ipynb) — **T1 training pipeline**: cleaned CSV → JSONL → QLoRA fine-tune via `SFTTrainer` → validation eval (`eval_metrics.json`, `eval_report.txt`)
- [llm_fine_tuning_LORA_task2.ipynb](llm_fine_tuning_LORA_task2.ipynb) — **T2 training pipeline**, same structure as T1 v2 (33 cells, mirrors it deliberately)
- [llm_fine_tuning_LORA_task3.ipynb](llm_fine_tuning_LORA_task3.ipynb) — **T3 training pipeline** (LEDGAR provision classification); builds JSONL on Kaggle from the staged LEDGAR splits. Current `kernel-metadata.json` target.
- [llama_3.1_task_1_no_fine_tune.ipynb](llama_3.1_task_1_no_fine_tune.ipynb) — T1 **baseline**: evaluates the un-fine-tuned base model on the same validation set; writes to `no_finetune_baseline/`
- [kaggle_results_visualization.ipynb](kaggle_results_visualization.ipynb) — visualizes one run directory (point `RESULTS_DIR` at e.g. `kaggle_output/`)
- [finetune_vs_baseline_comparison.ipynb](finetune_vs_baseline_comparison.ipynb) — compares fine-tuned vs. baseline T1 metrics from `kaggle_output/`

Training notebooks are **environment-aware**: a config cell sets `ON_KAGGLE = Path("/kaggle").exists()` and derives `DATA_DIR`/`WORK_DIR` from it. Reads go through `DATA_DIR`, writes through `WORK_DIR` (`/kaggle/working` on Kaggle — the only writable/persisted dir). Keep this pattern when editing.

## Kaggle Workflow (remote GPU)

Batch remote execution — push notebook, Kaggle runs it top-to-bottom, pull outputs. No interactive remote session. **Step-by-step guide: [docs/kaggle/kaggle_connection_guide.md](docs/kaggle/kaggle_connection_guide.md)** and [kaggle/README.md](kaggle/README.md); T1 run record: [docs/kaggle/pipeline_state_and_kaggle_interaction.md](docs/kaggle/pipeline_state_and_kaggle_interaction.md).

```powershell
.\kaggle\run.ps1 -Wait     # push kernel, verify datasets attach, poll, download outputs to kaggle_output/
.\kaggle\stage_data.ps1    # stage LEDGAR (T3), then: python -m kaggle datasets version -p dataset_payload_ledgar -m "..."
```

Key facts:
- Always invoke the CLI as **`python -m kaggle`** (the `kaggle.exe` shim is blocked by Windows Application Control on this machine).
- `kaggle/kernel-metadata.json` → `code_file` selects **which notebook** gets pushed (currently the T3 notebook); edit it to switch tasks/runs. Keep the same `id` to overwrite that kernel, use a new `id` for a separate kernel.
- **Three datasets**, mounted read-only at `/kaggle/input/<slug>`: `cuad-master-clauses-cleaned` (T1/T2, uploaded with `--dir-mode zip` → subfolder flattened, so notebooks have a CSV fallback search), `ledgar-lexglue` (T3, staged flat, **no** `--dir-mode zip`), and the private `hf-token`.
- **HF token is delivered as the private `hf-token` dataset, not a web Secret** — the API can't attach secrets, and clicking *Save Version* in the web editor silently empties `dataset_sources`. `get_hf_token()` reads `/kaggle/input/*/hf_token.txt`, falling back to the Secrets vault, then local `.env`. Never save from the web editor; drive runs from the CLI.
- Local edits are invisible until pushed; save the notebook file before `run.ps1`.
- Use accelerator **GPU T4×2** (set once per kernel in the notebook's Settings on kaggle.com).
- Outputs land in `kaggle_output/` (git-ignored): adapter, `eval_metrics.json`, `eval_report.txt`, `train_metrics.json`, logs, generated JSONL.

## Data Format & Schema

`master_clauses.csv` column pairs: `[Category]` (clause text context) + `[Category]-Answer` (ground truth). Answer = `"No"` → clause absent (T1 negative); answer = text/date/name → clause present (T1 positive / T2 value).

Training JSONL schemas (both include a `category` field for per-category eval):
```json
// T1
{"instruction": "Is the following contract text a \"[Category]\" clause? Answer strictly \"Yes\" or \"No\".",
 "category": "[Category]", "input": "<clause text>", "output": "Yes|No"}
// T2
{"instruction": "Extract the \"[Category]\" from the contract text below. Return the result as a JSON object with the single key \"[Category]\". If multiple values exist, return them as a list of strings. If the value is not present, use null.",
 "category": "[Category]", "input": "<clause text>", "output": "{\"[Category]\": \"<value>\"}"}
```

Prompt template used during training (completion = `output`):
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

## QLoRA Configuration

Target model: `meta-llama/Meta-Llama-3.1-8B` (production runs); `meta-llama/Llama-3.2-1B` is the smoke-test option in the same notebooks.

Key settings (same in T1 v2 and T2 notebooks):
- 4-bit NF4 quantization via `BitsAndBytesConfig`
- LoRA rank `r=16`, `lora_alpha=16`, targets `["q_proj", "k_proj", "v_proj", "o_proj"]`
- Optimizer `paged_adamw_32bit`
- Adapter saved under `WORK_DIR` (e.g. `llama-3.1-8B-cuad-task1/`)

## Documentation

`docs/` is organized by topic — check the relevant subfolder before reworking a pipeline:
- `docs/task_1/` — T1 design, LoRA application, hard-negatives plan
- `docs/task_2/` — T2 logic, entity-extraction plan, notebook cell guide
- `docs/kaggle/` — **`kaggle_connection_guide.md`** (how the connection works + step-by-step run guide) and `pipeline_state_and_kaggle_interaction.md` (T1 end-to-end run record)
- `docs/data_processing/` — CUAD data roles, contract→clause mapping, table→examples
- `docs/MODEL_SELECTION.md`, `docs/GPU_SCALING_OPTIONS.md`, `docs/llm_finetuning_parameters.md`

## Important Data Decisions

- **Train/val split is contract-level** (not clause-level) at 85/15 to prevent leakage — a contract's clauses must not appear in both sets
- T1/T2 use disjoint category sets: the 9 T2 entity categories (plus `Filename`) are excluded from T1's Yes/No categories
- `CUAD_v1.json` is SQuAD extractive format; it is **not used for training** generative models
- `full_contract_pdf/` and `full_contract_txt/` are intentionally unused — raw contracts exceed context windows and the CSV already contains extracted clauses
- Class imbalance is significant for T1 (rare risk clauses <10% frequency); negatives use real non-related clause text (not placeholders) so the model sees hard "No" examples

## Working Principles

Guidance for how to write code in this repo. The governing rule: **build the simplest thing that works, add complexity only when reality forces it.**

- **KISS / YAGNI / no premature optimization.** Solve today's requirement the simplest way. Don't add abstractions, config, or speed hacks for imagined future needs. Before writing: *is there a shorter, clearer way, and is this needed now?*
- **Question every dependency.** A new package is complexity forever — don't pull one in for what a small function solves. `torch` here is CPU-only; heavy model code only runs on Kaggle.
- **Plan before large changes; surface unknowns first.** For anything non-trivial, lay out the approach and ask before guessing (which notebook/task, which categories, edge cases). Work in small, reviewable increments.
- **Prove success with evidence, don't assert it.** Report the exact command run and its output. Never claim a training run or eval "works" without the actual result — real GPU runs happen on Kaggle (`kaggle_output/` metrics), not locally. If a step was skipped or failed, say so plainly.
- **Preserve the environment-aware pattern.** Keep `ON_KAGGLE` / `DATA_DIR` / `WORK_DIR` derivation intact; reads via `DATA_DIR`, writes via `WORK_DIR`. Don't hardcode paths.

### Documentation & style
- **Self-documenting code first.** Clear names over comments. Comments explain *why* (the constraint, the non-obvious decision), not *what*. No commented-out zombie code.
- **Docstrings on helper functions/scripts** — inputs, outputs, gotchas. Update docs in `docs/` and this file in the *same* change as the code they describe; stale docs are worse than none.

### ML reproducibility (this is research code)
- **Version code + config together.** Each config change is a distinct experiment.
- **Log inputs to files, not just the terminal** — hyperparameters, chosen settings, metrics, data version. Terminal-only output is lost. The notebooks already write `eval_metrics.json` / `train_metrics.json` / `eval_report.txt`; keep that habit.
- **Record seeds; pin the environment** (`requirements.txt`). Don't overwrite prior run outputs when a fresh comparison matters.
- **Build and document a fair baseline** before claiming a fine-tuning gain (cf. the T1 baseline notebook) so improvements reflect the model, not a pipeline difference.

### Logging & formatting
- `print` is fine for notebook/prototype work; use log **levels** deliberately if a script grows. Never log secrets (HF token). Prefer an auto-formatter over manual style debates.
