# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for fine-tuning LLMs (Llama 3 / Mistral) on legal contract review tasks using QLoRA on consumer hardware. Work is done primarily in Jupyter notebooks.

**Three tasks:**
- **T1 – Risk Clause Recognition:** Binary clause identification (32 Yes/No categories) from CUAD
- **T2 – Structured Entity Extraction:** Extract dates/names/terms as valid JSON (9 categories) from CUAD
- **T3 – Jurisdiction Identification:** Provision classification from LEDGAR (not yet implemented)

## Environment Setup

```powershell
# Activate the local venv
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

The `DATA_DIR` environment variable controls the data root (defaults to `data`). CUAD files are expected at `data/CUAD_v1/`:
- `data/cuad/master_clauses.csv` — primary training source (545 contracts, 41 clause categories)
- `data/cuad/master_clauses_cleaned.csv` — preprocessed version used by the LoRA notebook
- `data/cuad/master_clauses_cleaned_sampled.csv` — small sample for quick iteration
- `data/cuad/CUAD_v1.json` — SQuAD-format JSON (used for EDA only, not for training)
- `data/cuad/train/cuad_train.jsonl` — generated training JSONL
- `data/cuad/validation/cuad_validation.jsonl` — generated validation JSONL

## Notebooks

- [CUAD_dataset_exploration.ipynb](CUAD_dataset_exploration.ipynb) — EDA: loads CSV/JSON, cleans column names, analyzes class imbalance, context lengths, and previews instruction-tuning format
- [llm_fine_tuning_LORA.ipynb](llm_fine_tuning_LORA.ipynb) — Full training pipeline: loads `master_clauses_cleaned.csv`, formats to JSONL, then runs QLoRA fine-tuning via `SFTTrainer`

## Data Format & Schema

`master_clauses.csv` column pairs: `[Category]` (clause text context) + `[Category]-Answer` (ground truth).

- Answer = `"No"` → clause absent (T1 negative)
- Answer = text/date/name → clause present or entity extracted (T1 positive / T2 output)

Training JSONL schema:
```json
{"instruction": "Extract the [Category] from the contract text. Return in JSON format.",
 "input": "{\"[Category]\": \"<clause text>\"}",
 "output": "{\"[Category]\": \"<answer>\"}"}
```

Prompt template used during training:
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

## QLoRA Configuration

Target model: `meta-llama/Meta-Llama-3-8B` (or `mistralai/Mistral-7B-v0.1`)

Key settings in `llm_fine_tuning_LORA.ipynb`:
- 4-bit NF4 quantization via `BitsAndBytesConfig`
- LoRA rank `r=16`, `lora_alpha=16`, targets `["q_proj", "v_proj"]`
- `max_seq_length=2048`, optimizer `paged_adamw_32bit`
- Trained model saved to `./llama-3-cuad-finetune`

## Important Data Decisions

- **Train/val split is contract-level** (not clause-level) at 85/15 to prevent leakage — a contract's clauses must not appear in both sets
- `CUAD_v1.json` is SQuAD extractive format; it is **not used for training** generative models
- `full_contracts_pdf/` and `full_contracts_txt/` are intentionally omitted — raw contracts exceed context windows and the CSV already contains extracted clauses
- Class imbalance is significant for T1 (rare risk clauses <10% frequency); training prompts must include "No" examples explicitly
