# Legal Domain LLM Fine-Tuning Research

## Project Overview
This research focuses on fine-tuning Large Language Models (LLMs) for the **Legal Domain**. The primary objective is to adapt a general-purpose model (Llama 3.1 8B) to recognize, classify, and extract specific elements from complex legal documents with high precision — using QLoRA so the whole pipeline runs on free/consumer-grade hardware (Kaggle T4 GPUs).

## Problem Statement
Legal documents require a degree of domain comprehension that general LLMs often lack. The specific challenges addressed in this research include:
*   **Unstructured Data:** Legal contracts contain critical data (dates, parties, clauses) buried in dense text.
*   **Strict Formatting Requirements:** Downstream systems often require output in valid JSON, which standard models often fail to produce consistently.
*   **Hallucinations:** In a legal context, factual accuracy is paramount. A model cannot invent clauses or dates.
*   **Hardware Constraints:** Fine-tuning large models typically requires enterprise-grade hardware (A100s). This research makes the process work on Kaggle's free T4×2 GPUs, driven entirely from a local machine with no GPU.

## Core Tasks & Current Status

| Task | Goal | Dataset | Status |
| :--- | :--- | :--- | :--- |
| **T1 – Risk Clause Recognition** | Binary Yes/No: "Is this text a *[category]* clause?" across all CUAD risk-clause categories | CUAD | ✅ **Done** — trained, evaluated, and compared against the un-fine-tuned baseline |
| **T2 – Structured Entity Extraction** | Extract 9 entity types (Parties, Agreement/Effective/Expiration Date, Governing Law, Renewal Term, etc.) as strict single-key JSON | CUAD | 🔄 **In progress** — pipeline notebook and train/validation JSONL ready; training run pending |
| **T3 – Jurisdiction & Governing Law Identification** | Classify the provision type / governing law of a contract clause | LEDGAR | ⬜ Not started |

### T1 Results (Llama-3.1-8B, 2,208 validation examples)

| Metric | Fine-tuned (QLoRA) | Base model (no fine-tune) |
| :--- | :--- | :--- |
| Accuracy | **96.2%** | 56.1% |
| Macro F1 | **0.951** | 0.535 |
| "Yes" F1 (rare class) | **0.928** | 0.426 |

Fine-tuning details and per-class breakdowns: [finetune_vs_baseline_comparison.ipynb](finetune_vs_baseline_comparison.ipynb) and `kaggle_output/eval_report.txt`.

## Repository Layout

```
├── CUAD_dataset_exploration.ipynb        # EDA: class imbalance, context lengths, format preview
├── llm_fine_tuning_LORA_task1_v2.ipynb   # T1: CSV → JSONL → QLoRA fine-tune → eval
├── llm_fine_tuning_LORA_task2.ipynb      # T2: same pipeline for entity extraction
├── llama_3.1_task_1_no_fine_tune.ipynb   # T1 baseline: base model on the same validation set
├── kaggle_results_visualization.ipynb    # Plots for a single training-run directory
├── finetune_vs_baseline_comparison.ipynb # T1 fine-tuned vs. baseline comparison
├── cuad/                                 # Generated train/validation JSONL (T1 + T2)
├── data/CUAD_v1/                         # Raw CUAD data (git-ignored; see scripts/download_cuad.py)
├── kaggle/                               # Remote-GPU runner: push kernel, pull outputs
├── kaggle_output/                        # Downloaded run artifacts: adapter, metrics, logs (git-ignored)
├── scripts/                              # download_cuad.py, preprocess_values_cuad.py, sampling
└── docs/                                 # Design docs per task + data processing + Kaggle workflow
```

## Workflow: Local Editing, Remote GPU

The local machine has no GPU, so training runs on **Kaggle's free T4×2** as batch kernels:

1. Edit a notebook locally in VS Code (data-prep cells run fine on CPU).
2. `.\kaggle\run.ps1 -Wait` — pushes the notebook, Kaggle runs it top-to-bottom, and the trained adapter + metrics download to `kaggle_output/`.
3. Analyze results locally with the visualization/comparison notebooks.

Notebooks are environment-aware (`ON_KAGGLE` flag) so the same file runs in both places. Setup and details: [kaggle/README.md](kaggle/README.md) and [docs/kaggle/pipeline_state_and_kaggle_interaction.md](docs/kaggle/pipeline_state_and_kaggle_interaction.md).

## Datasets

### 1. CUAD (Contract Understanding Atticus Dataset) — Tasks 1 & 2
*   **Description:** An expert-annotated NLP dataset for legal contract review: 545 contracts annotated across 41 clause categories (`master_clauses.csv`).
*   **How it is used here:** each `[Category]` / `[Category]-Answer` column pair becomes instruction-tuning examples. A cleaned CSV (`master_clauses_cleaned.csv`) replaces empty-negative placeholders with real non-related clause text so T1 sees hard "No" examples. The train/validation split is **contract-level (85/15)** to prevent leakage.
*   **Reference:** *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review* (NeurIPS 2021) by The Atticus Project.

### 2. LEDGAR (Labeled EDGAR) — Task 3 (planned)
*   **Description:** A large-scale multi-label corpus of ~850,000 contract provisions scraped from SEC filings, labeled with over 12,000 categories.
*   **Reference:** *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts* (LREC 2020) by Tuggener et al.

## Methodology & Techniques

*   **QLoRA:** the base model is loaded in 4-bit NF4 quantization (`BitsAndBytesConfig`) and adapted with LoRA (`r=16`, `lora_alpha=16`, targets `q/k/v/o_proj`) — memory for Llama-3.1-8B drops to fit a T4, and only the small adapter is trained and saved.
*   **Model:** `meta-llama/Meta-Llama-3.1-8B` for production runs; `meta-llama/Llama-3.2-1B` as a fast smoke-test in the same notebooks.
*   **Prompt Engineering:** instructions are baked into the training data using an `### Instruction / ### Input / ### Response` template; T1 constrains output to strict `Yes`/`No`, T2 to a single-key JSON object.

## Evaluation Metrics

*   **Accuracy / F1 / Precision / Recall** per class for T1 (macro-F1 is the headline number given the Yes/No imbalance), plus a confusion matrix.
*   **JSON Validity Score** for T2 — percentage of generations that parse as valid JSON with the expected single key.
*   **Exact Match (EM)** for rigid entities (dates, names) and **F1** for longer extractions where partial credit applies.

## Getting Started

```powershell
.\env\Scripts\Activate.ps1      # activate the local venv
pip install -r requirements.txt
python scripts\download_cuad.py # fetch CUAD into DATA_DIR (set DATA_DIR=data/CUAD_v1 in .env first)
jupyter notebook
```

A Hugging Face token with access to the gated Llama models is required (`.env` locally, `HF_TOKEN` secret on Kaggle). Development conventions live in [CLAUDE.md](CLAUDE.md).
