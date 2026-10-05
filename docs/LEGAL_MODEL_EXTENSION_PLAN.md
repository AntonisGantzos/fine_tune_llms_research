# Implementation Plan: Adding Saul-7B-Base (and a Mistral-7B Control) to the Legal LLM Study

Each bullet is one change. Steps are ordered; finish each phase before the next.
Paths follow the repository layout in `README.md`.

---

## Phase 0 — Prepare the repository

- [ ] Create a git branch `feature/legal-model-extension` from the current main branch.
- [ ] Commit the untracked `ledgar/` JSONL splits so the T3 data version is versioned.
- [ ] Commit the untracked `docs/RESULTS_OVERVIEW.md` before any new results are added.
- [ ] Create a `docs/extension/` folder for all design notes belonging to this extension.
- [ ] Add `docs/extension/DESIGN.md` describing the 3×2 grid: Llama, Mistral, Saul × zero-shot, QLoRA.

## Phase 1 — Add the shared harness

- [ ] Copy `legal_model_extension.py` into `scripts/` as the single code path for every new run.
- [ ] Replace its `score_t1` with an import of the scoring function from `llm_fine_tuning_LORA_task1_v2.ipynb`.
- [ ] Replace its `score_t2` with the task-2 notebook's scoring functions, extracted into `scripts/task2_metrics.py`.
- [ ] Replace its `score_t3` with the task-3 notebook's scoring functions, extracted into `scripts/task3_metrics.py`.
- [ ] Set the harness `BitsAndBytesConfig` double-quantisation flag to match the notebooks' exact setting.
- [ ] Copy any scheduler and warmup `SFTConfig` fields from the notebooks into the harness `finetune()` function.
- [ ] Add a `--limit N` argument that evaluates only the first N validation examples, for smoke tests.
- [ ] Add a CPU-only unit test `scripts/test_extension_scoring.py` covering valid, invalid and edge-case completions.

## Phase 2 — Pin the environment

- [ ] Remove the duplicated `pandas` and `scikit-learn` entries from `requirements.txt`.
- [ ] Add `scipy` to `requirements.txt` for the McNemar test in the `compare` command.
- [ ] Pin `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`, `accelerate` versions using `pip freeze` from a Kaggle session.
- [ ] Add `requirements-kaggle.txt` holding only the pinned GPU-side packages the runner notebook installs.

## Phase 3 — Model access

- [ ] Accept the Hugging Face terms for `mistralai/Mistral-7B-v0.1` with the account behind the `hf-token` dataset.
- [ ] Confirm `Equall/Saul-7B-Base` downloads with the same token from a local `huggingface_hub` call.
- [ ] Add `Mistral-7B` and `Saul-7B-Base` licence notes (Apache-2.0, MIT) to `docs/extension/DESIGN.md`.

## Phase 4 — Package inputs for Kaggle

- [ ] Create a private Kaggle dataset `legal-extension-code` containing `scripts/` and `requirements-kaggle.txt`.
- [ ] Create a private Kaggle dataset `legal-extension-data` containing `cuad/` and `ledgar/` JSONL plus `data/LEDGAR/labels.json`.
- [ ] Create a private Kaggle dataset `llama-adapters` containing the three saved `llama-3.1-8B-*` adapter folders.
- [ ] Add a `kaggle/push_extension_datasets.ps1` script that versions all three datasets with one command.

## Phase 5 — Build the runner notebook

- [ ] Create `legal_model_extension_runner.ipynb` with an `ON_KAGGLE` flag matching the existing notebooks' convention.
- [ ] Add a cell setting `CUDA_VISIBLE_DEVICES=0` to pin the run to one T4.
- [ ] Add a cell installing `requirements-kaggle.txt` from the mounted `legal-extension-code` dataset.
- [ ] Add a cell reading the HF token from the private `hf-token` dataset and logging in.
- [ ] Add a parameters cell defining `TASK`, `MODEL`, `MODE`, and `ADAPTER` as the only values edited per run.
- [ ] Add a cell mapping `TASK` to the correct train JSONL, validation JSONL and labels paths.
- [ ] Add a cell invoking `scripts/legal_model_extension.py run` with the parameters, writing to `/kaggle/working/runs/`.
- [ ] Add a final cell copying the run log into the output directory for retrieval.
- [ ] Add a `kaggle/kernel-metadata.extension.json` attaching the code, data, adapter and token datasets to the runner.
- [ ] Add a `-Metadata` parameter to `kaggle/run.ps1` so it can push the extension metadata file.
- [ ] Make `kaggle/run.ps1` download outputs into `kaggle_output_extension/<model>_t<task>_<mode>/` for extension runs.

## Phase 6 — Smoke tests (cheap, run before any long job)

- [ ] Run the scoring unit tests locally on CPU and fix any failure before using GPU time.
- [ ] Run `saul` task 3 baseline with `--limit 40` on Kaggle to validate loading, generation and scoring.
- [ ] Run `saul` task 1 finetune on a 200-example train subset to validate training and adapter reload.
- [ ] Check the smoke-run `val_inputs_trimmed` count for Saul task 3 against Llama's logged 9 of 1,945.
- [ ] If Saul trims far more than Llama, set `--max_seq_len 1536` for every arm and record why.

## Phase 7 — Zero-shot baselines (about one hour each)

- [ ] Run `llama` baseline for task 1 through the harness to establish the single-code-path reference.
- [ ] Run `llama` baseline for task 2 through the harness.
- [ ] Run `llama` baseline for task 3 through the harness, closing the old fp16/bf16 and truncation asymmetries.
- [ ] Run `mistral` baseline for task 1.
- [ ] Run `mistral` baseline for task 2.
- [ ] Run `mistral` baseline for task 3.
- [ ] Run `saul` baseline for task 1.
- [ ] Run `saul` baseline for task 2.
- [ ] Run `saul` baseline for task 3.
- [ ] Optionally run `saul-instruct` with `--chat` on each task, reported separately as non-matched prompts.

## Phase 8 — Re-score the existing Llama adapters

- [ ] Run `llama` in `adapter` mode for task 1 using the saved `llama-3.1-8B-cuad-task1` adapter.
- [ ] Run `llama` in `adapter` mode for task 2 using the saved `llama-3.1-8B-cuad-task2` adapter.
- [ ] Run `llama` in `adapter` mode for task 3 using the saved `llama-3.1-8B-ledgar-task3` adapter.
- [ ] Record in `docs/extension/DESIGN.md` any gap between these scores and the paper's in-process scores.

## Phase 9 — QLoRA fine-tunes (one Kaggle session each)

- [ ] Fine-tune `mistral` on task 2 first, as the shortest run, to confirm timing estimates.
- [ ] Fine-tune `saul` on task 2.
- [ ] Fine-tune `mistral` on task 1.
- [ ] Fine-tune `saul` on task 1.
- [ ] Fine-tune `mistral` on task 3, keeping the 7-hour time-budget callback enabled.
- [ ] Fine-tune `saul` on task 3, keeping the 7-hour time-budget callback enabled.
- [ ] Reject and re-run any run whose `train_metrics.json` shows `stopped_on_time_budget: true`.

## Phase 10 — Paired comparisons

- [ ] Run `compare` Saul-baseline versus Mistral-baseline for each task, measuring legal pretraining without fine-tuning.
- [ ] Run `compare` Saul-finetune versus Mistral-finetune for each task, the headline legal-pretraining comparison.
- [ ] Run `compare` Saul-finetune versus Llama-adapter for each task, the best-model comparison.
- [ ] Run `compare` Saul-finetune versus Saul-baseline for each task, matching the paper's existing delta.
- [ ] Add `scripts/collect_extension_results.py` merging every `eval_metrics.json` and `compare_*.json` into one CSV.

## Phase 11 — Analysis notebook

- [ ] Create `extension_comparison.ipynb` loading the merged CSV from `kaggle_output_extension/`.
- [ ] Add a headline table per task: six arms, validity gate, headline metric, 95% CI.
- [ ] Add a validity-versus-content chart separating format gains from content gains for tasks 2 and 3.
- [ ] Add a task-3 per-label F1 comparison of Saul-finetune versus Mistral-finetune.
- [ ] Add a task-2 per-category table excluding `Warranty Duration` from the overall summary.
- [ ] Add a task-1 table comparing lenient accuracy with the new strict-validity rate for every arm.

## Phase 12 — Contamination check

- [ ] Read the SaulLM-7B corpus section and record whether SEC EDGAR data was in pretraining.
- [ ] If EDGAR was included, add a contamination caveat to every Saul result in the analysis notebook.

## Phase 13 — Documentation and paper

- [ ] Update the `README.md` task-status table: T2 and T3 marked done, extension marked in progress.
- [ ] Fix `README.md` contract count from 545 to 510, matching `master_clauses.csv`.
- [ ] Fix `README.md` LEDGAR description to the LexGLUE 100-label, 60k/10k/10k configuration actually used.
- [ ] Add the extension notebooks, scripts and output folder to the `README.md` repository layout.
- [ ] Add an extension section to `docs/kaggle/kaggle_connection_guide.md` covering the three new datasets.
- [ ] Add a methodology subsection to `RESEARCH_PAPER.md` describing the 3×2 design and the Mistral control.
- [ ] Add an extension results subsection to `RESEARCH_PAPER.md` with tables generated from the merged CSV.
- [ ] Add the per-tokenizer trimming counts and any `max_seq_len` change to the paper's limitations section.
- [ ] Add Mistral 7B (Jiang et al., 2023) and SaulLM-7B to the paper's references.
- [ ] Add every new artifact path to the paper's Appendix A provenance table.
- [ ] Open a pull request from `feature/legal-model-extension` once all phases are complete.
