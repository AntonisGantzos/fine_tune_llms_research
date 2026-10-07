# Implementation Plan: Adding Saul-7B-Base (and a Mistral-7B Control) to the Legal LLM Study

Each bullet is one change. Steps are ordered; finish each phase before the next.
Paths follow the current repository layout (see `CLAUDE.md`).

**Goal:** run Saul-7B-Base (legal domain-pretrained Mistral-7B) and plain Mistral-7B on the
*exact* validation sets, prompts, decoding and scoring used for the Llama-3.1-8B runs, and
compare all arms in a 3×2 grid: {Llama, Mistral, Saul} × {zero-shot, QLoRA}.

### Revision notes (2026-10-05) — what changed versus the first draft, and why

- `ledgar/` JSONL and `docs/RESULTS_OVERVIEW.md` are already committed; those Phase 0 steps were dropped.
- The local `cuad/` T1 JSONL is **stale** (1,282 train / 448 val) — the Kaggle runs generated
  6,106 / 2,208. The canonical eval data is the JSONL in `kaggle_output_task*_*/`, not the repo copies.
  (Local T2 and T3 JSONL match the Kaggle copies record-for-record.)
- `legal_model_extension.py` does not exist anywhere in the repo; Phase 1 now *writes* it.
- Scoring code lives inside notebooks, which cannot be imported; it is extracted into `scripts/` first.
- No notebook sets double quantisation, a scheduler or warmup — the harness uses the same defaults explicitly.
- The Kaggle CLI only pushes a folder containing a file named `kernel-metadata.json`, so the runner
  gets its own folder `kaggle/extension/` instead of a `kernel-metadata.extension.json` file.
- LEDGAR is **not** used at 60k/10k/10k: the notebooks sample 100/label train (9,801) and 20/label
  validation (1,945). The README fix in Phase 13 was corrected accordingly.
- The T3 trimming reference is confirmed from the run log: 9/1,945 validation provisions trimmed (670-token budget).
- Mistral/Saul use a 32k vocabulary (Llama: 128k), so the logits tensor that drove T3 OOMs is 4× smaller,
  but the instruction token count and provision budget must be recomputed per tokenizer.

### Scope update (2026-10-06)

Saul only, Task 1 first; Mistral arms deferred. Consequence: results rank Saul against Llama but cannot
attribute differences to legal pretraining (see `docs/extension/DESIGN.md`, "Current scope").

### Scope update (2026-10-07) — Mistral arms re-enabled

All Saul arms are done. Mistral now runs the same sequence, one task at a time: T1 baseline → T1 fine-tune →
T2 baseline → T2 fine-tune → T3 baseline → T3 fine-tune, each followed by its paired comparisons. Mistral's
tokenizer is identical to Saul's (same 32k vocabulary and special tokens; identical token ids on every
validation row of all three tasks, checked on CPU), so Saul's trim counts and step timings carry over and the
Phase 6 pipeline smoke tests are not repeated. Finally a notebook compares the three fine-tuned models (Phase 11).

### Is fine-tuning Saul required?

Yes. Saul-7B-Base is a *base* model: domain pretraining improves legal representations, not
adherence to a Yes/No answer, a JSON schema or a closed 100-label set. Zero-shot Llama-base shows
the failure is mostly format: T1 accuracy 0.56, T2 JSON-valid 0.47, T3 valid-label 0.40 (accuracy 0.07),
versus 0.96 / 0.997 / 0.77 after QLoRA. Comparing zero-shot Saul to fine-tuned Llama would confound domain
knowledge with task adaptation. Keep Saul zero-shot as a cheap arm, but the headline comparison
needs Saul-QLoRA versus Mistral-QLoRA.

---

## Phase 0 — Prepare the repository

- [x] Commit the pending edits (`docs/RESEARCH_PAPER.md`, the three `*_finetune_vs_baseline_comparison.ipynb`) on `main`.
- [x] Create a git branch `feature/legal-model-extension` from `main`.
- [x] Overwrite the stale `cuad/train/cuad_train.jsonl` and `cuad/validation/cuad_validation.jsonl` with the
      copies from `kaggle_output_task1_fine_tuned/cuad/` (6,106 / 2,208 rows) and commit them.
- [x] Record SHA-256 checksums of the six canonical train/validation JSONL files in `docs/extension/DESIGN.md`.
      (Hashes are of LF-normalised bytes — `core.autocrlf=true` makes working copies CRLF.)
- [x] Create a `docs/extension/` folder for all design notes belonging to this extension.
- [x] Add `docs/extension/DESIGN.md` describing the 3×2 grid: Llama, Mistral, Saul × zero-shot, QLoRA.

## Phase 1 — Write the shared harness

- [x] Extract T1 scoring (`predict` answer parsing + accuracy / classification report / confusion matrix)
      from `llm_fine_tuning_LORA_task1_v2.ipynb` into `scripts/task1_metrics.py`.
- [x] Extract T2 scoring (`normalize_answer`, `parse_prediction`, `norm`, `token_f1`, `value_f1`)
      from `llm_fine_tuning_LORA_task2.ipynb` into `scripts/task2_metrics.py`.
      (`normalize_answer` is data-building only — the JSONL already holds its output — so it is not needed for scoring.)
- [x] Extract T3 prompt + scoring (`build_prompt`, `truncate_input`, `predict_labels` parsing, `__INVALID__`
      sentinel, macro/micro F1) from `llm_fine_tuning_LORA_task3.ipynb` into `scripts/task3_metrics.py`.
      (`truncate_input` was already ported as the harness's `make_trimmer`; labels come from `--labels_file`.)
- [x] Write `scripts/legal_model_extension.py` with `run` (modes `baseline`, `finetune`, `adapter`) and `compare`
      subcommands, importing only those three modules for prompts and scoring.
      (T1, T2 and T3 registered.)
- [x] Use the notebooks' quantisation exactly: NF4, 4-bit, **no** double quantisation, compute dtype fp16 (T4 has no bf16).
- [x] Use the notebooks' training settings exactly (from `train_metrics.json`): 1 epoch, lr 2e-4, weight decay 0.001,
      `paged_adamw_32bit`, LoRA r=16 / α=16 / dropout 0.05 on q/k/v/o, `completion_only_loss=True`, no packing,
      TRL-default scheduler and warmup.
- [x] Use per-task batching as in the notebooks: T1/T2 batch 1 × accum 8; T3 batch 2 × accum 4 with `group_by_length=True`.
      (All three registered.)
- [x] Use the notebooks' decoding exactly: greedy, `max_new_tokens` 3 (T1) / 128 (T2) / 16 (T3), max length 1024.
      (All three. T3 uses 16 for every arm; see DESIGN.md for why not the fine-tune notebook's 7.)
- [x] Port the T3 safeguards into every arm: `embed_tokens`/`lm_head` recast to fp16 after trainer init,
      `generate()` inside `torch.autocast("cuda", dtype=float16)`, `TimeBudgetCallback` at 7 h (raised to 9 h on 2026-10-06, see Phase 9), pre-flight checks 9 and 10.
      Input-text trimming is applied to every task too — see `docs/extension/DESIGN.md` (Llama T1: 50 train / 6 val affected).
- [x] Set `pad_token = eos_token` when the tokenizer has none (Mistral/Saul ship without one) and log the choice.
- [x] Write `eval_metrics.json`, `train_metrics.json` and per-example `predictions.jsonl` (needed for paired tests) per run.
- [x] Add a `--limit N` argument that evaluates only the first N validation examples, for smoke tests.
      (Also `--train_limit N` for the Phase 6 200-example fine-tune smoke test.)
- [x] Add a CPU-only unit test `scripts/test_extension_scoring.py` covering valid, invalid and edge-case completions.
      (T1, T2, T3 + `compare` for each; 18 tests pass.)
- [x] Verify locally on CPU that the extracted scorers reproduce the existing `eval_metrics.json` numbers when fed the
      fine-tuned runs' saved predictions (if not saved, check against a hand-built fixture instead).
      T1: predictions were not saved; fixtures rebuilt from both saved confusion matrices reproduce both files exactly.
      T2: no per-example data saved; n and per-category counts checked instead. T3: rebuilt from saved per-label
      P/R/support; both runs reproduce exactly.

## Phase 2 — Pin the environment

- [x] Remove the duplicated `pandas` and `scikit-learn` entries from `requirements.txt`.
- [x] Add `scipy` explicitly to `requirements.txt` (already a scikit-learn dependency) for the McNemar test in `compare`.
- [x] Pin `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`, `accelerate` versions using `pip freeze` from a Kaggle session.
      `datasets==4.8.5` pinned from the first smoke run's `pip_freeze.txt` (2026-10-05).
      Partly done: the five versions the Llama notebooks install are pinned in `requirements-kaggle.txt` (not in the
      local CPU `requirements.txt` — transformers 4.55 would downgrade the local Python 3.14 env). `datasets` is pinned
      from the `pip_freeze.txt` the runner writes on the first Phase 6 smoke run.
- [x] Add `requirements-kaggle.txt` holding only the pinned GPU-side packages the runner notebook installs.

## Phase 3 — Model access

- [x] Accept the Hugging Face terms for `mistralai/Mistral-7B-v0.1` with the account behind the `hf-token` dataset.
      Not needed: the repo is no longer gated (verified via the HF API; config + tokenizer download with the token).
- [x] Confirm `Equall/Saul-7B-Base` downloads with the same token from a local `huggingface_hub` call (config only, no weights).
- [x] Add `Mistral-7B` and `Saul-7B-Base` licence notes (Apache-2.0, MIT) to `docs/extension/DESIGN.md`.
      (Also recorded: Saul ships 29 GB of fp32 weights — keep the HF cache off `/kaggle/working`.)

## Phase 4 — Package inputs for Kaggle

- [ ] Create a private Kaggle dataset `legal-extension-code` containing `scripts/` and `requirements-kaggle.txt`.
      Payload staged + verified (`kaggle/dataset_payload_ext_code/`: harness, `task*_metrics.py`, requirements);
      upload pending — run `push_extension_datasets.ps1`.
- [ ] Create a private Kaggle dataset `legal-extension-data` containing the six canonical JSONL files plus
      `data/LEDGAR/labels.json`, staged flat (no `--dir-mode zip`, like `ledgar-lexglue`).
      Payload staged; all six files match the DESIGN.md SHA-256 table. Upload pending.
- [ ] Create a private Kaggle dataset `llama-adapters` from `kaggle_output_task{1,2,3}_fine_tuned/llama-3.1-8B-*` (adapter files only, no `results/` checkpoints).
      Payload staged flat as `taskN__adapter_config.json` / `taskN__adapter_model.safetensors` (the CLI skips or
      flattens subfolders); the runner rebuilds the adapter folder in `/tmp`. Upload pending.
- [x] Add a `kaggle/push_extension_datasets.ps1` script that versions all three datasets with one command,
      calling the CLI as `python -m kaggle`. (Creates a dataset on first use, versions it afterwards; `-StageOnly`, `-Only`.)

## Phase 5 — Build the runner notebook

- [x] Create `legal_model_extension_runner.ipynb` with the `ON_KAGGLE` / `DATA_DIR` / `WORK_DIR` pattern of the existing notebooks.
- [x] Add a cell setting `CUDA_VISIBLE_DEVICES=0` to pin the run to one T4.
- [x] Add a cell installing `requirements-kaggle.txt` from the mounted `legal-extension-code` dataset.
      (Also writes `pip_freeze.txt`, which closes the Phase 2 pinning step after the first smoke run.)
- [x] Add a cell reading the HF token with the existing `get_hf_token()` (`hf-token` dataset → Secrets → `.env`); never print it.
- [x] Add a parameters cell defining `TASK`, `MODEL`, `MODE`, and `ADAPTER` as the only values edited per run.
      (Plus `LIMIT` / `TRAIN_LIMIT` for smoke tests and `MAX_SEQ_LEN`.)
- [x] Add a cell mapping `TASK` to the correct train JSONL, validation JSONL and labels paths under `/kaggle/input/`.
      (Files are found by name under `/kaggle/input`, so a change in Kaggle's mount layout cannot break it.)
- [x] Add a cell invoking `scripts/legal_model_extension.py run` with the parameters, writing to `/kaggle/working/runs/`.
- [x] Add a final cell copying the run log into the output directory for retrieval.
      (The launch cell tees the harness output into `runs/<run>/run.log`; the final cell adds `pip_freeze.txt` and
      fails the kernel if the harness failed.) Executed locally end to end: stops at the expected "CUDA not available".
- [x] Add `kaggle/extension/kernel-metadata.json` (new kernel `id`, `enable_internet: true`, GPU T4) attaching the code,
      data, adapter and `hf-token` datasets; `code_file` points at `../../legal_model_extension_runner.ipynb`.
- [x] Add a `-KernelDir` parameter to `kaggle/run.ps1` (default `kaggle/`) so it can push `kaggle/extension/`.
- [x] Make `kaggle/run.ps1` download extension runs into `kaggle_output_extension/<model>_t<task>_<mode>/` and git-ignore that folder.
      (`-OutDir kaggle_output_extension`; each run lands in `kaggle_output_extension/runs/<model>_t<task>_<mode>[_smoke]/`.)
- [ ] Set accelerator GPU T4×2 once for the new kernel on kaggle.com; never save from the web editor (it empties `dataset_sources`).

## Phase 6 — Smoke tests (cheap, run before any long job)

- [x] Run the scoring unit tests locally on CPU and fix any failure before using GPU time. (7/7 pass.)
- [x] Run `saul` task 3 baseline with `--limit 40` on Kaggle to validate loading, generation and scoring.
      Task-1-first order: run `saul` **task 1** baseline with `LIMIT = 40` first (T3 is not registered in the harness yet).
      T1 version done 2026-10-05: loading, generation and scoring all OK (see DESIGN.md run log). T3: the full
      zero-shot baseline is run directly instead (no training, so it is no more expensive than a smoke test).
- [x] Run `saul` task 1 finetune on a 200-example train subset to validate training and adapter reload.
      Done 2026-10-05: pre-flight 9/10 pass, training 13.5 s/step (full epoch ≈ 2.9 h), adapter saved, eval OK
      (0.905 on 200). Eval used the in-memory adapter; reload-from-disk (`adapter` mode) is first exercised in Phase 8.
- [x] Log the per-tokenizer instruction token count and provision budget for T3 (Llama: 331-token instruction, 670-token budget).
      Measured on CPU: Saul 406-token instruction, 584-token budget (DESIGN.md).
- [x] Compare the smoke-run validation trimming count for Saul task 3 with Llama's logged 9 of 1,945.
      CPU count with the real tokenizer: Saul 32 / 1,945 val, 210 / 9,801 train (Llama 9 and 79, reproduced exactly).
- [x] If Saul trims far more than Llama, set `--max_seq_len 1536` for every *new* arm, confirm pre-flight check 9 passes, and record why.
      Not applied: 1.6 % of validation trimmed; 1024 kept for comparability and the 7 h budget (DESIGN.md).

## Phase 7 — Zero-shot baselines (about one hour each)

- [ ] Run `llama` baseline for task 1 through the harness; it should reproduce accuracy 0.561 on 2,208 examples.
- [ ] Run `llama` baseline for task 2 through the harness; it should reproduce JSON-valid 0.472 / EM 0.090 / F1 0.141 on 631.
- [ ] Run `llama` baseline for task 3 through the harness, closing the old bf16 and prompt-truncation asymmetries
      of `llama_3.1_task_3_no_fine_tune.ipynb` (old: accuracy 0.065, macro-F1 0.066).
- [ ] Run `mistral` baseline for task 1.
- [ ] Run `mistral` baseline for task 2.
- [ ] Run `mistral` baseline for task 3.
- [x] Run `saul` baseline for task 1. (2026-10-06: answers "No" to all 2,208 → accuracy 0.747 = majority rate,
      macro-F1 0.43; see DESIGN.md run log.)
- [x] Run `saul` baseline for task 2.
      (2026-10-06: JSON-valid 0.097, EM 0.000, F1 0.069 — mostly bare arrays and runaway generation; see DESIGN.md.)
- [x] Run `saul` baseline for task 3.
      Done 2026-10-06: valid-label 0.976, accuracy 0.059, macro-F1 0.043 (collapses onto "No Defaults").
- [ ] Optionally run `saul-instruct` with `--chat` on each task, reported separately as non-matched prompts.

## Phase 8 — Re-score the existing Llama adapters

- [x] Run `llama` in `adapter` mode for task 1 with `llama-3.1-8B-cuad-task1` (paper: accuracy 0.962).
      (2026-10-06: 0.9620 vs notebook 0.9615 — one example; adapter reload from disk verified.)
- [x] Run `llama` in `adapter` mode for task 2 with `llama-3.1-8B-cuad-task2` (paper: EM 0.691, F1 0.819).
      (2026-10-06: EM 0.6910 identical, F1 0.818.)
- [x] Run `llama` in `adapter` mode for task 3 with `llama-3.1-8B-ledgar-task3` (paper: accuracy 0.769, macro-F1 0.751).
      (2026-10-07: accuracy 0.7676, macro-F1 0.7498; valid-label rate identical.)
- [x] Record in `docs/extension/DESIGN.md` any gap between these scores and the paper's in-process scores
      (T1/T2 Llama were trained and evaluated in bf16; the harness evaluates in fp16).
      T1 recorded (+0.0005 accuracy, i.e. one example); T2 recorded (EM identical, F1 −0.002); T3 recorded (−2 examples, macro-F1 −0.001).

## Phase 9 — QLoRA fine-tunes (one Kaggle session each)

- [ ] Fine-tune `mistral` on task 2 first, as the shortest run, to confirm timing estimates.
      Superseded (2026-10-07): Mistral follows Saul's task order (T1, T2, T3); timing is known from Saul's
      identical-tokenizer runs (T1 3.06 h, T2 0.99 h, T3 5.79 h).
- [x] Fine-tune `saul` on task 2.
      (2026-10-06: 0.99 h; JSON-valid 0.995, EM 0.724, F1 0.853.)
- [ ] Fine-tune `mistral` on task 1.
- [x] Fine-tune `saul` on task 1. (2026-10-06: 3.06 h, 1 full epoch; accuracy 0.968, macro-F1 0.959.)
- [ ] Fine-tune `mistral` on task 3 with the 7-hour time-budget callback (Llama T3 needed 6.1 h of it).
      (Budget is now 9 h, as for Saul T3.)
- [x] Fine-tune `saul` on task 3 with the 7-hour time-budget callback.
      Smoke (200 train / 200 val, 2026-10-06): 20.9 s/step → full epoch (1,226 steps) ≈ 7.25 h, over 7 h. The harness
      budget is raised to 9 h (≈ 0.8 h eval + setup still leaves ~2 h of Kaggle's 12 h), so the epoch completes.
      Full run 2026-10-07: 5.79 h, 1 full epoch; accuracy 0.766, macro-F1 0.748.
- [ ] Reject and re-run any run whose `train_metrics.json` shows `stopped_on_time_budget: true`.

## Phase 10 — Paired comparisons

- [ ] Run `compare` Saul-baseline versus Mistral-baseline for each task, measuring legal pretraining without fine-tuning.
- [ ] Run `compare` Saul-finetune versus Mistral-finetune for each task, the headline legal-pretraining comparison.
- [x] Run `compare` Saul-finetune versus Llama-adapter for each task, the best-model comparison.
      T1 done: 27 vs 40 discordant, McNemar p = 0.142, macro-F1 diff +0.007 [−0.002, +0.016] → no significant difference.
      T2 done: 19 vs 40, p = 0.0086, F1 diff +0.036 [+0.017, +0.056] → Saul better, but only on the two date
      categories, whose labels are punctuation-stripped digit strings (see DESIGN.md); other categories p = 0.87.
      T3 done: 104 vs 100, p = 0.83, macro-F1 diff −0.002 [−0.016, +0.012] → no significant difference.
- [x] Run `compare` Saul-finetune versus Saul-baseline for each task, matching the paper's existing delta.
      T1 done: p = 2.4e-99, macro-F1 +0.531 [+0.518, +0.543].
      T2 done: 0 vs 457, p = 5.4e-138, F1 +0.784 [+0.753, +0.814].
      T3 done: 19 vs 1,393, p < 1e-300, macro-F1 +0.705 [+0.682, +0.720].
- [x] Use McNemar on per-example correctness (T1 accuracy, T2 exact match, T3 accuracy) and bootstrap 95% CIs for F1 metrics.
      (Implemented in `compare`: exact binomial McNemar + paired bootstrap; T1 uses accuracy + macro-F1.)
- [ ] Add `scripts/collect_extension_results.py` merging every `eval_metrics.json` and `compare_*.json` into one CSV.

## Phase 11 — Analysis notebook

- [ ] Create `extension_comparison.ipynb` loading the merged CSV from `kaggle_output_extension/`.
- [ ] Add a section comparing the evaluation results of the three fine-tuned models (Llama-adapter, Mistral-finetune,
      Saul-finetune) on every task: headline metrics with 95% CIs, the three pairwise McNemar / bootstrap tests,
      and per-category (T1/T2) and per-label (T3) breakdowns.
- [ ] Add a headline table per task: six arms, validity gate, headline metric, 95% CI.
- [ ] Add a validity-versus-content chart separating format gains from content gains for tasks 2 and 3.
- [ ] Add a task-3 per-label F1 comparison of Saul-finetune versus Mistral-finetune.
- [ ] Add a task-2 per-category table excluding `Warranty Duration` (n=11) from the overall summary.
- [ ] Add a task-1 table comparing lenient accuracy with the new strict-validity rate for every arm.

## Phase 12 — Contamination check

- [x] Read the SaulLM-7B corpus section and record whether SEC EDGAR data was in pretraining.
      Yes: Table 1 of arXiv:2403.03883 lists EDGAR (about 5B tokens); no benchmark decontamination reported.
- [x] If EDGAR was included, add a contamination caveat to every Saul result — both CUAD and LEDGAR are EDGAR-sourced.
      Caveat written in `docs/extension/DESIGN.md`; it must be repeated beside Saul results in the paper (Phase 13).

## Phase 13 — Documentation and paper

- [ ] Update the `README.md` and `CLAUDE.md` task-status tables: T1–T3 marked done, extension marked in progress.
- [ ] Fix the contract count from 545 to 510 in `README.md` and `CLAUDE.md`, matching `master_clauses.csv`.
- [ ] Fix the `README.md` LEDGAR description: LexGLUE 100-label config, stratified 100/label train (9,801) and 20/label validation (1,945).
- [ ] Add the extension notebooks, scripts and output folder to the `README.md` layout and the `CLAUDE.md` notebook list.
- [ ] Add an extension section to `docs/kaggle/kaggle_connection_guide.md` covering the three new datasets and `kaggle/extension/`.
- [ ] Add a methodology subsection to `docs/RESEARCH_PAPER.md` describing the 3×2 design and the Mistral control.
- [ ] Add an extension results subsection to `docs/RESEARCH_PAPER.md` with tables generated from the merged CSV.
- [ ] Add the per-tokenizer trimming counts, the bf16-vs-fp16 Llama caveat and any `max_seq_len` change to the limitations section.
- [ ] Add Mistral 7B (Jiang et al., 2023) and SaulLM-7B (Colombo et al., 2024) to the paper's references.
- [ ] Add every new artifact path to the paper's Appendix A provenance table.
- [ ] Open a pull request from `feature/legal-model-extension` once all phases are complete.
