# Task 3 (LEDGAR) — Progress So Far & Next Steps

Snapshot of where Task 3 stands against [TASK3_PLAN.md](TASK3_PLAN.md), and the
ordered list of what remains until the task is complete (trained, evaluated,
compared against baseline, documented).

Status date: 2026-07-17.

---

## Where we are: plan steps 1–3 done, step 4 half-started

| Plan step | Status | Evidence |
| :--- | :--- | :--- |
| 1. Download script | ✅ Done | `scripts/download_ledgar.py`; `data/LEDGAR/` holds `labels.json` + `ledgar_train.csv` / `ledgar_validation.csv` / `ledgar_test.csv` |
| 2. EDA notebook | ✅ Done | `LEDGAR_dataset_exploration.ipynb`; findings written up in [TASK3_EDA_FINDINGS.md](TASK3_EDA_FINDINGS.md) (all four questions answered) |
| 3. Preprocess → JSONL | ✅ Done (locally) | Data cells 3.0–3.6 of `llm_fine_tuning_LORA_task3.ipynb`; generated `ledgar/train/ledgar_task3_train.jsonl` (9,801 rows) + `ledgar/validation/ledgar_task3_validation.jsonl` (1,945 rows) |
| 3b. Stage data to Kaggle | ❌ Not done | `kaggle/stage_data.ps1` still stages only the CUAD CSVs; no LEDGAR Kaggle dataset exists yet |
| 4. Training notebook | 🟡 Half done | Notebook exists (16 cells) but **stops at "save JSONL"** — no model loading, QLoRA config, pre-flight checks, training, or adapter-save cells yet |
| 5. Evaluation cells | ❌ Not done | No eval code anywhere for T3 |
| 6. Baseline + comparison | ❌ Not done | No `llama_3.1_task_3_no_fine_tune.ipynb`, no `task3_finetune_vs_baseline_comparison.ipynb` |
| 7. Documentation | 🟡 Partial | `TASK3_PLAN.md` + `TASK3_EDA_FINDINGS.md` exist; `TASK3_LOGIC.md`, `TASK3_NOTEBOOK_CELL_GUIDE.md`, `TASK3_EVALUATION_METRICS.md` missing; README/CLAUDE.md not yet updated |

Decisions already locked in (don't re-litigate):

- **Stratified sample sizes:** ~100/label train (9,801 after rare-label caps),
  ~20/label validation (1,945). Within the plan's 6k–10k / 1k–2k targets.
- **`Books` validation gap → Option A** from the EDA doc: the sanity-check cell
  requires full 100-label coverage in *train only*
  (`require_full_coverage=False` for validation). Official LexGLUE splits kept.
- **Instruction:** full 100-label menu (~331 tokens), `max_length=1024`,
  0.47% truncation accepted.
- The notebook's Kaggle config cell already expects the LEDGAR dataset to mount
  at **`/kaggle/input/ledgar-lexglue`** — the Kaggle dataset slug must match.
- `kaggle/kernel-metadata.json` already points `code_file` at the T3 notebook
  (kernel id `antonisgantzos/cuad-task3-finetune`), but `dataset_sources` still
  lists only the CUAD dataset — must be updated in step 1 below.

---

## Next steps, in order

### 1. Stage LEDGAR to Kaggle (blocker for everything remote)

The notebook builds the JSONL *on Kaggle* from the CSVs (cell 3.1 reads
`labels.json` + `ledgar_train.csv` + `ledgar_validation.csv` from `DATA_DIR`),
so stage the **raw split files**, not the generated JSONL:

1. Create a new Kaggle dataset whose mounted folder resolves to
   `/kaggle/input/ledgar-lexglue` (i.e. dataset slug `ledgar-lexglue`, owner
   `antonisgantzos`). Mirror the CUAD flow: a small staging script (extend
   `kaggle/stage_data.ps1` or add `kaggle/stage_ledgar.ps1`) plus a
   `dataset-metadata.json`, then `python -m kaggle datasets create` /
   `datasets version`. Payload: `labels.json`, `ledgar_train.csv`,
   `ledgar_validation.csv` (~test CSV optional, unused for now).
2. Watch the known `--dir-mode zip` flattening quirk — either stage the files
   flat (no subfolder) so `DATA_DIR / "labels.json"` resolves directly, or add
   the same fallback-search the CUAD notebooks use.
3. Add the new slug to `dataset_sources` in `kaggle/kernel-metadata.json`
   (alongside or replacing the CUAD source for this kernel).

### 2. Finish the training notebook (plan step 4)

Port the training half of `llm_fine_tuning_LORA_task2.ipynb` (cells 24–30
there) into `llm_fine_tuning_LORA_task3.ipynb`, changing as little as possible:

- The GPU-pin cell (`CUDA_VISIBLE_DEVICES` before any torch import) and UTF-8
  check cell from the top of the T2 notebook — T3 currently lacks both.
- HF-token diagnostic cell (Kaggle secret `HF_TOKEN`, gated-repo access check).
- Model + tokenizer load with the shared QLoRA config: 4-bit NF4, r=16,
  alpha=16, targets `q/k/v/o_proj`, `paged_adamw_32bit`,
  `completion_only_loss=True`, same `### Instruction / ### Input / ### Response`
  template. Keep the `meta-llama/Llama-3.2-1B` smoke-test switch.
- **Pre-flight checks cell**, with the T2 "completion parses as JSON" assert
  swapped for "completion is exactly one of the 100 labels".
- Train + save adapter to `WORK_DIR / "llama-3.1-8B-ledgar-task3"`.

### 3. Add the evaluation cells (plan step 5)

Same notebook, after training — generate on the 1,945-row validation JSONL and
compute, in this order:

1. **Valid-label rate** — exact match against the 100-label list after
   strip/lowercase (T3's analogue of T2's JSON-validity).
2. **Accuracy**.
3. **Macro-F1** — the headline number (imbalance, per EDA Finding 1).
4. **Micro-F1** — for LexGLUE-leaderboard comparison (published BERT-class
   yardstick: ~87–88 micro / ~82 macro).
5. **Per-label precision/recall/F1** + **top ~15 confused label pairs** (watch
   the known look-alikes: Governing Laws/Jurisdictions, Assigns/Successors,
   Amendments/Modifications, Waivers/No Waivers).

Write `eval_metrics.json` + `eval_report.txt` to `WORK_DIR` so they come back
as Kaggle artifacts, same filenames as T1/T2.

### 4. Smoke test on Kaggle, then production run

1. Save the notebook file, then `.\kaggle\run.ps1 -Wait` with the
   **Llama-3.2-1B** smoke config (optionally drop `TRAIN_PER_LABEL` to ~10 for
   a minutes-long end-to-end proof, mirroring T1/T2's sampled-CSV smoke runs).
2. Flip to `meta-llama/Meta-Llama-3.1-8B` + full stratified sample and run the
   production fine-tune (GPU T4×2, `HF_TOKEN` secret). Outputs land in
   `kaggle_output/` — keep them (adapter, metrics, report), e.g. in a
   `kaggle_output_task3_fine_tuned/` copy like the T1/T2 runs.

### 5. Baseline run (plan step 6a)

- Create `llama_3.1_task_3_no_fine_tune.ipynb` by mirroring
  `llama_3.1_task_2_no_fine_tune.ipynb`: same validation JSONL, same
  100-label instruction (fairness requires the base model to see the menu),
  base model with no adapter, same metrics, writes to a
  `no_finetune_baseline/`-style directory.
- Switch `code_file` in `kaggle/kernel-metadata.json` to it and run on Kaggle;
  keep the outputs (e.g. `kaggle_output_task3_baseline/`).

### 6. Comparison notebook (plan step 6b)

`task3_finetune_vs_baseline_comparison.ipynb`, mirroring
`task2_finetune_vs_baseline_comparison.ipynb`: side-by-side valid-label rate,
accuracy, macro/micro-F1, per-label deltas, top confusions for both models.

### 7. Documentation + housekeeping (plan step 7)

- Write `docs/task_3/TASK3_LOGIC.md`, `TASK3_NOTEBOOK_CELL_GUIDE.md` (once the
  notebook is final), and `TASK3_EVALUATION_METRICS.md`, mirroring
  `docs/task_2/`.
- Update the README status table and CLAUDE.md (T3 → implemented; add the
  LEDGAR data-layout, JSONL paths, adapter name, and staging script).
- Commit the currently untracked/modified T3 work: `ledgar/` JSONL,
  `llm_fine_tuning_LORA_task3.ipynb`, `kaggle/kernel-metadata.json`, and the
  new docs.

---

## Suggested immediate next action

Step 1 (Kaggle staging) and step 2 (training cells) are independent — the
notebook can be completed locally while the dataset uploads. Everything after
that is strictly sequential: smoke test → production run → baseline →
comparison → docs.
