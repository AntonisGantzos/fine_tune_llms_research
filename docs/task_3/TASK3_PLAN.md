# Task 3 — Jurisdiction / Provision-Type Classification (LEDGAR): Implementation Plan

Step-by-step plan for adding the dataset, exploring it, preprocessing it, training the
model the same way as Tasks 1 and 2, and evaluating it with the right metrics.

Status: **planning** — nothing for T3 exists yet (no notebook, no data script, no docs
besides this file).

---

## What Task 3 actually is

The README defines T3 as: *given a contract clause, classify what type of provision it
is* (e.g. "Governing Laws", "Terminations", "Amendments"), using the **LEDGAR**
dataset. Where T1 asked a Yes/No question and T2 asked for a JSON value, T3 asks the
model to pick **one label out of a fixed list**. It is the simplest of the three tasks
conceptually — plain multi-class classification — but with a new dataset, so most of
the work is in data handling.

### Key decision: LexGLUE's LEDGAR, not raw LEDGAR

Raw LEDGAR is ~850,000 provisions with 12,000+ messy labels — far too big and too
noisy to fine-tune on a free Kaggle T4. The standard research version is the
**LexGLUE benchmark's LEDGAR subset** (`coastalcph/lex_glue`, config `ledgar` on
Hugging Face):

- ~80,000 provisions (60k train / 10k validation / 10k test)
- only the **100 most frequent labels**
- exactly **one label per provision** (single-label, not multi-label)
- a ready-made train/validation/test split (by SEC filing year)

Published papers report numbers on this version, so our results can be compared
against theirs.

---

## Step 1 — Get the dataset (`scripts/download_ledgar.py`)

Write a small script mirroring `scripts/download_cuad.py`, but using
`datasets.load_dataset("coastalcph/lex_glue", "ledgar")` and saving to `data/LEDGAR/`
(git-ignored, like CUAD). Each example is just `{text, label}` where `label` is a
number 0–99; the dataset's metadata provides the label names. Save the three splits
(train/validation/test) as CSV/parquet files so everything downstream reads plain
files, same as CUAD's CSV.

## Step 2 — Explore the data (`LEDGAR_dataset_exploration.ipynb`)

Mirror `CUAD_dataset_exploration.ipynb` and answer four questions before any training:

1. **Label balance** — how many examples per label? LEDGAR is imbalanced (some
   provision types are 10× more common than others). This decides whether macro-F1 or
   accuracy is the honest headline metric (as with T1: macro-F1).
2. **Text length** — histogram of provision length in tokens. Training uses
   `max_length=1024`; confirm provisions + the instruction fit. LEDGAR provisions are
   short (usually well under 200 tokens), but verify.
3. **Prompt length** — the instruction must contain the list of 100 allowed labels
   (otherwise the model can't know its options, and the baseline comparison would be
   meaningless). 100 label names ≈ 300–400 tokens. Confirm instruction + longest
   provision still fits in 1024.
4. **Eyeball examples** — print a few provisions per label to sanity-check that the
   labels make sense.

## Step 3 — Preprocess into training JSONL

Follow the exact T1/T2 schema so all existing tooling works:

```json
{"instruction": "Classify the following contract provision. Answer with exactly one label from this list: [Adjustments, Agreements, ..., Withholdings].",
 "category": "<gold label name>",
 "input": "<provision text>",
 "output": "<gold label name>"}
```

Preprocessing decisions, in order:

- **Subsample.** 60,000 training examples would take many Kaggle sessions. T1 trained
  on ~2,600 examples and worked well. Take a **stratified sample** — e.g. 60–100
  examples per label → 6,000–10,000 train examples — so every label is represented.
  Sample validation similarly (e.g. 10–20 per label → 1,000–2,000 examples) to keep
  eval generation time sane.
- **Use the dataset's built-in splits.** Unlike CUAD there is no contract-leakage
  problem to solve manually — LexGLUE already split LEDGAR by filing year, so sample
  *within* the given train and validation splits.
- **Map label IDs to names** once, and use the human-readable name as the `output`
  (the model generates text, not a number).
- Write task-specific filenames (as T2 did):
  `ledgar_task3_train.jsonl` + `ledgar_task3_validation.jsonl`.
- **Sanity-check cell** (mirror T2's pre-save checks): every one of the 100 labels
  appears in both splits, and every `output` is verbatim in the allowed-label list.

Then stage the data for Kaggle: add the subsampled files to the Kaggle dataset via
`kaggle/stage_data.ps1` + `python -m kaggle datasets version`, same as the CUAD CSVs.

## Step 4 — Training notebook (`llm_fine_tuning_LORA_task3.ipynb`)

Copy `llm_fine_tuning_LORA_task2.ipynb` and change as little as possible — the T2
notebook was itself a deliberate mirror of T1, and that discipline paid off. The
deltas:

- Stage 2–4 (data cells) load the LEDGAR files instead of the CUAD CSV and build the
  JSONL from Step 3.
- Same base model (`meta-llama/Meta-Llama-3.1-8B`), same QLoRA config (4-bit NF4,
  r=16, alpha=16, targets `q/k/v/o_proj`), same
  `### Instruction / ### Input / ### Response` prompt template, same
  `completion_only_loss=True`. The completion is now a short label string — closer to
  T1's Yes/No than T2's JSON.
- Adapter saves to `llama-3.1-8B-ledgar-task3/`.
- Keep the pre-flight-check cell, but swap the "completion parses as JSON" assert for
  "completion is one of the 100 labels".

**Run order on Kaggle** (existing workflow — edit `code_file` in
`kaggle/kernel-metadata.json`, then `.\kaggle\run.ps1 -Wait`):

1. **Smoke test first** with `meta-llama/Llama-3.2-1B` and the sampled data — proves
   the pipeline end-to-end in minutes.
2. **Production run** with Llama-3.1-8B on the full stratified sample.

## Step 5 — Evaluation metrics

Standard multi-class classification: T1's metrics generalized from 2 classes to 100,
plus one new "format" metric that plays the role T2's JSON-validity played:

1. **Valid-label rate** — fraction of generations that are *exactly* one of the 100
   allowed labels (after strip/lowercase). A generative model can output anything;
   this measures format discipline first, because the other metrics only mean
   something for parseable outputs. Expect the base model to be weak here and
   fine-tuning to fix it — same story as T2's JSON validity.
2. **Accuracy** — simple fraction correct.
3. **Macro-F1** — the **headline number**. It averages F1 over the 100 labels equally,
   so rare provision types count as much as common ones; with imbalanced labels,
   accuracy alone can look good while rare classes fail completely. Same reasoning as
   T1's headline choice.
4. **Micro-F1** — reported because it is what the LexGLUE leaderboard uses, letting us
   compare against published results (fine-tuned BERT-class models score roughly
   87–88 micro / 82 macro on this task — a useful external yardstick).
5. **Per-label breakdown + top confusions** — precision/recall/F1 per label (like
   T1's per-category eval), plus the most common wrong-label pairs (e.g. "Governing
   Laws" vs "Jurisdictions" are plausibly confusable). A full 100×100 confusion
   matrix is unreadable; list the top ~15 confused pairs instead.

Write `eval_metrics.json` + `eval_report.txt` to `WORK_DIR` as before so they come
back as Kaggle artifacts.

## Step 6 — Baseline + comparison

Mirror the T1/T2 pattern exactly:

- `llama_3.1_task_3_no_fine_tune.ipynb` — the un-fine-tuned base model evaluated on
  the *same* validation JSONL with the *same* prompt (label list included, so it has a
  fair chance). Writes to a `no_finetune_baseline/`-style directory.
- `task3_finetune_vs_baseline_comparison.ipynb` — side-by-side table of valid-label
  rate, accuracy, macro/micro-F1, and per-label deltas.

Expected result shape: valid-label rate improves massively (format discipline),
macro-F1 improves a lot (the base model won't know LEDGAR's exact label vocabulary),
and the fine-tuned model should land in the neighborhood of the published
encoder-model scores.

## Step 7 — Documentation

Fill out `docs/task_3/` mirroring `docs/task_2/`:

- `TASK3_LOGIC.md` — what the task is
- `TASK3_PLAN.md` — this file
- `TASK3_NOTEBOOK_CELL_GUIDE.md` — cell-by-cell guide once the notebook exists
- `TASK3_EVALUATION_METRICS.md` — metric definitions and rationale

Update the README status table and CLAUDE.md when done.

---

## Suggested order of work

1. Download script + EDA notebook (all local, no GPU needed) — validates the dataset
   choice cheaply.
2. Preprocessing cells + JSONL + sanity checks (local).
3. Stage data to Kaggle, smoke-test with the 1B model.
4. Production 8B fine-tune, then baseline run, then comparison notebook.
5. Docs as you go.

The only genuinely new engineering is Steps 1–3 (new dataset in, new label-list
prompt); everything from QLoRA config to the Kaggle workflow to the eval-report
structure is a copy of what already works.
