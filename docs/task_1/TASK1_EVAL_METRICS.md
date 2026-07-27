# Task 1 — Evaluation Metrics

Documents the metrics computed in [llm_fine_tuning_LORA_task1_v2.ipynb](../../llm_fine_tuning_LORA_task1_v2.ipynb) (Step 6b, "Evaluate on validation").

## What is actually being scored

**Training data.** CUAD `master_clauses_cleaned.csv` → `cuad/train/cuad_train.jsonl` + `cuad/validation/cuad_validation.jsonl`, one example per (contract, category) pair across the **32 Yes/No categories** (the 9 Task 2 entity categories and `Filename` are excluded — the two tasks use disjoint category sets). The split is **contract-level 85/15**, so no clause from a validation contract appears in training. Every `No` example uses a **hard negative** — a real clause borrowed from a *different* category of the *same* contract — so `Yes` and `No` inputs are both genuine legal text and the model cannot cheat on "real text vs. placeholder".

The fine-tuned Kaggle run trained on ≈6.1k examples (764 optimizer steps × effective batch 8, one epoch) and is scored on **2,208 validation examples**.

**Input the model sees at eval time** — byte-identical to the training prompt, with the response left empty:

```
### Instruction:
Is the following contract text a "Cap On Liability" clause? Answer strictly "Yes" or "No".

### Input:
<clause text>

### Response:
```

**Output it is expected to produce** — the completion it was fine-tuned on: the single token-string `Yes` or `No`, nothing else. This is the *only* thing fine-tuning teaches it to emit here; the base model tends to answer in a sentence, which is the main behavioural difference the baseline comparison exposes.

**How the raw completion becomes a prediction.** Greedy generation, `max_new_tokens=3`, then `"Yes" if "yes" in completion.lower() else "No"`. Note the consequence: this parse is **lenient** — it never fails, so Task 1 has no "format validity" metric. A rambling completion is silently coerced into a class rather than counted as malformed (Tasks 2 and 3 do report a validity rate, because their output formats can genuinely break). Every metric below is computed on those coerced `Yes`/`No` labels versus the gold `output` field.

## The task in one line

For each contract clause, the model answers **`Yes`** (this *is* a "[Category]" clause) or **`No`** (it is not). So every prediction is one of:

- **True Positive (TP)** — clause is present, model said `Yes` ✅
- **True Negative (TN)** — clause is absent, model said `No` ✅
- **False Positive (FP)** — clause is absent, but model said `Yes` ❌ (false alarm)
- **False Negative (FN)** — clause is present, but model said `No` ❌ (missed it)

All metrics below are just different ways of counting these four outcomes.

## Why not just accuracy?

**Accuracy** = fraction of predictions that are correct. Simple, but **misleading here**.

Most clauses are absent (`No`), so a lazy model that *always* answers `No` can score 80%+ while catching **zero** real clauses. Accuracy is reported, but it is **not** the headline number. The metrics below are what actually tell us if the model works.

## The core metrics (reported per class)

Each is computed separately for the `Yes` class and the `No` class.

| Metric | Plain-English question | Formula | Hurts when... |
|--------|------------------------|---------|---------------|
| **Precision** | When the model says `Yes`, how often is it right? | TP / (TP + FP) | Model cries wolf (many false alarms) |
| **Recall** | Of all the real `Yes` clauses, how many did it catch? | TP / (TP + FN) | Model misses real clauses |
| **F1** | Single score balancing the two above | 2·(P·R)/(P+R) | *Either* precision *or* recall is bad |

### How to read them

- **High precision, low recall** → model is cautious: rarely wrong when it says `Yes`, but misses a lot.
- **Low precision, high recall** → model is trigger-happy: catches most clauses, but with many false alarms.
- **F1** is the one-number summary — use it to compare models. In legal review, **recall on the `Yes` class matters most**: missing a risky clause (FN) is usually worse than a false alarm (FP) a human can dismiss.

## Support

**Support** = how many real examples of each class were in the validation set. It is context, not a score — an F1 based on 5 examples is far less trustworthy than one based on 500.

## Confusion Matrix

A 2×2 table (rows = truth, cols = prediction) showing the raw TP / TN / FP / FN counts:

```
                pred Yes    pred No
true Yes          TP          FN
true No           FP          TN
```

Everything above is derived from these four numbers. Read it to see *how* the model fails — a big FN cell means it misses clauses; a big FP cell means it over-flags.

## The sanity check

Because train/val is split at the **contract level** (no leakage), a model that ignores the input should score around **50%**, not ~100%. If validation F1 is near-perfect right away, suspect a data leak and re-inspect the inputs.

## What gets saved

The notebook writes two artifacts to `WORK_DIR` (retrieved via `kaggle kernels output`):

- **`eval_metrics.json`** — machine-readable: accuracy, full per-class report, confusion matrix.
- **`eval_report.txt`** — human-readable printout of the same.

The research deliverable is the **delta** between this fine-tuned run and the base-model baseline ([llama_3.1_task_1_no_fine_tune.ipynb](../../llama_3.1_task_1_no_fine_tune.ipynb)) — compared in [finetune_vs_baseline_comparison.ipynb](../../finetune_vs_baseline_comparison.ipynb).
