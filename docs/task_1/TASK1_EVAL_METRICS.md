# Task 1 — Evaluation Metrics

Documents the metrics computed in [llm_fine_tuning_LORA_task1_v2.ipynb](../../llm_fine_tuning_LORA_task1_v2.ipynb) (Step 6b, "Evaluate on validation").

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
