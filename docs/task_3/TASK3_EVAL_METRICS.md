# Task 3 — Evaluation Metrics Explained

Documentation for the metrics computed in [llm_fine_tuning_LORA_task3.ipynb](../../llm_fine_tuning_LORA_task3.ipynb) (Step 5).

**The task:** given one contract provision, pick the correct label out of 100 (e.g. *Governing Laws*, *Notices*, *Terminations*).

The model is scored on the 1,945-example validation set with greedy generation. Metrics are written to `eval_metrics.json` and `eval_report.txt`.

---

## What is actually being scored

**Training data.** LEDGAR (LexGLUE), staged as `ledgar-lexglue` and built into JSONL on Kaggle. The raw splits (60,000 train / 10,000 validation) are **stratified down to ~100 examples per label** for train and ~20 for validation, because LEDGAR is 137× imbalanced and the full set would drown the model in *Governing Laws*. Result: **9,801 train / 1,945 validation** examples, all 100 labels present in train, 99 in validation (*Books* has too few rows to survive the split). Unlike Tasks 1 and 2, there is no contract-level split to construct — LEDGAR ships its own train/validation/test splits and the test set is held out entirely.

**Input the model sees at eval time** — byte-identical to the training prompt (same `build_prompt()`, same provision trimming to 670 tokens), response left empty:

```
### Instruction:
Classify the following contract provision. Answer with exactly one label from this list:
[Adjustments, Agreements, Amendments, ... , Waivers, Warranties, Withholdings].

### Input:
<provision text>

### Response:
```

The 100-label menu is **341 tokens of every single prompt** — the model is not asked to recall the label set from memory, it is asked to pick from a list it is shown each time.

**Output it is expected to produce** — the completion it was fine-tuned on: the label string verbatim and nothing else, e.g.

```
Governing Laws
```

**How the raw completion becomes a prediction.** Greedy generation, `max_new_tokens` = longest label + 2, then: take the **first line only**, strip, lowercase, and look it up in the 100-label table. A hit maps back to the canonical label; a miss becomes the sentinel `__INVALID__`, which is *not* in `LABELS` and therefore counts as a **wrong answer** in every metric below — never as a free pass. That lookup is what metric 1 measures.

---

## The metrics, in the order they matter

### 1. Valid-label rate
**What:** Fraction of answers that are *exactly one of the 100 allowed labels* (after strip + lowercase, first line only).

**Why it matters:** The model generates free text, so it could invent a label, add extra words, or ramble. This checks it actually answers in the required format. Anything unparseable is marked `__INVALID__` and counted as **wrong** — never quietly forgiven.

**Reading it:** Close to `1.0` = the model learned the output format. Low = the model is misbehaving before we even ask if it's *correct*.

---

### 2. Accuracy
**What:** Fraction of provisions where the predicted label == the gold label.

**Why it matters:** The most intuitive "how often is it right?" number.

**Caveat:** It can be misleading here. LEDGAR is **137× imbalanced** — some labels are very common. A model that only nails the common labels can still post a high accuracy while failing on rare ones. That's why accuracy is *not* the headline.

---

### 3. Macro-F1  ← headline number
**What:** F1 score computed **per label**, then averaged with every label weighted equally.

(F1 = balance of precision and recall. Precision = "when it says label X, how often is it right?" Recall = "of all true label-X provisions, how many did it catch?")

**Why it matters:** Because it treats a rare label the same as a common one, a model can't hide weak performance on rare classes behind the popular ones. This is the truest measure of whether the model learned **all** 100 provision types, not just the frequent handful.

**Reading it:** Higher = the model is competent across the board. Published BERT-class models score ~82 macro on LEDGAR — that's the yardstick.

---

### 4. Micro-F1
**What:** F1 computed over **all predictions pooled together** (each example counts once, so common labels dominate).

**Why it matters:** It's the standard number on the LexGLUE public leaderboard, so it lets us compare against published results (~87–88 micro). It rewards getting the high-volume labels right.

**Macro vs. micro in one line:** macro asks "are you good at *every* label?"; micro asks "are you right on *most provisions*?"

---

### 5. Per-label precision / recall / F1
**What:** The three scores broken out for each of the 100 labels, sorted worst-F1-first.

**Why it matters:** The single averaged numbers hide *where* the model struggles. This table points to the exact labels that need more data or are inherently hard.

---

### 6. Top 15 confused (gold → pred) pairs
**What:** The most frequent mistakes, as "true label → what the model guessed instead".

**Why it matters:** Shows *which* labels the model mixes up. Some pairs are genuinely look-alike legal concepts — watch for:
- Governing Laws ↔ Jurisdictions
- Assigns ↔ Successors
- Amendments ↔ Modifications
- Waivers ↔ No Waivers

Confusions on these are more forgivable than random errors, and they suggest where clearer training examples would help most.

---

## Quick summary

| Metric | Question it answers | Good sign |
|---|---|---|
| Valid-label rate | Does it answer in the right format? | ~1.0 |
| Accuracy | How often is it right overall? | High (but can mislead) |
| **Macro-F1** | Is it good at **every** label, rare ones included? | **High — the real test** |
| Micro-F1 | Is it right on **most** provisions? | High (leaderboard-comparable) |
| Per-label table | *Which* labels does it fail on? | Few low-F1 rows |
| Top confusions | *Which* labels get mixed up? | Only look-alike pairs |

**Bottom line:** valid-label rate confirms the model behaves, macro-F1 is the headline for whether it truly learned the task, and the per-label table + confusions tell you where to improve next.
