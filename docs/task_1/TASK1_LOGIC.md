# Task 1 (Risk Clause Recognition) — Task Logic, I/O Format & Evaluation

This document defines **what Task 1 is**, independent of how the notebook pipeline
is built. For the deeper design notes see
[TASK1_HARD_NEGATIVES_PLAN.md](TASK1_HARD_NEGATIVES_PLAN.md) (the hard-negatives
fix) and [TASK1_LORA_APPLICATION.md](TASK1_LORA_APPLICATION.md) (how LoRA is
attached). The pipeline lives in
[llm_fine_tuning_LORA_task1_v2.ipynb](../../llm_fine_tuning_LORA_task1_v2.ipynb).

---

## 1. The Goal

**Given a single contract clause, decide whether it is an example of a specific
risk-clause category — answer strictly `Yes` or `No`.**

A human contract reviewer does this all day: they read a paragraph and decide
"yes, that's a *Non-Compete*" or "no, that's just a *Governing Law* line." Task 1
teaches the model to make that one binary call, asked once per category.

The whole task hinges on one hard requirement:

- **The model must actually read the clause.** It is not enough to answer `Yes`
  or `No` at the right overall rate — the answer has to depend on the *content* of
  the text in front of it. As we'll see in §2, the raw data makes it dangerously
  easy to cheat this, and the central design decision of the pipeline exists to
  stop that.

Contrast with Task 2:

| | Task 1 (Risk Clause Recognition) | Task 2 (Entity Extraction) |
| :--- | :--- | :--- |
| Question type | "Is this text a *X* clause?" | "What is the *X* in this text?" |
| Output space | Exactly `Yes` or `No` | Open-ended value wrapped in JSON |
| Skill measured | Clause recognition | Extraction + normalization + format discipline |
| Categories | 32 risk-clause columns | 9 entity columns |
| Failure looks like | Wrong class (or ignoring the input) | Wrong value, unparseable output, or hallucinated value |

### The 32 risk-clause categories

CUAD's `master_clauses.csv` has 41 category column-pairs. Task 1 uses the **32**
that are labelled `Yes`/`No` (clause present or absent) — e.g. `Non-Compete`,
`Exclusivity`, `Termination For Convenience`, `Change Of Control`,
`Uncapped Liability`. The remaining 9 entity categories (plus `Filename`) belong to
Task 2 and are **excluded** from Task 1 so the two tasks stay disjoint.

Each category has a `<Category>` column (the clause excerpt = model input) and a
`<Category>_Answer` column (`Yes`/`No` = ground truth).

### Why measure fine-tuning with this task

A base (non-fine-tuned) model given the same prompt often rambles, hedges, or
explains its reasoning instead of emitting a clean `Yes`/`No`. Fine-tuning on
(clause → Yes/No) pairs should collapse that into the one-token answer and, more
importantly, teach the model to base that answer on the clause. The before/after
comparison (see [llama_3.1_task_1_no_fine_tune.ipynb](../../llama_3.1_task_1_no_fine_tune.ipynb))
isolates what fine-tuning bought: **decision reliability and answer discipline** on
a highly imbalanced label distribution.

---

## 2. The Central Problem: the "cheating" leak, and hard negatives

This is the single most important idea in Task 1. Everything in the data pipeline
exists to serve it.

### The leak

In the raw CSV, a category's **input text is non-empty exactly when the answer is
`Yes`**. If a contract has a `Non-Compete` clause, the `Non-Compete` cell holds the
clause text; if it doesn't, the cell is blank (or a placeholder). So the label is
perfectly predictable from *whether there is any text at all*, without reading a
single word of it.

Train naively on that and the model learns a shortcut:

> **"Real text → `Yes`. Empty/placeholder → `No`."**

It scores near-perfectly and has learned **nothing about legal clauses**. The
metric lies.

### The fix — hard negatives

Every `No` example **borrows a real clause from a *different* category in the same
contract**. Concretely, for each contract:

1. Collect every clause actually present in that contract, as
   `{category: clause text}`.
2. For each of the 32 categories, ask the Yes/No question once:
   - **`Yes` case** → the contract really has this clause; use the category's own
     real text as the input.
   - **`No` case (hard negative)** → the contract does *not* have this clause;
     instead of a placeholder, pick a real clause from one of the **other**
     categories present in this contract, and label it `No`. If the contract has
     no other clause to borrow, skip the example.

Now **both `Yes` and `No` inputs are genuine legal text.** "Is there text?" is no
longer a usable signal — the *only* way to answer is to recognize the clause type.
These are called *hard* negatives because they are hard: real, plausible legal
prose that is simply the wrong category.

> A quick self-check in the pipeline asserts that no `No` input still contains the
> old `[No matching clause excerpt found...]` placeholder string. If it does, the
> leak isn't closed.

---

## 3. Input / Output Format

Every example is one Yes/No question about one category on one clause. The model
sees a prompt in the `### Instruction / ### Input / ### Response` template (the same
one Task 2 uses) and must produce **only** `Yes` or `No` after `### Response:`.

### The prompt template

```
### Instruction:
Is the following contract text a "{Category}" clause? Answer strictly "Yes" or "No".

### Input:
{clause excerpt}

### Response:
{Yes|No}        ← the model generates only this token
```

### The training JSONL schema

```json
{"instruction": "Is the following contract text a \"Non-Compete\" clause? Answer strictly \"Yes\" or \"No\".",
 "category": "Non-Compete",
 "input": "<clause text>",
 "output": "Yes"}
```

The `category` field is not fed to the model — it is carried along so evaluation can
break results down per category later.

### Case A — Positive (the clause is present)

**Input (Non-Compete):**
```
### Instruction:
Is the following contract text a "Non-Compete" clause? Answer strictly "Yes" or "No".

### Input:
During the Term, Distributor shall not, directly or indirectly, sell or market any
products that compete with the Products within the Territory.

### Response:
```

**Expected output:**
```
Yes
```

### Case B — Hard negative (a real clause from another category)

The contract has **no** `Non-Compete` clause, so we borrow a real clause that *is*
present — say its `Governing Law` text — and ask the `Non-Compete` question about it.

**Input (Non-Compete):**
```
### Input:
This Agreement shall be governed by and construed in accordance with the laws of
the State of New York.
```

**Expected output:**
```
No
```

The input is real legal text; only clause *recognition* gets the answer right.

### Outputs that count as WRONG

| Model output | Why it fails |
| :--- | :--- |
| `Yes, this appears to be a non-compete clause.` | Extra prose — the decoder only reads the first `Yes`/`No`; discipline still matters |
| `Maybe` / `N/A` | Not one of the two allowed labels |
| `No` when the clause really is present | Wrong class (a false negative — the costly error for rare risk clauses) |
| `Yes` on every input regardless of text | The exact "cheating" failure hard negatives are designed to expose |

---

## 4. Making the Training Signal Honest

Two more pieces protect the signal, on top of hard negatives.

### Split by contract, not by clause

The train/validation split is done on **contracts first (85/15)**, and examples are
built from each side afterwards. If we split at the clause level instead, clauses
from the *same contract* could land in both train and validation, and the model
could recognize contract-specific wording it had already seen — a leak that inflates
validation scores. Splitting whole contracts keeps validation genuinely unseen.

### Balance the classes — on train only

Depending on how many clauses each contract has, hard negatives can leave `Yes` and
`No` imbalanced. Left alone, the model drifts toward always answering the majority
class. So the **train** split is downsampled toward a ~1:1 `Yes`:`No` ratio.

**Validation is deliberately left at its natural distribution** — balancing it too
would hide how the model behaves on the real, imbalanced world and make the metrics
dishonest.

### Completion-only loss

The answer is a single token (`Yes`/`No`), but the prompt (instruction + clause) is
long. If we computed the training loss over the whole sequence, the gradient would
be dominated by the model learning to reproduce the clause text, and the tiny
`Yes`/`No` decision signal would be drowned out.

Instead the pipeline uses **completion-only loss**: the dataset is fed as
`prompt` / `completion` pairs and `SFTConfig(completion_only_loss=True)` masks every
prompt token (`-100`) so **only the answer token contributes to the loss**. The
`### Response:\n` marker sits exactly at the boundary where masking switches off.

> This is a silent-failure risk: if masking is misconfigured it can mask *every*
> token, the loss is then identically 0, and training does nothing while looking
> fine. The train cell asserts the final loss is finite and **greater than 0** to
> catch that.

---

## 5. QLoRA in one paragraph

The base model (`meta-llama/Meta-Llama-3.1-8B`) is loaded in **4-bit NF4**
quantization so an 8B model fits in a single T4's 16 GB. Its weights are **frozen**;
training only updates a small **LoRA adapter** (`r=16`, `lora_alpha=16`, targeting
`q_proj, k_proj, v_proj, o_proj`). That's the "QLoRA" recipe — quantized base +
low-rank adapter — and it's why the whole run fits on Kaggle's free GPU. Only the
adapter (a few MB) is saved as the output, not a full 8B checkpoint. Details in
[TASK1_LORA_APPLICATION.md](TASK1_LORA_APPLICATION.md).

---

## 6. Evaluating the Fine-Tuned Model

Generation is greedy (`do_sample=False`, `max_new_tokens=3`) on the held-out
**validation contracts**. Each prediction is reduced to `Yes`/`No` (the decoded text
is `Yes` if it contains "yes", else `No`), then scored.

### Why accuracy alone is useless here

The label distribution is heavily imbalanced — most risk clauses are rare, so most
questions have the answer `No`. A model that **always answers `No`** can score 80 %+
accuracy while catching zero actual risk clauses. Accuracy (and training loss) hide
exactly the failure we care about.

### The metric — per-class precision / recall / F1 + confusion matrix

The pipeline reports `sklearn`'s `classification_report` for both classes plus the
confusion matrix. What each number means here:

| Metric (for the `Yes` class) | Plain-English question |
| :--- | :--- |
| **Precision** | When the model says `Yes`, how often is it right? (few false alarms) |
| **Recall** | Of the clauses that really are present, how many did it catch? (few misses) |
| **F1** | The balance of the two |

For rare risk clauses, **recall on `Yes` is usually the number that matters most** —
missing a real risk clause (a false negative) is the costly error in contract
review.

```
              precision    recall  f1-score   support
         Yes       ...       ...       ...       ...
          No       ...       ...       ...       ...

Confusion matrix (rows = true [Yes, No], cols = pred [Yes, No]):
[[TP  FN]
 [FP  TN]]
```

Numbers are persisted as `eval_metrics.json` + `eval_report.txt` in `WORK_DIR` so
Kaggle runs return them as downloadable artifacts (and training stats go to
`train_metrics.json`).

### The sanity check that proves the leak is closed

**A model that ignores the input should now score around 50 %, not ~100 %.** If
validation F1 comes back near-perfect immediately, that is a red flag — re-inspect
the inputs, because the hard-negatives leak may not actually be closed. A *believable*
score that is clearly better than chance but short of perfect is the sign the model
is genuinely recognizing clauses.

### The measurement that answers the research question

Run the **identical evaluation twice** — once on the base
`meta-llama/Meta-Llama-3.1-8B` with no adapter
([llama_3.1_task_1_no_fine_tune.ipynb](../../llama_3.1_task_1_no_fine_tune.ipynb)),
once on the fine-tuned adapter — same prompts, same validation JSONL, same metrics.
The deliverable is the delta table (built in
[finetune_vs_baseline_comparison.ipynb](../../finetune_vs_baseline_comparison.ipynb)):

| Metric | Base model | Fine-tuned | Δ |
| :--- | :---: | :---: | :---: |
| `Yes` precision | *(measured)* | *(measured)* | |
| `Yes` recall | *(measured)* | *(measured)* | |
| `Yes` F1 | *(measured)* | *(measured)* | |

Expected shape: the base model tends to be trigger-happy (says `Yes` too often, or
answers in prose that gets read as `Yes`), giving high recall but poor precision;
fine-tuning should tighten precision and lift F1 while keeping recall respectable. If
the fine-tuned model instead collapses to always-`No` (high accuracy, near-zero `Yes`
recall), suspect over-aggressive class balancing or too few positive examples before
suspecting the model.
