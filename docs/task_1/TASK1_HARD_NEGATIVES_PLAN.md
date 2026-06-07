# Task 1 — Hard Negatives Plan

A focused, single-option implementation guide for fixing
[llm_fine_tuning_LORA_task1.ipynb](../llm_fine_tuning_LORA_task1.ipynb).

We are committing to **Option A — Hard Negatives**.

---

## 1. The goal, in one sentence

> Make the model read **real contract text in both the Yes and No cases**, so it
> has to *understand the clause* instead of guessing from the shape of the input.

---

## 2. Why the current notebook is broken (the leak)

In `master_clauses_cleaned.csv`, every category has two columns:

| Column | When answer = Yes | When answer = No |
|---|---|---|
| `<Category>` (the model **input**) | the real clause text | **empty** |
| `<Category>_Answer` (the **label**) | `Yes` | `No` |

The input is non-empty *exactly when* the answer is Yes. So today the notebook trains on:

- **Yes** → input = real clause text → output `Yes`
- **No**  → input = the placeholder `"[No matching clause excerpt found in contract.]"` → output `No`

The model only has to learn one trick: **placeholder → No, any real text → Yes.**
It never reads a single legal clause. Loss looks great; nothing is learned.

---

## 3. The fix, with an example

For a `No` example, **stop using the placeholder.** Instead, borrow a *real clause
from a different category in the same contract*. Now the model can't cheat — both
Yes and No inputs are genuine legal text, and the only way to answer is to actually
recognize the clause type.

**Concrete example** — building examples for the category `"Non-Compete"`:

| Case | Old input (broken) | New input (hard negative) | Label |
|---|---|---|---|
| Contract *has* a Non-Compete clause | "The Seller agrees not to compete within..." | "The Seller agrees not to compete within..." | `Yes` |
| Contract has *no* Non-Compete clause | `[No matching clause excerpt found...]` | "This Agreement shall be governed by the laws of Delaware..." *(a real Governing-Law clause borrowed from the same contract)* | `No` |

The question the model must now answer becomes a real one:

> *"Is the following contract text a `Non-Compete` clause? Yes / No"*

—and for the `No` row, the borrowed Governing-Law text is genuine legal language
that simply **isn't** a Non-Compete clause. That is exactly the discrimination we
want the model to learn.

---

## 4. The logic to follow

1. Split **contracts** into train/val *first* (never let one contract's clauses
   land in both sets).
2. For each contract row, gather **all clauses actually present** in that contract,
   keyed by category.
3. For each of the 32 categories, build one example:
   - **Yes** → use that category's own real clause text.
   - **No** → randomly pick a real clause from a **different** present category
     (a "hard negative"). If the contract has no other clause to borrow, skip it.
4. Balance the classes **on train only**; leave validation at its natural
   distribution so metrics stay honest.
5. Report **per-class precision / recall / F1**, not just accuracy or loss.

---

## 5. Step-by-step technical changes to the notebook

> Cell IDs below match the current notebook. Steps 1, 2, 4 are "leave it alone";
> the real work is Step 3 (hard negatives), Step 5 (balancing), Step 6 (metrics),
> and Step 7 (completion-only loss).

### Step 1 — Load the full CSV, not the 20-row sample
- **Cell:** `7a681f30` (`MASTER_CLAUSES_PATH`)
- **Change:** point at `master_clauses_cleaned.csv` (510 contracts) instead of
  `master_clauses_cleaned_sampled.csv`. Keep the sample only for a quick smoke test.
- **Why:** 20 contracts → ~3 validation contracts → metrics are pure noise. A fix
  can't be judged on 3 contracts.

```python
# MASTER_CLAUSES_PATH = CUAD_PATH/'master_clauses_cleaned_sampled.csv'  # smoke test only
MASTER_CLAUSES_PATH = CUAD_PATH/'master_clauses_cleaned.csv'
```

### Step 2 — Leave load / clean / 32-category cells unchanged
- **Cells:** `a46465a4`, `db61ef7a`, `step3code`
- **Change:** none. Column cleaning, the `_Answer` rename, and the
  `assert len(task1_categories) == 32` derivation are all correct.

### Step 3 — Rewrite `build_examples` to use hard negatives ⭐ (the real fix)
- **Cell:** `0515b709`
- **Change:** a `No` example now borrows a real clause from a *different* present
  category instead of emitting the placeholder.

```python
import random
random.seed(42)

def nonempty_context(row, cat):
    v = row.get(cat)
    return str(v).strip() if pd.notna(v) and str(v).strip() else None

def build_examples(frame):
    rows = []
    for _, row in frame.iterrows():
        # every clause actually present in THIS contract, keyed by category
        present = {c: nonempty_context(row, c)
                   for c in task1_categories if nonempty_context(row, c)}
        for category in task1_categories:
            if to_binary(row[f"{category}_Answer"]) == "Yes":
                text, label = present[category], "Yes"
            else:
                # hard negative: a real clause from a DIFFERENT category
                others = [t for c, t in present.items() if c != category]
                if not others:
                    continue                      # nothing to borrow → skip
                text, label = random.choice(others), "No"
            rows.append({
                "instruction": f'Is the following contract text a "{category}" clause? Answer strictly "Yes" or "No".',
                "category": category,
                "input": text,
                "output": label,
            })
    return rows
```

- **Why:** this closes the leak. `No` inputs are now real legal text, so the model
  must learn what each clause type *looks like* rather than detecting a placeholder.
- **Note:** the instruction wording changes from *"Does this contract contain..."*
  to *"Is the following contract text a ... clause?"* — because we now feed a single
  clause excerpt, not a whole contract.

### Step 4 — Keep the contract-level split
- **Cell:** `0515b709` (`train_test_split(df, ...)` then build per side)
- **Change:** none — still split `df` (contracts) *before* building examples.
- **Why:** prevents a contract's clauses leaking across train/val.

### Step 5 — Balance classes on the train split only (new cell)
- **Where:** new cell, after `train_data` is built, before saving.
- **Change:** downsample the majority class toward ~1:1 (or up to 3:1) on **train
  only**. Leave `val_data` at its natural distribution.

```python
from collections import defaultdict

def balance(data, ratio=1.0, seed=42):
    rng = random.Random(seed)
    pos = [e for e in data if e["output"] == "Yes"]
    neg = [e for e in data if e["output"] == "No"]
    keep_neg = min(len(neg), int(len(pos) * ratio))
    neg = rng.sample(neg, keep_neg)
    out = pos + neg
    rng.shuffle(out)
    return out

train_data = balance(train_data, ratio=1.0)   # train only — val untouched
```

- **Why:** without balancing the model drifts to always answering the majority
  class. Validation must stay natural so metrics reflect reality.

### Step 6 — Report real metrics, not just counts
- **Cell:** replace count-only `step6code`; add an eval after training.
- **Change:** compute **per-class precision / recall / F1 + confusion matrix** on
  validation (`sklearn.metrics.classification_report`). Keep the Yes/No count print
  as a one-line sanity check.
- **Why:** under imbalance, accuracy is meaningless ("always No" can score 80%+).
  P/R/F1 is the only honest signal that the model learned something.

### Step 7 — Train on the answer only (completion-only loss)
- **Cell:** `aaf3edf8` (the `SFTTrainer` setup)
- **Change:** mask loss so only the text after `### Response:` is learned.

```python
from trl import DataCollatorForCompletionOnlyLM

collator = DataCollatorForCompletionOnlyLM(
    response_template="### Response:\n",
    tokenizer=tokenizer,
)
# pass data_collator=collator into SFTTrainer(...)
```

- **Why:** today loss is computed over the whole prompt (instruction + clause +
  answer). With a 1-token answer, the gradient is dominated by reproducing the
  clause text, drowning out the Yes/No signal. Masking focuses learning on the
  decision.

### Step 8 — Minor training-config tidy-ups (optional)
- **Cell:** `aaf3edf8`
- **Changes:**
  - prefer `bf16=True` over `fp16=True` if the GPU supports it (more stable);
  - widen LoRA targets to `["q_proj","k_proj","v_proj","o_proj"]` for a slightly
    stronger adapter;
  - lower `max_seq_length` (e.g. `1024`) — single clauses are short, so this speeds
    training and saves memory.

---

## 6. How to know the fix worked

1. **Inspect the data:** after Step 3, print a few `No` examples — their `input`
   must be *real clause text*, never the old placeholder string.
2. **Dumb-baseline test:** a model that ignores the input should now score ~50%,
   not ~100%. If validation F1 is near-perfect immediately, the leak isn't fully
   closed — re-inspect the inputs.
3. **Headline number = per-class F1 on validation**, not loss or accuracy.

---

## 7. Checklist

- [ ] Step 1 — load full CSV (510 contracts); sample only for smoke test
- [ ] Step 2 — load / clean / 32-category cells left unchanged
- [ ] Step 3 — `build_examples` uses **hard negatives** (real clause from another category) ⭐
- [ ] Step 4 — split stays contract-level, before example building
- [ ] Step 5 — class balancing applied to **train only**
- [ ] Step 6 — per-class precision/recall/F1 + confusion matrix on val
- [ ] Step 7 — completion-only loss (mask everything before `### Response:`)
- [ ] Step 8 — (optional) bf16, wider LoRA targets, lower `max_seq_length`
- [ ] Validation — `No` inputs are real text; dumb baseline ≈ 50%
