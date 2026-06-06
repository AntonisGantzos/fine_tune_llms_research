# Plan — Fixing `llm_fine_tuning_LORA_task1.ipynb`

This plan turns the Task 1 notebook from a **degenerate quiz** (where the answer
is hidden in the shape of the input) into a **real risk-clause recognition task**.

---

## The one problem that matters

In `master_clauses_cleaned.csv`, for each category there are two columns:

| Column | Filled when answer = Yes | Filled when answer = No |
|---|---|---|
| `<Category>` (the "context" / model input) | the actual clause text | **empty** |
| `<Category>_Answer` (the label) | `Yes` | `No` |

I verified this is a **perfect correlation** across all 510 contracts — context is
non-empty *exactly when* the answer is Yes, with zero exceptions.

The notebook feeds that context column to the model as `input`. So:

- Yes example → input is the clause itself → output `Yes`
- No example → input is the fixed placeholder `"[No matching clause excerpt found in contract.]"` → output `No`

The label is recoverable from one bit: *is this the placeholder or real text?* The
model learns `placeholder → No, text → Yes` and never reads a single clause. Loss
looks great; nothing legal is learned.

**Goal of this plan:** the model must see *real contract text in both the Yes and
No cases*, so presence/absence is no longer encoded by the input's shape.

---

## Two ways to fix it (pick one)

### Option A — Hard negatives (recommended; runs on consumer GPU)
For a `No` example of category X, **don't** use a placeholder. Use a *real clause
pulled from a different category* in the same contract. Now both Yes and No inputs
are genuine contract text, and the model must decide *"is this text an X clause?"*
— a legitimate classification task. No need for full contracts, so it stays light.

### Option B — Full-contract input (most faithful to true Task 1)
Feed the whole contract (from `data/CUAD_v1/full_contract_txt/`, chunked to fit
`max_seq_length`) and ask whether clause X appears anywhere. This is the "real"
T1 but is heavier (long inputs, truncation, retrieval concerns). Use only if
Option A proves too easy.

The steps below implement **Option A**, with notes for Option B where relevant.

---

## Step-by-step changes

### Step 1 — Train on the full CSV, not the 20-row sample
- **Where:** cell `7a681f30` (`MASTER_CLAUSES_PATH`)
- **Change:** point to `master_clauses_cleaned.csv` (510 contracts), not
  `..._sampled.csv`. Keep the sample only for a quick smoke test.
- **Why:** 20 contracts → 3 validation contracts → metrics are noise. You can't
  judge a fix on 3 contracts.

### Step 2 — Keep Steps 1–3 as they are (load, clean, derive 32 categories)
- **Where:** cells `a46465a4`, `db61ef7a`, `step3code`
- **Change:** none.
- **Why:** column cleaning, the `_Answer` rename, and the 32-category derivation
  (Task 2 fields excluded, `assert == 32`) are all correct. Don't touch what works.

### Step 3 — Replace the example builder with hard negatives  ⭐ the real fix
- **Where:** cell `0515b709` (`build_examples`)
- **Change:** rewrite so a `No` example uses a *real clause from another category*
  instead of the placeholder. Sketch:

  ```python
  import random
  random.seed(42)

  def nonempty_context(row, cat):
      v = row.get(cat)
      return str(v).strip() if pd.notna(v) and str(v).strip() else None

  def build_examples(frame):
      rows = []
      for _, row in frame.iterrows():
          # all clauses actually present in THIS contract, by category
          present = {c: nonempty_context(row, c)
                     for c in task1_categories if nonempty_context(row, c)}
          for category in task1_categories:
              if to_binary(row[f"{category}_Answer"]) == "Yes":
                  text, label = present[category], "Yes"
              else:
                  # hard negative: a real clause from a DIFFERENT category
                  others = [t for c, t in present.items() if c != category]
                  if not others:
                      continue          # skip if contract has no other clause to borrow
                  text, label = random.choice(others), "No"
              rows.append({
                  "instruction": f'Is the following contract text a "{category}" clause? Answer strictly "Yes" or "No".',
                  "category": category,
                  "input": text,
                  "output": label,
              })
      return rows
  ```
- **Why:** this is the whole point. Now `No` inputs are real legal text, so the
  model can't cheat on input shape and must actually learn what each clause type
  looks like. (Option B instead: set `input` to chunked full-contract text.)

### Step 4 — Keep the contract-level split
- **Where:** cell `0515b709` (`train_test_split(df, ...)` then build per side)
- **Change:** none — keep splitting `df` (contracts) *before* building examples.
- **Why:** already correct; prevents the same contract appearing in train and val.

### Step 5 — Balance the classes on the train split only
- **Where:** new cell, after building `train_data` (before saving)
- **Change:** downsample the majority class per category to ~1:1 or 3:1 on **train
  only**. Leave `val_data` untouched.
- **Why:** most categories are mostly one class; without balancing the model drifts
  to the majority answer. Validation must stay at the natural distribution so
  metrics reflect reality.

### Step 6 — Report real metrics, not just counts
- **Where:** replace the count-printing cell `step6code` (and add an eval after
  training)
- **Change:** compute **per-class precision / recall / F1 and a confusion matrix**
  on validation (e.g. `sklearn.metrics.classification_report`). Keep the Yes/No
  count print as a sanity line.
- **Why:** with imbalance, accuracy is meaningless ("always No" can score 80%+).
  P/R/F1 is the only honest signal that the model learned anything.

### Step 7 — Train on the answer only (completion-only loss)
- **Where:** cell `aaf3edf8` (the `SFTTrainer` setup)
- **Change:** mask the loss so only the text after `### Response:` is learned, e.g.
  `DataCollatorForCompletionOnlyLM(response_template="### Response:\n", tokenizer=...)`
  passed as `data_collator`.
- **Why:** right now loss is computed over the whole prompt (instruction + clause +
  answer). With a 1-token answer, the gradient is dominated by reproducing the
  clause text, diluting the actual Yes/No signal. Masking focuses learning on the
  decision.

### Step 8 — Minor training-config tidy-ups
- **Where:** cell `aaf3edf8`
- **Changes (optional):**
  - prefer `bf16=True` over `fp16=True` if the GPU supports it (more stable);
  - consider `target_modules=["q_proj","k_proj","v_proj","o_proj"]` (or all-linear)
    for a slightly stronger adapter;
  - lower `max_seq_length` (e.g. 1024) for Option A — single clauses are short, so
    this speeds training and saves memory.
- **Why:** small robustness/speed wins; none are blockers.

---

## How to know the fix worked

1. **Sanity check the data:** after Step 3, print a few `No` examples — their
   `input` should be *real clause text*, never the old placeholder string.
2. **The "dumb baseline" test:** a model that ignores the input should now score
   ~50%, not ~100%. If validation F1 is still near-perfect immediately, the leak
   isn't fully closed — re-inspect the inputs.
3. **Per-class F1 on validation** should be the headline number you quote, not loss
   or accuracy.

---

## Checklist

- [ ] Step 1 — load full CSV (510 contracts), sample only for smoke test
- [ ] Step 2 — leave load/clean/32-category cells unchanged
- [ ] Step 3 — `build_examples` uses **hard negatives** (real clause from another
      category) instead of the placeholder  ⭐
- [ ] Step 4 — split stays contract-level, before example building
- [ ] Step 5 — class balancing applied to **train only**
- [ ] Step 6 — report per-class precision/recall/F1 + confusion matrix on val
- [ ] Step 7 — completion-only loss (mask everything before `### Response:`)
- [ ] Step 8 — (optional) bf16, wider LoRA targets, lower `max_seq_length`
- [ ] Validation — `No` inputs are real text; dumb baseline ≈ 50%
