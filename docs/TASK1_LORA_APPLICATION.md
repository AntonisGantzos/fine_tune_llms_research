# Task 1 (Risk Clause Recognition) — Insights & How to Apply Them to `llm_fine_tuning_LORA.ipynb`

This note distills **only the Task 1 findings** from
[CUAD_dataset_exploration.ipynb](CUAD_dataset_exploration.ipynb) and translates
them into concrete, ordered changes to
[llm_fine_tuning_LORA.ipynb](llm_fine_tuning_LORA.ipynb).

Task 1 = **binary clause identification**: given a contract clause excerpt,
decide whether a specific risk clause is present (`Yes`) or absent (`No`).

---

## Part A — What the exploration notebook established about Task 1

1. **Task 1 is exactly 32 Yes/No categories.** They are derived from `cuad`
   by taking every `<Category>` / `<Category>_Answer` pair and **excluding the
   Task 2 entity-extraction fields**: `Filename`, `Document Name`, `Parties`,
   `Agreement Date`, `Effective Date`, `Expiration Date`, `Renewal Term`,
   `Notice Period To Terminate Renewal`, `Governing Law`, `Warranty Duration`.
   The remaining 32 categories (`task1_categories`) are the only ones Task 1
   should train on.

2. **Two columns, two roles** (per category):
   - `<Category>` → **context**: the raw clause excerpt = the *model input*.
     `NaN`/empty when the clause was not found.
   - `<Category>_Answer` → **label**: `"No"` (or empty) ⇒ clause **absent**;
     anything else ⇒ clause **present**. This is the *model output*.

3. **The label is binary, but the raw value is not.** Answer cells contain
   `"No"`, `"Yes"`, or a raw excerpt/garbled fragment (dates leaking into
   `Parties`, bracketed list-strings, etc.). For Task 1 every non-`No`/non-empty
   value must be **normalized to `"Yes"`** — never fed through verbatim.

4. **Severe class imbalance.** Most risk clauses appear in well under ~10% of
   contracts (the sparsity bar chart). The model can score high by always
   predicting `No`, so negatives must be handled deliberately, not dropped.

5. **Context is empty exactly when the answer is `No`.** A negative example
   therefore has *no* excerpt to show the model. This is the single biggest
   trap when wiring up Task 1 inputs (see Step 4 below).

6. **Splitting must be contract-level (by row), not clause-level**, to prevent
   the same contract's style/text appearing in both train and validation.

---

## Part B — Step-by-step application to `llm_fine_tuning_LORA.ipynb`

### Step 1 — Fix the data path (cell `7a681f30`)
The load cell currently fails: `master_clauses_cleaned.csv` is read from
`data/cuad/` but the file isn't found. Confirm the cleaned CSV exists at
`data/cuad/master_clauses_cleaned.csv` (per `CLAUDE.md`) and that `CUAD_PATH`
points there. Nothing else downstream runs until `df` loads.

### Step 2 — Keep column cleaning/rename as-is (cells `a46465a4`, `db61ef7a`)
These produce the `<Category>` / `<Category>_Answer` naming the exploration
notebook relies on. No change needed — just be aware the cleaned names are
`NonCompete`, `RofrRofoRofn`, `NoSolicit Of Customers`, etc.

### Step 3 — Restrict to the 32 Task 1 categories (new cell, after `db61ef7a`)
Before building training rows, derive the Task 1 category list the same way the
exploration notebook does, so Task 2 fields never enter Task 1 training:

```python
task2_categories = [
    "Filename", "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Warranty Duration",
]
task1_categories = [
    col for col in df.columns
    if not col.endswith("_Answer") and col.strip() != ""
    and col not in task2_categories
    and f"{col}_Answer" in df.columns
]
assert len(task1_categories) == 32
```

### Step 4 — Rebuild the formatting loop as classification (replace cell `0515b709`)
The current loop is written for **extraction** and has three Task-1 bugs:
- it phrases every example as *"Extract the {category} … Return in JSON"*;
- it sets `input` to the **context excerpt**, which is **empty for every `No`
  example** → negatives carry no signal;
- it emits the raw answer text as `output` instead of `Yes`/`No`.

Rework it into a binary task. Each row contributes one example **per Task 1
category**, the input is the clause excerpt (or an explicit "no clause found"
placeholder for negatives), and the output is strictly `Yes`/`No`:

```python
def to_binary(answer):
    return "No" if (pd.isna(answer) or str(answer).strip().lower() == "no") else "Yes"

formatted_rows = []
for _, row in df.iterrows():
    for category in task1_categories:
        label = to_binary(row[f"{category}_Answer"])
        context = row.get(category)
        context = str(context).strip() if pd.notna(context) and str(context).strip() else "[No matching clause excerpt found in contract.]"
        formatted_rows.append({
            "instruction": f'Does this contract contain a "{category}" clause? Answer strictly "Yes" or "No".',
            "category": category,
            "input": context,
            "output": label,
        })
```

> Keep the `No` rows — do **not** `continue` past them. Per insight #4 the
> negatives are the majority class and the whole point of Task 1.

### Step 5 — Split by contract, before exploding to examples (still cell `0515b709`)
The current `train_test_split(formatted_rows, ...)` splits **per-clause**, which
leaks a contract across train/val (insight #6). Split the **contracts (`df`
rows) first**, then build examples from each side:

```python
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
train_data = build_examples(train_df)   # the Step-4 loop, factored into a function
val_data   = build_examples(val_df)
```

### Step 6 — (Recommended) Address class imbalance (optional cell before saving)
Because most categories are overwhelmingly `No`, consider either:
- **downsampling negatives** per category to, e.g., a 3:1 No:Yes ratio on the
  *train* split only (leave validation untouched so metrics stay honest), or
- leaving the data as-is but reporting **per-class precision/recall** rather
  than accuracy.
Either way, log the final `Yes`/`No` counts so the imbalance is visible.

### Step 7 — Make the save/load paths consistent (cells `12ff4335`, `6b34808f`, `aaf3edf8`)
The notebook **saves** to `data/cuad/train/cuad_train.jsonl` /
`data/cuad/validation/cuad_validation.jsonl` but the training cell **loads**
`data/train.jsonl` / `data/validation.jsonl`. Point `load_dataset` at the same
files that were written:

```python
dataset = load_dataset("json", data_files={
    "train":      str(CUAD_TRAIN_PATH / "cuad_train.jsonl"),
    "validation": str(CUAD_VALIDATION_PATH / "cuad_validation.jsonl"),
})
```
Also ensure the `train/` and `validation/` directories exist before saving.

### Step 8 — Keep the prompt template; outputs are now tiny (cell `aaf3edf8`)
The `### Instruction / ### Input / ### Response` formatting func is fine. Two
Task-1-specific notes:
- Responses are now just `Yes`/`No`, so the model learns a crisp decision head.
- `max_seq_length=2048` is comfortably more than enough — Task 1 inputs are
  single clause excerpts, so most examples are short (the exploration notebook's
  token-length analysis showed clauses well under 512 tokens). You could lower
  `max_seq_length` to speed up training if memory is tight.

---

## Part C — Checklist

- [ ] CSV path fixed; `df` loads (Step 1)
- [ ] `task1_categories` derived = 32, Task 2 fields excluded (Step 3)
- [ ] Instruction reframed as Yes/No; output is `Yes`/`No` (Step 4)
- [ ] Negatives kept; empty context replaced with explicit placeholder (Step 4)
- [ ] Split done on **contracts** before building examples (Step 5)
- [ ] Class imbalance handled or at least reported (Step 6)
- [ ] Save paths == load paths (Step 7)
