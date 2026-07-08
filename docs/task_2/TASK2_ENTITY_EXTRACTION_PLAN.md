# Task 2 (Structured Entity Extraction) — Objective & Notebook Pipeline Plan

This document breaks down **Task 2** and maps it onto a notebook pipeline with the
same shape as the Task 1 pipeline in
[llm_fine_tuning_LORA_task1_v2.ipynb](../../llm_fine_tuning_LORA_task1_v2.ipynb).
Each step is a single piece of functionality, explained simply, with an example of
how it could be implemented.

For the task definition itself — goal, exact input/output format with worked
examples, and the evaluation methodology — see [TASK2_LOGIC.md](TASK2_LOGIC.md).

---

## Part A — The objective, as simply as possible

**Task 2 = read a contract clause, pull out the specific fact inside it, and return
that fact as valid JSON.**

Where Task 1 asks a *yes/no question* ("does this contract contain a NonCompete
clause?"), Task 2 asks an *extraction question*:

> "Here is the clause about the agreement date. **What is the agreement date?**
> Answer in JSON."

The model's answer is not free text — it must be a machine-readable JSON object:

```json
{"Agreement Date": "5/8/2014"}
```

This matters because the whole point of contract-review automation is feeding the
answers into downstream software (databases, dashboards, alerts). Free-text answers
would need a second parsing step; JSON answers are directly usable. So Task 2 is
really testing **two skills at once**:

1. **Extraction accuracy** — did the model find the right value?
2. **Format discipline** — did it wrap that value in syntactically valid JSON with
   the right key?

### The 9 entity categories

CUAD's `master_clauses.csv` has 41 category/answer column pairs. Task 1 uses 32 of
them (the yes/no risk clauses). Task 2 uses the **other 9**, the ones whose answers
are *values* rather than yes/no:

| Category | What the answer looks like |
| :--- | :--- |
| `Document Name` | "MARKETING AFFILIATE AGREEMENT" |
| `Parties` | "Birch First Global Investments Inc.; Mount Kowledge Holdings Inc." |
| `Agreement Date` | "5/8/2014" |
| `Effective Date` | "5/8/2014" |
| `Expiration Date` | "12/31/2021" |
| `Renewal Term` | "successive 1 year terms" |
| `Notice Period To Terminate Renewal` | "30 days" |
| `Governing Law` | "Nevada" |
| `Warranty Duration` | "90 days" |

(`Filename` is also excluded from Task 1 but it is **metadata, not an entity** — it
never appears in clause text, so it is *not* a Task 2 category either.)

### Why this measures fine-tuning well

A base LLM already "knows English", but it does **not** reliably (a) restrict itself
to the exact normalized value, and (b) emit strict JSON with an exact key, every
single time. Fine-tuning on a few thousand (clause → JSON) pairs should move both
numbers sharply. Comparing **exact-match / JSON-validity before vs. after
fine-tuning** (same base model, same prompts — like
[llama_3.1_task_1_no_fine_tune.ipynb](../../llama_3.1_task_1_no_fine_tune.ipynb)
does for Task 1) is the measurement.

### Key data facts to respect (from the EDA docs)

- **Answers are expert-normalized** (e.g. "8th day of May" in the clause →
  `"5/8/2014"` in the answer column). The model must learn to *normalize*, not just
  copy — this is intentional and is what makes the task non-trivial.
- **Multi-value answers use semicolons** ("Party A; Party B"). Per
  [data_roles_cuad.md](../data_processing/data_roles_cuad.md) §3C, these should be
  emitted as a **JSON list of strings**, and the instruction must say so explicitly.
- **Some answer cells are bracketed list-strings** like `"['5/8/2014']"` — these
  must be parsed during preprocessing, never fed through verbatim.
- **Not every contract has every entity.** Empty answer cells mean "not found";
  the pipeline must decide what to do with them (Step 5 below teaches an explicit
  `null` instead of dropping them).
- **The split must be contract-level, 85/15** — same leakage rule as Task 1.

---

## Part B — The pipeline, step by step

Same skeleton as the Task 1 v2 notebook: *environment → load → clean → select
categories → build examples → split → sanity-check → save JSONL → QLoRA train →
evaluate*. Only the example-building, prompt, and evaluation steps genuinely differ
from Task 1.

### Step 0 — GPU pinning + environment config (reuse Task 1 cells verbatim)

**What it does:** Pins to a single GPU before torch is imported (Kaggle T4 x2 would
otherwise shard the model across GPUs) and sets `DATA_DIR` / `WORK_DIR` so the same
notebook runs unchanged locally or on Kaggle.

**Example:** copy cells 1 and 5 of the Task 1 v2 notebook unchanged:

```python
ON_KAGGLE = Path("/kaggle").exists()
if ON_KAGGLE:
    DATA_DIR = Path("/kaggle/input/cuad-master-clauses-cleaned")
    WORK_DIR = Path("/kaggle/working")
else:
    DATA_DIR = Path(os.getenv("DATA_DIR", "data")) / "CUAD_v1"
    WORK_DIR = Path(".")
```

### Step 1 — Load the cleaned CSV

**What it does:** Reads `master_clauses_cleaned.csv` into a DataFrame `df`. One row
= one contract; columns come in `<Category>` / `<Category>_Answer` pairs.

**Example:**

```python
df = pd.read_csv(DATA_DIR / "master_clauses_cleaned.csv")
print(f"{len(df)} contracts, {len(df.columns)} columns")
```

### Step 2 — Clean and rename columns (reuse Task 1 cells)

**What it does:** Normalizes column names so every answer column is exactly
`<Category>_Answer` and every context column is `<Category>`. Identical to Task 1 —
copy the `clean_text` / rename cells.

### Step 3 — Select the 9 Task 2 categories

**What it does:** Builds the list of entity categories this notebook trains on —
the mirror image of the Task 1 cell, which *excluded* these fields. `Filename` is
metadata and stays out. One assert catches schema drift immediately.

**Example:**

```python
task2_categories = [
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Warranty Duration",
]
for c in task2_categories:  # every category must have both columns
    assert c in df.columns and f"{c}_Answer" in df.columns, f"missing pair for {c}"
assert len(task2_categories) == 9
```

### Step 4 — Normalize raw answer cells into clean values

**What it does:** One function that turns whatever is in an `_Answer` cell into a
clean Python value the rest of the pipeline can trust. This is the Task 2
counterpart of Task 1's `to_binary()` — the single most important preprocessing
function in the notebook. It handles the three messy cases:

1. empty / NaN → `None` (entity not found),
2. bracketed list-strings `"['5/8/2014']"` → unwrap to `"5/8/2014"`,
3. semicolon-separated multi-values → list of strings.

**Example:**

```python
import ast

def normalize_answer(raw):
    """Raw _Answer cell -> None | str | list[str]."""
    if pd.isna(raw) or not str(raw).strip():
        return None
    s = str(raw).strip()
    # Case: bracketed list-string like "['5/8/2014']" or "['A', 'B']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                parsed = [str(p).strip() for p in parsed if str(p).strip()]
                if not parsed:
                    return None
                s = "; ".join(parsed)      # fall through to semicolon handling
        except (ValueError, SyntaxError):
            pass                            # not a real list-string; keep as-is
    # Case: multi-value -> JSON list; single value -> plain string
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return parts if len(parts) > 1 else parts[0]
```

### Step 5 — Build extraction examples (the Task 2 core)

**What it does:** Turns the contract table into individual training examples — one
per (contract, category) where a clause excerpt exists. Each example is: an
instruction naming the entity, the clause text as input, and the JSON answer as
output. Design decisions, stated explicitly:

- **Positive example** — context and answer both present: output is
  `{"<Category>": "<value>"}` (or a list, per Step 4).
- **"Not found" example** — context present but the normalized answer is `None`
  (rare, but the columns can disagree): output is `{"<Category>": null}`. This
  teaches the model to say "not there" in JSON instead of hallucinating a value —
  the Task 2 analogue of Task 1 keeping its "No" examples.
- **Skip** when there is **no context at all** — with no clause text there is
  nothing to extract from (same guard as Task 1's `present.get(category)` skip).
- The instruction states the JSON contract verbatim, including the list rule from
  the EDA docs.

**Example:**

```python
def nonempty_context(row, cat):
    v = row.get(cat)
    return str(v).strip() if pd.notna(v) and str(v).strip() else None

def build_examples(frame):
    rows = []
    for _, row in frame.iterrows():          # one contract at a time
        for category in task2_categories:    # one question per entity category
            context = nonempty_context(row, category)
            if not context:
                continue                     # nothing to extract from -> skip
            value = normalize_answer(row[f"{category}_Answer"])
            rows.append({
                "instruction": (
                    f'Extract the "{category}" from the contract text below. '
                    f'Return the result as a JSON object with the single key '
                    f'"{category}". If multiple values exist, return them as a '
                    f'list of strings. If the value is not present, use null.'
                ),
                "category": category,
                "input": context,
                # json.dumps handles str, list AND None -> null uniformly
                "output": json.dumps({category: value}, ensure_ascii=False),
            })
    return rows
```

### Step 6 — Split by contract, then build examples from each side

**What it does:** Exactly the Task 1 rule: split the **contracts** 85/15 first, then
explode each side into examples, so no contract's clauses appear in both sets.

**Example:**

```python
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
train_data = build_examples(train_df)
val_data   = build_examples(val_df)
print(f"Contracts — train: {len(train_df)}, val: {len(val_df)}")
print(f"Examples  — train: {len(train_data)}, val: {len(val_data)}")
```

> **No class balancing step.** Task 1 needed hard negatives + Yes/No balancing
> because it was a binary classifier. Task 2 has no majority class to collapse
> into — its analogue is the per-category example count check in Step 7.

### Step 7 — Sanity checks before saving

**What it does:** Two cheap checks that catch silent data bugs before an expensive
training run (the Task 2 version of Task 1's Step 6a):

1. **Per-category counts** — every one of the 9 categories should contribute
   examples; a zero means a renamed/missing column.
2. **Every output must round-trip through `json.loads`** — if a single training
   label is malformed JSON, the model is being *taught* to emit bad JSON.

**Example:**

```python
from collections import Counter

counts = Counter(e["category"] for e in train_data)
for c in task2_categories:
    print(f"{c:45s} {counts.get(c, 0):5d}")
    assert counts.get(c, 0) > 0, f"no examples for {c}"

for e in train_data + val_data:
    parsed = json.loads(e["output"])            # raises if malformed
    assert list(parsed.keys()) == [e["category"]]
print("All outputs are valid single-key JSON.")
```

### Step 8 — Save train/validation JSONL (save paths == load paths)

**What it does:** Writes both splits to the writable `WORK_DIR` (Kaggle's input dir
is read-only) with the same `save_jsonl` helper, under Task-2-specific filenames so
they never collide with the Task 1 artifacts.

**Example:**

```python
CUAD_TRAIN_PATH = WORK_DIR / "cuad" / "train"
CUAD_VALIDATION_PATH = WORK_DIR / "cuad" / "validation"
CUAD_TRAIN_PATH.mkdir(parents=True, exist_ok=True)
CUAD_VALIDATION_PATH.mkdir(parents=True, exist_ok=True)
save_jsonl(train_data, CUAD_TRAIN_PATH / "cuad_task2_train.jsonl")
save_jsonl(val_data,   CUAD_VALIDATION_PATH / "cuad_task2_validation.jsonl")
```

### Step 9 — Load model + QLoRA config (reuse Task 1 cell, two changes)

**What it does:** Loads `meta-llama/Meta-Llama-3.1-8B` in 4-bit NF4 on one GPU,
logs into HF (Kaggle Secret or local `.env`), and defines the same LoRA config
(`r=16`, `alpha=16`, targets `q/k/v/o_proj`). Only two Task 2 deltas:

- `new_model_name = "llama-3.1-8B-cuad-task2"` — separate adapter output.
- `MAX_SEQ_LENGTH` — keep 1024 to start, but note that Task 2 *completions* are
  JSON objects (tens of tokens), not a single `Yes`/`No` token; the token-length
  histogram from Step 7 counts can confirm 1024 still covers prompt + completion.

### Step 10 — Convert to prompt/completion with completion-only loss

**What it does:** Same trl 1.x mechanism as Task 1: map each example to `prompt`
(everything up to and including `### Response:\n`) + `completion` (the JSON
string), and let `SFTConfig(completion_only_loss=True)` mask the prompt so **only
the JSON answer contributes to the loss**. This is even more important for Task 2
than Task 1 — the loss signal concentrates entirely on producing the right JSON.

**Example:** identical to Task 1's cell; only the data differs:

```python
RESPONSE_TEMPLATE = "### Response:\n"

def to_prompt_completion(ex):
    prompt = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"{RESPONSE_TEMPLATE}"
    )
    return {"prompt": prompt, "completion": ex["output"]}
```

### Step 11 — Pre-flight checks, then train

**What it does:** Reuse the Task 1 pre-flight cell (verifies the collator actually
masks prompts, the dataset loaded, VRAM headroom, etc. — a misconfigured collator
fails *silently*), then `trainer.train()` with the same `SFTConfig`
(batch 1 × grad-accum 8, gradient checkpointing, bf16, `paged_adamw_32bit`,
1 epoch). Save the adapter under the Task 2 name.

**Example:**

```python
train_result = trainer.train()
trainer.save_model(str(WORK_DIR / new_model_name))
```

### Step 12 — Evaluate: JSON validity, exact match, and per-category scores

**What it does:** The step that differs most from Task 1. There is no
`Yes`/`No` to classify, so instead of precision/recall, generate the model's JSON
for every validation example and score **three things**:

1. **JSON validity rate** — does `json.loads` succeed and is the key right?
   (Measures format discipline.)
2. **Exact match (EM)** — after light normalization (lowercase, strip), is the
   predicted value identical to the gold value? (The headline metric per
   [data_roles_cuad.md](../data_processing/data_roles_cuad.md) §1B.)
3. **Token-level F1** — word-overlap between predicted and gold value, so
   near-misses ("State of Nevada" vs "Nevada") get partial credit.

Report all three **per category** — dates may be easy while `Parties` is hard, and
an aggregate would hide that. Persist everything to `WORK_DIR` as
`eval_metrics.json` / `eval_report.txt` exactly like Task 1, so
`kaggle kernels output` retrieves it.

**Example:**

```python
def predict_json(example):
    prompt = (f"### Instruction:\n{example['instruction']}\n\n"
              f"### Input:\n{example['input']}\n\n### Response:\n")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

def norm(v):
    if isinstance(v, list):
        return sorted(str(x).strip().lower() for x in v)
    return str(v).strip().lower() if v is not None else None

def token_f1(pred, gold):
    p, g = str(pred).lower().split(), str(gold).lower().split()
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)

results = []
for ex in val_data:
    raw = predict_json(ex)
    gold = json.loads(ex["output"])[ex["category"]]
    try:
        pred = json.loads(raw)[ex["category"]]
        valid = True
    except (json.JSONDecodeError, KeyError, TypeError):
        pred, valid = None, False
    results.append({
        "category": ex["category"],
        "json_valid": valid,
        "exact_match": valid and norm(pred) == norm(gold),
        "f1": token_f1(pred, gold) if valid and pred and gold else 0.0,
    })

# Aggregate per category, then overall
rdf = pd.DataFrame(results)
print(rdf.groupby("category")[["json_valid", "exact_match", "f1"]].mean())
print("\nOverall:", rdf[["json_valid", "exact_match", "f1"]].mean().to_dict())
```

### Step 13 (recommended) — Baseline comparison notebook

**What it does:** Mirrors `llama_3.1_task_1_no_fine_tune.ipynb`: run Step 12's
evaluation against the **base model without the adapter** on the same validation
JSONL. The fine-tuning claim of the research is the delta between the two runs —
especially on JSON validity, where base models fail most visibly.

---

## Part C — Checklist

- [ ] Environment + load + column-clean cells reused from Task 1 v2 (Steps 0–2)
- [ ] `task2_categories` = exactly 9 entity fields; `Filename` excluded (Step 3)
- [ ] `normalize_answer` handles NaN, bracketed list-strings, semicolons (Step 4)
- [ ] Outputs built with `json.dumps`; `null` taught for missing values; no-context rows skipped (Step 5)
- [ ] Split at **contract level** 85/15 before building examples (Step 6)
- [ ] All 9 categories contribute examples; every output round-trips `json.loads` (Step 7)
- [ ] Task-2-specific JSONL filenames; save paths == load paths (Step 8)
- [ ] Separate adapter name `llama-3.1-8B-cuad-task2`; completion-only loss on (Steps 9–11)
- [ ] Eval reports JSON validity + EM + token F1, **per category**, persisted to `WORK_DIR` (Step 12)
- [ ] Base-model (no fine-tune) run of the same eval for comparison (Step 13)
