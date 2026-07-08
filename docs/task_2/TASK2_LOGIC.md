# Task 2 (Structured Entity Extraction) — Task Logic, I/O Format & Evaluation

This document defines **what Task 2 is**, independent of how the notebook pipeline
is built. For the step-by-step implementation plan see
[TASK2_ENTITY_EXTRACTION_PLAN.md](TASK2_ENTITY_EXTRACTION_PLAN.md).

---

## 1. The Goal

**Given a contract clause, extract one specific fact from it and return that fact
as a valid JSON object.**

A human contract reviewer does this constantly: they skim a clause and jot down
"governing law: Nevada" or "expires: 12/31/2021" into a review sheet. Task 2
teaches the model to do exactly that, with two hard requirements:

1. **The value must be correct** — the model must find and *normalize* the fact,
   not quote the surrounding sentence. If the clause says *"...executed on the 8th
   day of May, 2014"*, the correct answer is `"5/8/2014"`, because that is how
   CUAD's expert annotators normalized it in the ground-truth column.
2. **The container must be machine-readable** — the answer must be a
   syntactically valid JSON object whose single key is the category being asked
   about. Anything else (prose, markdown, a bare value, extra keys) is a format
   failure even if the value inside is right.

Contrast with Task 1:

| | Task 1 (Risk Clause Recognition) | Task 2 (Entity Extraction) |
| :--- | :--- | :--- |
| Question type | "Is this text a *X* clause?" | "What is the *X* in this text?" |
| Output space | Exactly `Yes` or `No` | Open-ended value wrapped in JSON |
| Skill measured | Clause understanding | Extraction + normalization + format discipline |
| Categories | 32 risk-clause columns | 9 entity columns |
| Failure looks like | Wrong class | Wrong value, unparseable output, or hallucinated value |

### The 9 entity categories

`Document Name`, `Parties`, `Agreement Date`, `Effective Date`, `Expiration Date`,
`Renewal Term`, `Notice Period To Terminate Renewal`, `Governing Law`,
`Warranty Duration`.

Each has a `<Category>` column (the clause excerpt = model input) and a
`<Category>_Answer` column (the expert-normalized value = ground truth) in
`master_clauses_cleaned.csv`.

### Why measure fine-tuning with this task

A base (non-fine-tuned) model given the same prompt typically fails in ways that
have nothing to do with legal understanding: it explains its answer in prose,
copies the whole clause, invents its own JSON schema, or answers correctly but
un-normalized. Fine-tuning on (clause → JSON) pairs should collapse all of that
into the one exact format. So the before/after comparison isolates what
fine-tuning actually bought: **format reliability and normalization**, on top of
any accuracy gain.

---

## 2. Input / Output Format

Every example is one question about one category in one clause. The model sees a
prompt in the same `### Instruction / ### Input / ### Response` template used by
Task 1, and must produce **only** the JSON object after `### Response:`.

### The prompt template

```
### Instruction:
Extract the "{Category}" from the contract text below. Return the result as a
JSON object with the single key "{Category}". If multiple values exist, return
them as a list of strings. If the value is not present, use null.

### Input:
{clause excerpt from the <Category> column}

### Response:
{"{Category}": ...}        ← the model generates only this line
```

The instruction spells out all three output rules (single key, list for
multi-values, `null` for absent) so the format contract is explicit in every
example — the model is never left to guess the schema.

### Case A — Single value (the common case)

The clause contains exactly one value for the category.

**Input (Governing Law):**
```
### Instruction:
Extract the "Governing Law" from the contract text below. Return the result as a
JSON object with the single key "Governing Law". If multiple values exist, return
them as a list of strings. If the value is not present, use null.

### Input:
This Agreement is to be construed according to the laws of the State of Nevada.

### Response:
```

**Expected output:**
```json
{"Governing Law": "Nevada"}
```

Note the normalization: the gold answer is `"Nevada"`, not
`"the laws of the State of Nevada"`. The model must learn to reduce the span to
the annotated value.

### Case B — Normalized date

**Input (Agreement Date):**
```
### Input:
THIS AGREEMENT is made and entered into as of the 8th day of May, 2014, by and
between the parties identified below.
```

**Expected output:**
```json
{"Agreement Date": "5/8/2014"}
```

The written-out date in the clause maps to the `M/D/YYYY` string in the answer
column. This is the clearest example of *extraction + normalization* rather than
copying.

### Case C — Multiple values → JSON list

Answer cells with several values (semicolon-separated in the CSV) become a JSON
**list of strings**.

**Input (Parties):**
```
### Input:
...by and between Birch First Global Investments Inc., a corporation organized
under the laws of the U.S. Virgin Islands ("Company") and Mount Knowledge
Holdings Inc., a corporation incorporated in Nevada ("Marketing Affiliate").
```

**Expected output:**
```json
{"Parties": ["Birch First Global Investments Inc.", "Mount Knowledge Holdings Inc."]}
```

### Case D — Value not present → null

When a clause excerpt exists but the annotators recorded no value for the
category, the correct answer is JSON `null` — the model must say "not there"
in-format instead of hallucinating a plausible value.

**Expected output:**
```json
{"Warranty Duration": null}
```

### Outputs that count as WRONG even when "close"

| Model output | Why it fails |
| :--- | :--- |
| `The governing law is Nevada.` | Not JSON at all (validity failure) |
| `Nevada` | Bare value, no JSON object (validity failure) |
| `{"governing_law": "Nevada"}` | Wrong key — must be exactly `"Governing Law"` |
| `{"Governing Law": "Nevada", "confidence": 0.9}` | Extra key; schema is single-key |
| `{"Governing Law": "the laws of the State of Nevada"}` | Valid JSON, wrong (un-normalized) value — scores 0 on exact match, partial on F1 |
| ```` ```json {"Governing Law": "Nevada"} ``` ```` | Markdown fencing around the JSON (validity failure unless stripped before parsing — the evaluator must not silently forgive it, since downstream consumers wouldn't) |

---

## 3. Evaluating the Fine-Tuned Model

Generation is greedy (`do_sample=False`) with `max_new_tokens=128`, on the
held-out **validation contracts** (contract-level 85/15 split — no clause from a
validation contract was ever seen in training). Every validation example is scored
on three metrics, in a deliberate order: each metric only makes sense if the
previous one passed.

### Metric 1 — JSON validity rate (format discipline)

**Question:** can a machine consume the output at all?

An output is *valid* iff `json.loads` succeeds **and** the result is an object
whose only key is exactly the requested category. Everything downstream depends on
this — an invalid output scores 0 on the remaining metrics by definition.

```python
def parse_prediction(raw_text, category):
    """Returns (value, is_valid). Invalid -> (None, False)."""
    try:
        obj = json.loads(raw_text)
        if not isinstance(obj, dict) or set(obj.keys()) != {category}:
            return None, False
        return obj[category], True
    except json.JSONDecodeError:
        return None, False
```

This is the metric where base vs. fine-tuned should differ most dramatically: a
base model may be valid on a minority of outputs, a fine-tuned one should be near
100%.

### Metric 2 — Exact Match (headline accuracy)

**Question:** is the extracted value *identical* to the expert-annotated value?

Compared after light normalization only — lowercase and whitespace-strip; for
lists, compare as order-insensitive sets (the order of parties is not meaningful).
No fuzzy matching: `"State of Nevada"` ≠ `"Nevada"`.

```python
def norm(v):
    if v is None:
        return None
    if isinstance(v, list):
        return tuple(sorted(str(x).strip().lower() for x in v))
    return str(v).strip().lower()

exact_match = valid and norm(pred_value) == norm(gold_value)
```

A `null` gold answer is only matched by a predicted `null` — predicting any value
for an absent entity is a miss (that's a hallucination, the exact failure mode
Case D trains against).

### Metric 3 — Token-level F1 (partial credit)

**Question:** when the value isn't an exact match, how close is it?

Word-overlap F1 between predicted and gold strings, standard from SQuAD-style
evaluation. It distinguishes "almost right" from "completely wrong":

| Gold | Prediction | EM | Token F1 |
| :--- | :--- | :---: | :---: |
| `Nevada` | `Nevada` | 1 | 1.00 |
| `Nevada` | `State of Nevada` | 0 | 0.50 |
| `successive 1 year terms` | `1 year terms` | 0 | 0.86 |
| `Nevada` | `Delaware` | 0 | 0.00 |
| any value | *(invalid JSON)* | 0 | 0.00 |

```python
def token_f1(pred, gold):
    p, g = str(pred).lower().split(), str(gold).lower().split()
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)
```

For list-valued answers, align each predicted item to its best-matching gold item
and average; unmatched items on either side count as 0.

### Report per category, then overall

The 9 categories are not equally hard — dates are short and highly normalized,
`Parties` is long and multi-valued, `Renewal Term` is free-ish text. A single
aggregate hides this, so the report is a 9-row table plus an overall row:

```
Category                                json_valid   exact_match   token_f1
Agreement Date                              1.00         0.91        0.95
Parties                                     0.99         0.62        0.81
Governing Law                               1.00         0.88        0.93
...
────────────────────────────────────────────────────────────────────────
OVERALL                                     0.99         0.78        0.88
```

Persist the numbers as `eval_metrics.json` + `eval_report.txt` in `WORK_DIR`
(same convention as Task 1) so Kaggle runs return them as artifacts.

### The measurement that answers the research question

Run the **identical evaluation twice** — once on the base
`meta-llama/Meta-Llama-3.1-8B` with no adapter (mirroring
[llama_3.1_task_1_no_fine_tune.ipynb](../../llama_3.1_task_1_no_fine_tune.ipynb)),
once on the fine-tuned adapter — same prompts, same validation JSONL, same
metrics. The deliverable is the delta table:

| Metric | Base model | Fine-tuned | Δ |
| :--- | :---: | :---: | :---: |
| JSON validity | *(measured)* | *(measured)* | |
| Exact match | *(measured)* | *(measured)* | |
| Token F1 | *(measured)* | *(measured)* | |

Expected shape of the result: the largest gain on JSON validity (pure format
discipline), a substantial gain on exact match (normalization behavior), a smaller
gain on token F1 (the base model often finds roughly the right span but packages
it wrong). If instead JSON validity barely moves, the fine-tuning run — not the
model — should be suspected first (e.g. completion-only loss misconfigured).
