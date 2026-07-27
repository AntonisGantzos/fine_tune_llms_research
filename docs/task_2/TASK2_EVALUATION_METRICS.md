# Task 2 — Evaluation Metrics: Full Logic Breakdown

This document explains **why** the Task 2 evaluation is built the way it is and **exactly how**
each metric is computed, edge case by edge case. The implementing code lives in the final
cell of [llm_fine_tuning_LORA_task2.ipynb](../../llm_fine_tuning_LORA_task2.ipynb)
("Evaluate on validation"). For task-level context (prompt format, the 9 categories, output
cases A–D) see [TASK2_LOGIC.md](TASK2_LOGIC.md); this doc is the metrics deep-dive.

---

## 0. What is actually being scored

**Training data.** CUAD `master_clauses_cleaned.csv` → `cuad/train/cuad_task2_train.jsonl` +
`cuad/validation/cuad_task2_validation.jsonl`, one example per (contract, category) pair across the
**9 entity categories**: Document Name, Parties, Agreement Date, Effective Date, Expiration Date,
Renewal Term, Notice Period To Terminate Renewal, Governing Law, Warranty Duration. Split is
**contract-level 85/15** — no clause from a validation contract was seen in training. The fine-tuned
run trained on **2,454 examples** (307 optimizer steps × effective batch 8, one epoch) and is scored
on **631 validation examples**.

**Input the model sees at eval time** — byte-identical to the training prompt, response left empty:

```
### Instruction:
Extract the "Governing Law" from the contract text below. Return the result as a JSON object
with the single key "Governing Law". If multiple values exist, return them as a list of strings.
If the value is not present, use null.

### Input:
<clause text>

### Response:
```

**Output it is expected to produce** — the completion it was fine-tuned on: one JSON object, no
markdown fence, no prose, whose only key is the requested category:

```json
{"Governing Law": "Nevada"}
```

with a list value when several entities exist (`{"Parties": ["Acme Inc.", "Beta LLC"]}`) and
`null` when the entity is absent (`{"Expiration Date": null}`). Teaching that output *contract* is
most of what fine-tuning buys here — the base model reliably wraps its answer in ` ```json ` fences
or explanatory text, which is precisely what metric 1 below counts.

**How the raw completion becomes a prediction.** Greedy generation, `max_new_tokens=128`, then
`json.loads` on the raw string with a strict key check. Unlike Task 1's lenient `Yes`/`No` coercion,
nothing here is repaired: a fenced or chatty completion fails parsing and scores 0 on all three
metrics.

---

## 1. Why Task 1's metrics don't transfer

Task 1 is binary classification: the completion is literally `Yes` or `No`, so
precision/recall/F1 over two classes describe it completely.

Task 2 is **generative structured extraction**: the model must emit a JSON object like

```json
{"Governing Law": "Nevada"}
```

There is no fixed label set, so "accuracy" alone is meaningless — a prediction can fail in
three qualitatively different ways, and the research question needs them separated:

| Failure mode | Example | What it tells us |
|---|---|---|
| **Broken format** | ` ```json {"Governing Law": "Nevada"} ``` ` or prose around the JSON | The model didn't learn the output *contract* |
| **Wrong content** | `{"Governing Law": "Delaware"}` when gold is Nevada | The model didn't learn the *extraction* |
| **Almost-right content** | `{"Governing Law": "the laws of the State of Nevada"}` | The model extracts but doesn't *normalize* |

Fine-tuning is expected to improve each of these differently (format most, extraction
somewhat, normalization in between), so the evaluation scores each example on **three
metrics in a deliberate order**, where each metric only makes sense if the previous one
passed.

---

## 2. The evaluation pipeline

For every example in the held-out validation set (contract-level split — no clause from a
validation contract was seen in training):

```
prompt = training template with empty response  →  greedy generation  →  parse  →  score 3 metrics
```

**Generation settings** (`predict_json`):

- The prompt is the *identical* `### Instruction / ### Input / ### Response:` template used
  in training, ending right after `### Response:\n` — the model completes from exactly the
  position it was trained to complete from.
- `do_sample=False` (greedy decoding). Deterministic: the same model + data always produce
  the same metrics, so a fine-tuned vs. baseline delta can't be sampling noise.
- `max_new_tokens=128` — generous for the longest legitimate output (a multi-party list),
  but bounds runaway generations.
- Only the tokens *after* the prompt are decoded (`out[0][prompt_len:]`), so the score never
  accidentally includes the prompt itself.

**Gold value:** parsed from the example's training-format output,
`json.loads(ex["output"])[ex["category"]]` — so gold is a real Python value
(string, list of strings, or `None`), not a raw JSON string. Predictions and gold are
compared **as values**, never as raw text.

---

## 3. Metric 1 — JSON validity (the gate)

**Question answered:** *did the model obey the output contract at all?*

Implemented by `parse_prediction(raw_text, category)`. The raw completion is valid **iff
both** conditions hold:

1. `json.loads(raw_text)` succeeds on the *entire* raw string, and
2. the result is a `dict` whose key set is **exactly** `{category}` — the one requested
   category, nothing more, nothing less, spelled and capitalized exactly.

Everything else is invalid, deliberately including things a human would forgive:

| Output | Why invalid |
|---|---|
| ```` ```json\n{"Governing Law": "Nevada"}\n``` ```` | Markdown fencing — `json.loads` fails on the whole string |
| `The answer is {"Governing Law": "Nevada"}` | Leading prose — not pure JSON |
| `"Nevada"` or `Nevada` | Bare value, not an object |
| `{"governing_law": "Nevada"}` | Wrong key (case/format) — key must be exactly `"Governing Law"` |
| `{"Governing Law": "Nevada", "Confidence": 0.9}` | Extra key |
| `{}` | Missing the requested key |

**Why so strict?** The premise of Task 2 is that the output feeds a downstream program
(`json.loads` + a dict lookup by category name). Any output that program couldn't consume
without extra repair code is a failure *of the thing being measured*. If the evaluator
stripped markdown fences before parsing, it would report format discipline the model
doesn't have — and JSON validity is precisely the metric where fine-tuning is expected to
show its largest gain over the base model, so it must not be diluted.

**The gating rule:** an invalid example scores `exact_match = False` and `f1 = 0.0` by
definition — content is never inspected inside an unparseable output. This produces a
per-example invariant used for sanity-checking any report:

```
exact_match  ≤  token_f1  ≤  json_valid        (per example, hence also per aggregate)
```

(`exact_match = 1` forces `token_f1 = 1` because both use the same normalization;
`json_valid = 0` forces both others to 0.)

---

## 4. Metric 2 — Exact match (headline accuracy)

**Question answered:** *of well-formed outputs, is the extracted value exactly right?*

```python
exact_match = valid and norm(pred) == norm(gold)
```

`norm(v)` applies **light normalization only**:

- `None` stays `None`;
- a scalar becomes `str(v).strip().lower()`;
- a list becomes a **sorted tuple** of its normalized items — list comparison is
  order-insensitive, because the order of e.g. contract parties carries no meaning.

What normalization deliberately does **not** do:

- **No fuzzy/substring matching.** `"State of Nevada"` ≠ `"Nevada"`. The training data
  teaches a normalized target form; producing a different (even semantically equivalent)
  form is a normalization failure, and exact match is the metric that detects it. Partial
  credit is Metric 3's job, not this one's.
- **No null leniency.** A gold value of `null` (entity absent from the clause) is matched
  *only* by a predicted `null`. Predicting any value for an absent entity is a
  **hallucination** — the single most damaging failure mode for a legal-review tool — and
  scores 0. Symmetrically, predicting `null` when a value exists is a miss.
- **No type coercion between scalar and list.** `norm("Acme")` is a string,
  `norm(["Acme"])` is a 1-tuple — they don't compare equal. The training data is consistent
  about when a category yields a list, so the model is expected to be too.

---

## 5. Metric 3 — Token-level F1 (partial credit)

**Question answered:** *when the value isn't exactly right, how close is it?*

Two functions implement it: `token_f1` for a pair of scalars, and `value_f1` extending it
to `None` and lists.

### 5.1 Scalar case — `token_f1` (SQuAD-style)

Both strings are lowercased and whitespace-split into bags of words; overlap is counted
with a multiset intersection (`Counter(p) & Counter(g)`), so a repeated word only matches
as many times as it appears in both:

```
precision = overlap / len(pred_tokens)
recall    = overlap / len(gold_tokens)
F1        = 2 · precision · recall / (precision + recall)      (0.0 if overlap == 0)
```

Worked example — gold `"State of Nevada"`, prediction `"the laws of the State of Nevada"`:

- gold tokens `{state, of, nevada}` (3), pred tokens (7: the·2, laws, of·2, state, nevada)
- overlap = 3 (`state`, `of`×1 — capped by gold's single `of` — `nevada`)
- precision = 3/7, recall = 3/3 = 1.0 → **F1 ≈ 0.60**

So the model gets substantial credit for finding the right span while being penalized for
not normalizing it — exactly the "almost-right content" failure mode from §1, now
quantified instead of collapsed to 0.

Known, accepted limitations: it's bag-of-words (word order is ignored) and has no notion of
semantics — `"NY"` vs `"New York"` scores 0. That's acceptable because the training data
fixes one canonical surface form per value, so surface overlap tracks correctness here.

### 5.2 `None` handling

| Gold | Pred | `value_f1` | Meaning |
|---|---|---|---|
| `None` | `None` | **1.0** | Correctly said "not there" — full credit |
| `None` | any value | **0.0** | Hallucination — no partial credit ever |
| any value | `None` | **0.0** | Missed a present entity |

Hallucinations get zero even if the invented value shares tokens with something in the
clause — partial credit for making things up would be perverse.

### 5.3 List case — greedy one-to-one alignment

For lists (e.g. `Parties`), scalars are first wrapped into 1-element lists, then:

1. For each **gold** item in order, find the **unused** predicted item with the highest
   `token_f1`; mark it used (one-to-one: a predicted item can't match two gold items).
2. A gold item with no unused prediction left scores 0.
3. Final score = `sum(per-gold-item scores) / max(len(pred_list), len(gold_list))`.

The `max(...)` denominator is what penalizes **both directions of length mismatch**:
missing gold items score 0 in the numerator, and *extra* predicted items inflate the
denominator — so padding the answer with plausible-looking parties costs score rather
than gaming it.

Worked example — gold `["Acme Corp", "Beta LLC"]`, prediction `["Acme Corporation"]`:

- `"Acme Corp"` ↔ `"Acme Corporation"`: overlap 1 (`acme`), precision ½, recall ½ → 0.5
- `"Beta LLC"`: no predictions left → 0.0
- score = (0.5 + 0.0) / max(1, 2) = **0.25**

The alignment is greedy in gold order, not globally optimal (a Hungarian-algorithm
assignment could score slightly higher in contrived cases). That's a deliberate
simplicity/fidelity trade-off: lists here are short (typically 2–4 parties), items are
usually distinct, and any bias is identical for the fine-tuned and baseline runs — and the
comparison between those two runs is the deliverable.

---

## 6. Aggregation and reporting

Each example yields `{category, json_valid, exact_match, f1}`. Aggregation:

- **Per category** — mean of each metric over that category's examples, reported for all
  9 categories in canonical order (`reindex(task2_categories)`, so a category with zero
  validation examples still shows up, as NaN, instead of silently vanishing).
- **Overall** — mean over **all examples** (micro-average). Categories with more
  validation examples weigh more; this represents "expected score on a random extraction
  request." It is *not* the mean of the 9 per-category rows (macro-average) — check
  `per_category_counts` in the JSON before comparing overall numbers across runs with
  different data.

Per-category reporting is non-negotiable for this task: dates (`Agreement Date`,
`Expiration Date`) are short, highly-templated, and easy; `Parties` is multi-valued
free text and hard. A single aggregate would average an easy 0.9 against a hard 0.4 and
hide exactly the structure the research write-up needs.

### Persisted artifacts (same convention as Task 1)

Both files are written to `WORK_DIR` (`/kaggle/working` on Kaggle) so
`kaggle kernels output` / `run.ps1 -Wait` pulls them into `kaggle_output/`:

- **`eval_metrics.json`** — machine-readable: model name, task tag,
  `n_validation_examples`, `overall` (the 3 means), `per_category` (3 means × 9
  categories), and `per_category_counts` (examples per category, for judging statistical
  weight — a 0.9 over 12 examples is weaker evidence than a 0.7 over 200).
- **`eval_report.txt`** — the human-readable fixed-width table (9 category rows +
  `OVERALL`), for quick eyeballing without loading the JSON.

---

## 7. Reading the results: fine-tuned vs. baseline

The identical evaluation runs in a separate no-fine-tune notebook against the same
validation set (mirroring the T1 baseline setup). The **deltas** are the deliverable, and
each metric isolates one layer of what fine-tuning taught:

| Metric | Expected pattern | Interpretation |
|---|---|---|
| **JSON validity** | large gain (base models fence/chat/add keys) | fine-tuning taught the *format contract* |
| **Exact match** | substantial gain | taught extraction + the canonical *normalized form* |
| **Token F1** | smaller gain, and gap vs. EM narrows | base model often finds the right span; fine-tuning mostly cleans it up |

Diagnostic gaps worth reading directly off the table:

- **`json_valid` − `token_f1`** ≈ "well-formed but wrong content" mass.
- **`token_f1` − `exact_match`** ≈ "right span, wrong normalization" mass (the F1 > EM
  region is populated almost entirely by un-normalized extractions).
- A category where the *baseline* already scores high on F1 but low on validity means the
  base model knows the content and only needed format training — evidence for the thesis
  that QLoRA fine-tuning primarily buys format/normalization discipline.

One caution when comparing runs: because invalid outputs are hard-zeroed on EM and F1, a
model with low JSON validity has its content metrics dragged down mechanically. If you
want "content quality among valid outputs," divide `exact_match` by `json_valid` (both are
means over the same denominator, so the ratio is EM *conditional on validity*).
