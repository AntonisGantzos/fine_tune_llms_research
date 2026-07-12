# Task 2 Notebook — Cell-by-Cell Guide

What every cell in [llm_fine_tuning_LORA_task2.ipynb](../../llm_fine_tuning_LORA_task2.ipynb)
does, in order, as simply as possible.

Background docs: [TASK2_LOGIC.md](TASK2_LOGIC.md) (what the task is),
[TASK2_ENTITY_EXTRACTION_PLAN.md](TASK2_ENTITY_EXTRACTION_PLAN.md) (why the pipeline
is shaped this way).

**The pipeline in one sentence:** load the CUAD contract table → turn it into
(clause → JSON answer) training examples → fine-tune Llama 3.1 8B with QLoRA so it
learns to answer extraction questions in strict JSON → measure how well it does.

---

## Stage 1 — Environment (cells 0–2)

### Cell 0 (markdown) — Title
What Task 2 is: given a clause, extract one fact and return it as single-key JSON,
e.g. `{"Agreement Date": "5/8/2014"}`. Same base model and skeleton as Task 1.

### Cell 1 (code) — Pin one GPU + Kaggle installs
Two things, both **before** torch is imported:
1. `CUDA_VISIBLE_DEVICES=0` — hides Kaggle's second T4. Otherwise the model gets
   sharded across 2 GPUs and trl's loss computation crashes with a device mismatch.
2. If running on Kaggle, pip-installs the exact library versions the notebook needs
   (`transformers==4.55.4`, `trl==0.20.0`, etc.). Does nothing locally.

### Cell 2 (code) — UTF-8 check
Prints whether Python is in UTF-8 mode. Purely informational.

---

## Stage 2 — Load and clean the data (cells 3–10)

### Cell 3 (markdown) — Section header.

### Cell 4 (code) — Imports
pandas, json, pathlib, csv, re, `train_test_split`.

### Cell 5 (code) — Where is the data, where do outputs go
Sets two paths so the notebook runs unchanged locally or on Kaggle:
- `DATA_DIR` — where the CSV lives (Kaggle input dataset, or local `data/CUAD_v1`).
- `WORK_DIR` — the only writable/persisted dir (`/kaggle/working`, or `.` locally).

### Cell 6 (code) — Load the CSV
Reads `master_clauses_cleaned.csv` into a DataFrame. One row = one contract;
columns come in pairs: `<Category>` (clause text) + `<Category>-Answer` (gold value).
If the file isn't at the expected path (Kaggle mounts can vary), it searches
`/kaggle/input` for it before giving up.

### Cell 7 (code) — Peek
`df.head(3)` — eyeball the loaded table.

### Cell 8 (code) — Clean column names
Strips special characters from every column name (`"Parties-Answer"` → `"PartiesAnswer"`).

### Cell 9 (code) — Rename answer columns
`"PartiesAnswer"` → `"Parties_Answer"`, so every pair is uniformly
`<Category>` / `<Category>_Answer`.

### Cell 10 (code) — Print the final column names to confirm the rename worked.

---

## Stage 3 — Build the Task 2 dataset (cells 11–20)

### Cell 11 (markdown) — Section header.

### Cell 12 (code) — Pick the 9 entity categories
The exact list Task 1 *excluded*: `Document Name`, `Parties`, `Agreement Date`,
`Effective Date`, `Expiration Date`, `Renewal Term`,
`Notice Period To Terminate Renewal`, `Governing Law`, `Warranty Duration`.
(`Filename` is file metadata, not an entity — left out.) Asserts each category has
both its columns, so a schema change fails loudly here instead of silently later.

### Cell 13 (code) — `save_jsonl` helper
Writes a list of dicts to a file, one JSON object per line.

### Cell 14 (markdown) — Section header.

### Cell 15 (code) — `normalize_answer`: raw answer cell → clean value
The most important preprocessing function. Turns whatever is in an `_Answer` cell
into one of three clean shapes:

| Raw cell | Becomes |
| :--- | :--- |
| empty / NaN | `None` (entity not found) |
| `"['5/8/2014']"` (bracketed list-string) | `"5/8/2014"` |
| `"Party A; Party B"` (semicolons) | `["Party A", "Party B"]` |
| `"Nevada"` | `"Nevada"` |

Ends with asserts that self-test exactly these cases.

### Cell 16 (markdown) — Section header.

### Cell 17 (code) — Build examples + split by contract
The core cell. For every (contract, category) pair:
- **No clause text** → skip (nothing to extract from).
- **Clause text + answer** → example with output `{"<Category>": "<value>"}`
  (or a JSON list).
- **Clause text but no answer** → example with output `{"<Category>": null}` —
  teaches the model to say "not there" in JSON instead of inventing a value.

Every output is built with `json.dumps`, so labels are valid JSON by construction.
The **contracts are split 80/20 first**, then examples are built from each side —
so no contract's clauses can leak into both train and validation.
Result: ~2,600 train / ~480 val examples from 510 contracts.

### Cell 18 (code) — Show one example of each shape
Prints one single-value, one list-value, and one null example, so you can visually
confirm all three output shapes exist and look right.

### Cell 19 (markdown) — Section header.

### Cell 20 (code) — Sanity checks before spending GPU time
1. Per-category example counts — asserts every one of the 9 categories contributes
   to both splits (a zero = a renamed/missing column).
2. Round-trips every output through `json.loads` and asserts its only key is the
   category — a malformed label would *teach* the model to emit bad JSON.

There is **no class-balancing step** (unlike Task 1): there's no majority class here.

---

## Stage 4 — Save the dataset (cells 21–23)

### Cell 21 (markdown) — Section header.

### Cell 22 (code) — Create output dirs
`WORK_DIR/cuad/train` and `WORK_DIR/cuad/validation` (Kaggle's input dir is
read-only, so generated files must go to `WORK_DIR`).

### Cell 23 (code) — Write the JSONL files
`cuad_task2_train.jsonl` + `cuad_task2_validation.jsonl` — Task-2-specific names so
they never collide with the Task 1 files.

---

## Stage 5 — QLoRA fine-tuning (cells 24–30)

### Cell 24 (markdown) — What's different from Task 1
Only two deltas: adapter is saved as `llama-3.1-8B-cuad-task2`, and completions are
small JSON objects instead of a single Yes/No token.

### Cell 25 (markdown) — Reminder to set `HF_TOKEN` before running locally.

### Cell 26 (code) — Token diagnostic
Checks whose HuggingFace account the token belongs to and whether that account has
access to the gated `meta-llama/Meta-Llama-3.1-8B` repo — so a 403 is explained in
seconds, not mid-download.

### Cell 27 (code) — Load everything for training
One big setup cell:
1. **HF login** (Kaggle Secret or local `.env`).
2. **Base model** — Llama 3.1 8B in 4-bit NF4 (~5–6 GB, fits one T4), all on GPU 0.
3. **Tokenizer** — pad = EOS, right padding.
4. **Dataset** — loads the two JSONL files saved in cell 23.
5. **Prompt/completion mapping** — each example becomes
   `prompt` = `### Instruction / ### Input / ### Response:` template,
   `completion` = the JSON answer. With `completion_only_loss=True`, trl masks all
   prompt tokens, so **only the JSON answer contributes to the loss**.
6. **LoRA config** — r=16, alpha=16, targets `q/k/v/o_proj`.
7. **SFTConfig** — batch 1 × grad-accum 8, gradient checkpointing, bf16,
   `paged_adamw_32bit`, `max_length=1024`, 1 epoch.
8. **SFTTrainer** — ties it all together.

### Cell 28 (markdown) — Section header.

### Cell 29 (code) — Pre-flight checks (run before training, always)
Asserts everything that could waste an hour-long run, in seconds:
GPU present with enough VRAM; model really is 4-bit; tokenizer padding correct;
both dataset splits present and **every completion parses as single-key JSON**;
**the completion-only masking actually works** (pulls one real batch and checks
some tokens are masked and some aren't — a broken collator fails silently with
loss ≡ 0); token lengths fit in 1024; LoRA attached with <5% trainable params.

### Cell 30 (code) — Train and save
`trainer.train()`, then:
- asserts the final loss is a real non-zero number (loss = 0 means the masking ate
  every label),
- saves the adapter to `WORK_DIR/llama-3.1-8B-cuad-task2/` and asserts the adapter
  files actually exist,
- writes `train_metrics.json` (final loss + all hyperparameters) so every run is
  self-describing and comes back as a Kaggle artifact.

---

## Stage 6 — Evaluation (cells 31–32)

### Cell 31 (markdown) — The three metrics and why they're ordered
Each metric only means something if the previous one passed:
1. **JSON validity** — can a machine parse the output at all?
2. **Exact match** — is the value identical to the gold answer?
3. **Token F1** — if not identical, how close?

### Cell 32 (code) — Generate, score, report, persist
For each of the ~480 validation examples:
1. **Generate** the model's answer (greedy, `max_new_tokens=128`).
2. **`parse_prediction`** — valid only if `json.loads` succeeds AND the result is an
   object whose *only* key is the requested category. Prose, bare values, markdown
   fences, wrong or extra keys all count as invalid — no forgiveness, because a
   downstream program wouldn't forgive them either.
3. **Exact match** — compare after lowercase+strip only; lists compared as
   order-insensitive sets; a `null` gold is only matched by a predicted `null`
   (predicting a value for an absent entity = hallucination = miss).
4. **Token F1** — word-overlap partial credit (`"State of Nevada"` vs `"Nevada"`
   scores 0.5, not 0). For lists, each gold item is matched to its best predicted
   item; unmatched items on either side score 0.

Then aggregates all three metrics **per category** (9 rows — dates are easy,
`Parties` is hard; an average would hide that) plus an OVERALL row, prints the
table, and writes `eval_metrics.json` + `eval_report.txt` to `WORK_DIR` so Kaggle
returns them as downloadable artifacts.

---

## What "success" looks like

Run the same evaluation on the base model without the adapter (separate notebook,
mirroring the Task 1 baseline). Expected deltas from fine-tuning, largest first:

1. **JSON validity** ↑↑ — pure format discipline; base models mostly fail this.
2. **Exact match** ↑ — the model learns to output the *normalized* value.
3. **Token F1** ↑ (smaller) — the base model often finds roughly the right span
   but packages it wrong.

If JSON validity barely moves, suspect the fine-tuning run first (e.g.
completion-only loss misconfigured), not the model.
