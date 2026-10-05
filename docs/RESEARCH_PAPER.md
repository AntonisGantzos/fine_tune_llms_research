# Parameter-Efficient Adaptation of an Open-Weight LLM to Three Legal Contract-Review Tasks

**A QLoRA study on CUAD and LEDGAR under free-tier GPU constraints**

---

## 1. Abstract

This study asks whether a general-purpose open-weight large language model can be adapted to three legal contract-review tasks using only free, consumer-grade compute. One base checkpoint, Meta-Llama-3.1-8B, was fine-tuned separately per task with QLoRA (4-bit NF4 quantisation plus a rank-16 LoRA adapter) on a single Kaggle Tesla T4. The tasks were binary risk-clause recognition and nine-category JSON entity extraction, both from CUAD, and 100-way provision classification from LexGLUE's LEDGAR. Each adapter was scored against the identical un-adapted base model on the same held-out validation examples, prompts and decoding settings. Fine-tuning improved macro-F1 from .536 to .951 (task 1), token-F1 from .141 to .819 (task 2) and macro-F1 from .066 to .751 (task 3). Much of the gain is output-format compliance rather than legal comprehension: valid-output rates rose from .472 to .997 and .401 to .996. Results are single-run, single-seed, validation-only.

*(150-word limit; 140 words.)*

---

## 2. Literature Review

**TODO** — completed under a separate work item; to be merged in.

---

## 3. Methodology

### 3.0 Framing and scope

The project treats contract review as three *generative* tasks rather than three classifier heads. In every case the model is shown an instruction and a piece of contract text, and must *write* the answer — the string `Yes`, a JSON object, or a label string. That choice is deliberate and has a cost that the evaluation is designed to expose: a generative model can fail by producing the wrong answer *or* by producing something that is not an answer at all. Sections 4.1 and 4.2 keep those two failure types separate throughout.

Three constraints shaped every design decision and should be read as part of the method :

1. **No local GPU.** Editing happens on a CPU-only Windows machine; all model loading and training happens remotely on Kaggle's free accelerator as a *batch* job (where the notebook is pushed to kaggle and run in that environment using kaggle's GPU access. The artifacts acquired from training are then pulled to the repository comprising this research). There is no interactive remote session, so a mistake costs a whole run.
2. **A 12-hour wall-clock kill** on Kaggle kernels, and a single **Tesla T4** (16 GB, compute capability 7.5) after pinning to one of the two offered GPUs.
3. **Free-tier reproducibility**: everything a run produced — metrics, hyperparameters, generated data, logs — is written to files and pulled back, because terminal output from a remote batch job is lost.

### 3.1 Analysis of the datasets used in this research

Two public, expert-annotated legal corpora are used. Neither was collected for this project, which matters, as it means the labels are not self-generated and the comparison to published work is meaningful.

#### 3.1.1 CUAD — Contract Understanding Atticus Dataset (tasks 1 and 2)

CUAD was built by legal experts at The Atticus Project and released as a NeurIPS Datasets & Benchmarks paper (Hendrycks et al., 2021). It contains 13,101 expert annotations across 41 clause categories drawn from 510 commercial contracts, and the published task is extractive: highlight the spans of a contract a lawyer would need to review.

This project does **not** use CUAD's extractive SQuAD-format file (`CUAD_v1.json`). It uses the flat summary table `master_clauses.csv`, verified in this repository as **510 rows × 83 columns**. The 83 columns are structured as paired columns per category:

- `[Category]` — the clause *excerpt* found in that contract, or empty when absent;
- `[Category]-Answer` — the expert ground-truth *value*: the literal string `"No"` when the clause is absent, or a normalised answer (a date, a party name, a term) when present.

Two consequences follow directly from that layout and drive the whole data design:

- **"Absent" rows carry no text.** When a category is absent from a contract, the context column is empty or holds a placeholder. A model trained on raw CUAD could therefore reach high accuracy on a Yes/No task by learning "short or empty input → No", never reading legal language at all. Task 1 closes this with hard negatives (§3.3.2), after the preprocessing described in §3.2.

- **Ground truth is *normalised*, not quoted.** The `-Answer` column holds the expert's canonical value (e.g. a date rendered as `5/8/14`), not the sentence it was lifted from. Task 2 is therefore a *normalisation* task, not a span-copying task. This point is mentioned because it actively changes how its exact-match score should be read (§4.1.2). This researches own preprocessing then normalises that value a *second* time, to `5814`, which turns out to matter a great deal; §3.2.2 documents it and §4.2.2 reads the consequences.

A preprocessing script within the repository of this research (`scripts/preprocess_values_cuad.py`) produces `master_clauses_cleaned.csv` (also 510 rows × 83 columns), the version both training notebooks read. The raw contract PDFs and plain-text files shipped with CUAD are intentionally unused, as full contracts exceed the model's practical prompt budget, and the CSV already holds the extracted clauses in a structured and more-easily read format.

CUAD ships **no official train/test split**, so one had to be constructed; §3.3.2 explains why it is made at the contract level.

#### 3.1.2 LEDGAR via LexGLUE (task 3)

LEDGAR is a large multi-label corpus of contract provisions scraped from U.S. Securities and Exchange Commission filings (Tuggener et al., 2020). The raw corpus has thousands of noisy, long-tailed labels, so this project uses the **LexGLUE** redistribution instead (Chalkidis et al., 2022), which keeps the 100 most frequent labels, makes the task single-label multi-class, and ships fixed splits of **60,000 train / 10,000 validation / 10,000 test** examples.

Using LexGLUE rather than raw LEDGAR buys two things: a clean, standard label set, and a published yardstick — Chalkidis et al. (2022) report BERT at roughly 87.6 micro-F1 / 81.8 macro-F1 and Legal-BERT at roughly 88.5 / 82.4 on this exact configuration.

Exploratory analysis of the LexGLUE LEDGAR splits produced four findings that each changed a design decision:

| Finding | Measurement | Decision it forced |
| :--- | :--- | :--- |
| **Severe class imbalance** | Most frequent label *Governing Laws* has 3,167 training examples; rarest, *Books*, has 23 — a 137× gap. Median ≈ 426. | Headline metric is **macro-F1**, not accuracy; training data is **stratified down** to a near-uniform per-label count. |
| **Provisions are short** | Median 104 tokens, 90th percentile 290, 99th percentile 585, longest 1,749 (real Llama-3.1 tokenizer). Only 26 of 70,000 provisions exceed 1,024 tokens unaided. | `max_length = 1024` is thus sufficient and no long-context model needed. |
| **The instruction is large** | The 100-label menu is **331 tokens**; with the `### Instruction / ### Input / ### Response` scaffolding, **341 tokens** of fixed overhead per prompt (confirmed in the production run log: *"Prompt budget: 341 instruction+template tokens + 670 provision tokens + ≤ 5 label tokens ≤ 1024"*). | The menu is kept in full — it is what makes the base-model baseline fair — and the *provision text* is trimmed to fit, never the assembled prompt. |
| **Some labels are near-synonyms** | e.g. *Governing Laws* / *Jurisdictions*, *Assigns* / *Successors*, *Amendments* / *Modifications*, *Waivers* / *No Waivers*. | Evaluation reports a **top-confusions** table, so a confusion between look-alikes can be distinguished from a random error. |

One genuine wrinkle: **all 100 labels appear in LexGLUE's train and test splits, but `Books` is absent from the validation split.** The project kept the official splits (the reason the published yardstick is comparable) and relaxed its own sanity check to require full 100-label coverage in *train only*. §5.1 quantifies the small metric artifact this creates.

#### 3.1.3 Processed training data 

All counts below were recomputed directly from the JSONL files the runs themselves emitted and returned as Kaggle artifacts, not taken from prose.

| Task | Source | Train examples | Validation examples | Categories / labels | Label balance |
| :--- | :--- | ---: | ---: | ---: | :--- |
| T1 Risk clause recognition | CUAD cleaned CSV | **6,106** | **2,208** | 32 | Train balanced **3,053 Yes / 3,053 No**; validation untouched at **558 Yes / 1,650 No** (25.3% positive) |
| T2 Entity extraction | CUAD cleaned CSV | **2,454** | **631** | 9 | n/a (free-form values) |
| T3 Provision classification | LexGLUE LEDGAR | **9,801** | **1,945** | 100 (train), **99 present in validation** | Train 23–100 per label; validation 0–20 per label |

T1 and T2 use **disjoint category sets**: the nine entity categories used by T2 (Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period To Terminate Renewal, Governing Law, Warranty Duration) plus `Filename` are excluded from T1, leaving exactly 32 Yes/No categories — asserted in code, not assumed.

Every record, in all three tasks, has the same four fields, which is what lets one trainer configuration and one evaluation idiom serve all three:

```json
{"instruction": "...", "category": "...", "input": "<contract text>", "output": "<target completion>"}
```

The `category` field is never shown to the model; it exists so that evaluation can be broken down per category or label.

### 3.2 Data pre-processing

This section documents every transformation applied between the published corpora and the JSONL files the trainer and the evaluator actually read. It is reported in detail because two of the transformations have measurable, previously undocumented effects on the task-2 results in §4.2.2, and because the pipeline cannot be reproduced without them.

An important structural point first: **all three tasks build their training *and* evaluation data in the same notebook cells, from the same source, with the same code.** There is no separate evaluation preprocessing path. The split happens at the contract (or, for task 3, the official-split) level, and the two halves are then passed through an identical builder. The baseline notebooks do not rebuild anything — they load the JSONL files the fine-tuned run emitted and returned as Kaggle artifacts, which is what makes the record-by-record validation-set equality check in §3.3.2 possible. The evaluator additionally reuses the *same* prompt-assembly function as training, so "preprocessing for evaluation" is, by design, nothing more than "preprocessing for training, minus the target completion".

#### 3.2.1 Acquisition

| Task | Script | What it fetches | What it writes |
| :--- | :--- | :--- | :--- |
| 1, 2 | `scripts/download_cuad.py` | The CUAD v1 release | `data/CUAD_v1/` including `master_clauses.csv` |
| 3 | `scripts/download_ledgar.py` | `coastalcph/lex_glue`, config `ledgar`, via the `datasets` library | `data/LEDGAR/labels.json` plus `ledgar_{train,validation,test}.csv` |

The LEDGAR script does one piece of real work rather than a plain dump: the source dataset stores the label as an integer 0–99, so the script resolves the `ClassLabel` feature's `names` list once, writes an explicit `id → name` map to `labels.json`, and adds a `label_name` column to every CSV. Everything downstream therefore reads plain files and never re-resolves label ids — the same design principle as CUAD's CSV. Both scripts are idempotent (they skip an existing target unless `--force` is passed).

#### 3.2.2 Task 1 and Task 2 — the shared CUAD pre-processing chain

CUAD passes through **five** stages before it reaches the model. Stages 1–2 are shared by both tasks; stages 3–5 differ.

##### Stage 1 — Cell-level text cleaning (`scripts/preprocess_values_cuad.py`)

The raw CSV is re-read with `csv.DictReader` (rather than `pandas.read_csv`) using `encoding="utf-8", errors="replace"`, because the raw file contains inconsistent quoting and non-UTF-8 bytes that break a naive parse. The resulting frame then has a single `clean_text` function applied to **every cell**, via `df.map(clean_text)`. That function:

1. maps missing values to the empty string;
2. coerces to `str`;
3. removes `[`, `]`, `{`, `}` explicitly;
4. **removes every character that is not `[a-zA-Z0-9\s]`**;
5. collapses newlines, tabs and runs of whitespace into single spaces;
6. strips leading/trailing whitespace.

The output is `master_clauses_cleaned.csv` — still 510 × 83 — and it is this file, not the raw one, that both training notebooks read.

Step 4 is the consequential one, and its effects were measured during this review rather than assumed. Because it is applied to *every* cell, it transforms the **ground-truth answer columns** as well as the clause text:

| Column | Raw CUAD value | Value in `master_clauses_cleaned.csv` |
| :--- | :--- | :--- |
| `Agreement Date-Answer` | `5/8/14` | `5814` |
| `Expiration Date-Answer` | `12/31/14` | `123114` |
| `Expiration Date-Answer` | `6/30/10` | `63010` |
| `Governing Law-Answer` | `Nevada` | `Nevada` *(unchanged)* |
| `Parties-Answer` | `Birch First Global Investments Inc. ("Company"); Mount Kowledge Holdings Inc. ("Marketing Affiliate", "MA")` | `Birch First Global Investments Inc Company Mount Kowledge Holdings Inc Marketing Affiliate MA` |

Three consequences follow, all of which bear directly on §4.2.2 and none of which were previously recorded:

- **Every date target becomes an unseparated digit string.** The model is trained to answer `{"Agreement Date": "5814"}` for a clause reading *"8th day of May 2014"*. The target is lossy and ambiguous (`5814` is equally readable as 5/8/14 or 5/81/4), and the digit count varies with the date (`123114` is six digits, `63010` five). This is an arbitrary surface form the model must *infer*, not a value it can copy.
- **The `Parties` list separator is destroyed.** 99.2% of raw `Parties-Answer` values use `;` to delimit the parties; `clean_text` strips it, collapsing the list into one space-joined blob. Stage 4 below shows what that destroys.
- **Clause text loses all punctuation and symbols.** Verified on a 400-example sample of the generated task-1 inputs: the characters `.` `,` `%` `$` `(` `"` `;` occur in **0.0%** of them. So `10%` becomes `10` ("an amount equal to 10 of the hosting fees"), sentence boundaries vanish, and the quotation marks that mark defined terms in contract drafting are gone. For comparison, task 3's inputs are untouched — `.` appears in 100% and `,` in 89.2% of them — so the two CUAD tasks and the LEDGAR task do not feed the model comparable text.

No stage of the pipeline re-introduces the stripped information, and no sanity check tests for it. This is recorded as failure mode 12 in §5.1.

##### Stage 2 — Column-name normalisation (notebook cells 8–10, identical in both notebooks)

Both notebooks re-read the cleaned CSV with `csv.DictReader` and then normalise the **headers only** (a reduced `clean_text` that strips non-alphanumerics is applied to column names; cell values are *not* re-cleaned, having been cleaned in stage 1). Any header containing `Answer` is then renamed so that the suffix is `_Answer`:

```
CSV header            after header cleaning     after rename
"Document Name-Answer"  →  "Document NameAnswer"  →  "Document Name_Answer"
```

This gives the notebooks the `[Category]` / `[Category]_Answer` pairing their code relies on. It has one side effect that the model sees directly: because header cleaning strips hyphens and slashes, **12 of the 32 task-1 category names are mangled in the prompt**, e.g. `Non-Compete` → `NonCompete`, `Revenue/Profit Sharing` → `RevenueProfit Sharing`, `Anti-Assignment` → `AntiAssignment`, `Rofr/Rofo/Rofn` → `RofrRofoRofn`, `Unlimited/All-You-Can-Eat-License` → `UnlimitedAllYouCanEatLicense`. The model is therefore asked *"Is the following contract text a `RofrRofoRofn` clause?"*. Because the same mangled name is used in training, evaluation **and** the baseline, this does not bias the comparison; it does make the prompt less legible and is a plausible small handicap on both arms.

##### Stage 3 — Category selection

Task 2 takes the 9 entity categories by explicit list. Task 1 takes every column that has a matching `_Answer` partner and is **not** in task 2's list (which also excludes `Filename`), then asserts the result is exactly 32 — so a change in the source schema fails loudly instead of silently shifting the task definition.

##### Stage 4 — Example construction (the task-specific step)

**Task 1** converts each `[Category]_Answer` cell to a binary label (`"No"` or blank → `No`; anything else → `Yes`) and then builds one example per (contract, category) pair with the hard-negative procedure described in §3.3.2. Two silent filters apply: a `Yes` row whose context cell is blank is **skipped** (the two columns disagree), and a `No` row is skipped if the contract has no other populated category to borrow from.

**Task 2** runs a dedicated `normalize_answer` function over each answer cell, which is designed to produce `None | str | list[str]`:

1. missing or blank → `None`;
2. a bracketed list-string such as `"['5/8/2014']"` is parsed with `ast.literal_eval` and re-joined with `"; "`;
3. the string is split on `;`, blank parts dropped;
4. **more than one part → `list[str]`; exactly one part → plain `str`**.

The example is then emitted only if the *context* column is non-empty, with the target produced by `json.dumps({category: value})` — so the label is valid single-key JSON by construction, with `None` serialising to `null`. This skipping is why task 2 has **3,085 examples rather than 510 × 9 = 4,590**.

Two findings about this stage, both verified across all 3,085 generated examples:

- **The `list[str]` branch never fires.** Every target value is a `str` (2,972) or `null` (163); **not one is a JSON list.** The cause is the chain described in stage 1: `normalize_answer` splits on `;`, but `clean_text` already removed every `;`, so step 3 always yields exactly one part and step 4 always takes the scalar branch. The function's own self-tests pass because they are run on hand-written strings (`"Party A; Party B"`) that still contain the delimiter, which is why the problem went unnoticed. The consequence is that the instruction's promise *"If multiple values exist, return them as a list of strings"* is never demonstrated to the model, and the list-handling half of the evaluation metric is dead code on this dataset (see the correction in §4.1.2).
- **`Warranty Duration` has a constant target.** In raw CUAD, `Warranty Duration-Answer` is a presence column holding `Yes` (75 contracts) or `No` (435), not a duration. Because stage 4 keeps only rows with non-empty context, all 75 surviving examples have the gold value `"Yes"` — a single constant string. Task 2 therefore never learns to extract a warranty duration; it learns to emit `{"Warranty Duration": "Yes"}` unconditionally. Its reported 1.000 exact match (n = 11) is a degenerate constant-target result, not an extraction result.

A third measurement from this review puts the remaining per-category scores in context by asking how much of each target is *present in the input at all*:

| Category | gold == input exactly | gold is a substring of input | total | Reported FT exact match |
| :--- | ---: | ---: | ---: | ---: |
| Document Name | **501 (98.2%)** | 501 | 510 | 0.980 |
| Governing Law | 0 | **391 (89.5%)** | 437 | 0.920 |
| Notice Period To Terminate Renewal | 0 | 83 (74.8%) | 111 | 0.931 |
| Agreement Date | 24 | 46 (9.9%) | 466 | 0.840 |
| Effective Date | 9 | 22 (5.7%) | 388 | 0.792 |
| Renewal Term | 0 | 27 (15.3%) | 176 | 0.535 |
| Parties | 13 | 14 (2.8%) | 509 | 0.206 |
| Expiration Date | 0 | **4 (1.0%)** | 413 | 0.395 |
| Warranty Duration | 0 | **0 (0%)** | 75 | 1.000 |

This reframes §4.2.2 considerably. `Document Name` is a near-verbatim **copy** task (the gold answer *is* the input string in 98.2% of cases), and `Governing Law` is a **span-copy** task — which is why both score highest. `Expiration Date` is at the opposite extreme: its target appears in the input 1.0% of the time, because stage 1 turned it into a digit string that the clause text cannot contain. Its 0.395 exact match is therefore better read as *"the model inferred the arbitrary digit encoding 39.5% of the time"* than as a statement about legal date comprehension.

##### Stage 5 — Split and serialisation

Contracts are split **before** examples are built, with `sklearn.model_selection.train_test_split(df, random_state=42)`, so no contract contributes clauses to both sides (§3.3.2 explains why this matters for task 1 in particular). The two tasks do **not** use the same ratio:

| Task | `test_size` | Contracts (train / validation) | Examples (train / validation) |
| :--- | ---: | :--- | :--- |
| 1 | 0.15 | 433 / 77 | 6,106 / 2,208 *(after balancing)* |
| 2 | 0.20 | 408 / 102 | 2,454 / 631 |

Task 1 then applies a **train-only** 1:1 class balance (§3.3.2); task 2 applies no balancing. This is why task 1's *example* ratio (6,106 / 2,208 ≈ 73/27) looks nothing like its *contract* ratio (85/15). Downsampling the majority `No` class removes training examples only, so the training side shrinks while validation keeps every example. `docs/task_1/TASK1_EVALUATION_METRICS_ASSESSMENT.md` flags this gap as a possible split bug; the balancing step fully explains it. Both write JSONL with one JSON object per line. Note that task 1 and task 2 serialise with a plain `json.dumps` and no explicit file encoding, whereas task 3 uses `ensure_ascii=False` with `encoding="utf-8"`; for the CUAD tasks this is inconsequential because stage 1 has already removed every non-ASCII character.

Each notebook's sanity-check cell runs before serialisation and is part of the preprocessing contract, not an afterthought. Task 1 asserts that no placeholder string survives into any `No` input — the check that proves the hard-negative substitution worked — and prints the per-class counts. Task 2 asserts that every category has at least one train *and* one validation example, and that **every generated target round-trips through `json.loads` with exactly the requested key** — because a malformed target would be actively teaching the model to emit bad JSON.

#### 3.2.3 Task 3 — the LEDGAR pre-processing chain

Task 3's chain is shorter, and notably **applies no text cleaning at all**: the provision text goes from the LexGLUE CSV into the prompt as `str(row.text)`, punctuation, casing and typography intact (verified in §3.2.2, stage 1). LexGLUE has already done the normalisation work that CUAD required — single label, 100 classes, fixed splits — so there is nothing analogous to the CUAD cleaner to run.

1. **Label menu construction.** `labels.json` is read into an `id → name` map and flattened into `LABELS`, a list **in label-id order**. This ordering is the canonical menu: it is joined with `", "` and inlined into the instruction, and it also defines the lookup table used to parse predictions. Using one ordered list for both means the menu the model is shown and the vocabulary it is scored against cannot drift apart.
2. **Stratified subsampling.** `stratified_sample` groups the full split by `label_name` and takes `min(len(group), per_label)` rows from each with `random_state=42`, then shuffles the result. `TRAIN_PER_LABEL = 100` and `VAL_PER_LABEL = 20`. Rare labels cap at whatever they have: in train, exactly three labels fall below the target — `Books` (23), `Assigns` (31) and `Qualifications` (47) — yielding 9,801 rows; validation yields 1,945 rows with per-label counts of 0–20 (`Books` 0, `Assigns` 3, `Qualifications` 8, `Powers` 15, `Venues` 19). The 10,000-row official **test** split is downloaded and then left untouched.
3. **Example construction.** One example per row: the shared 100-label instruction, `input` = the provision text, `output` = `label_name`, `category` = the same label name (the per-label evaluation hook). No negatives to synthesise and no values to normalise — the target is already a member of a closed set.
4. **Sanity checks.** Every target must be one of the 100 allowed labels; no input may be blank; per-label counts are summarised; and label coverage is asserted to be complete **for train only** (`require_full_coverage=False` for validation), which is the explicit accommodation for `Books` being absent from LexGLUE's validation split (§3.1.2).
5. **Prompt-budget enforcement.** Unlike the CUAD tasks, task 3's length control happens at *prompt-assembly* time rather than at data-build time, in the shared `build_prompt()`: the provision is trimmed to `MAX_INPUT_TOKENS` measured with the real Llama-3.1 tokenizer, so the tail-positioned label can never be cut. The budget is derived, not hardcoded, and the production run logged it as *341 instruction+template tokens + 670 provision tokens + ≤ 5 label tokens ≤ 1024*, trimming 79/9,801 train (0.81%) and 9/1,945 validation (0.46%) provisions. §3.3.1 explains why trimming the input rather than the assembled prompt was mandatory.

#### 3.2.4 Summary — pre-processing by task

| Stage | Task 1 (CUAD) | Task 2 (CUAD) | Task 3 (LEDGAR) |
| :--- | :--- | :--- | :--- |
| Source | `master_clauses.csv` (510 × 83) | same | `ledgar_{train,validation}.csv` + `labels.json` |
| Text cleaning | strip all non-alphanumerics, every cell | same | **none** |
| Header normalisation | yes (mangles 12 of 32 category names) | yes | n/a |
| Target construction | `"No"`/blank → `No`, else `Yes` | `normalize_answer` → `str`/`null`, wrapped by `json.dumps` | `label_name` verbatim |
| Negatives | **hard negatives**: a real clause from another category of the same contract, seed 42 | n/a (skip rows with no context) | n/a |
| Rows skipped | `Yes` with blank context; `No` with nothing to borrow | any category with blank context (4,590 → 3,085) | none |
| Split | contract-level 85/15, seed 42 | contract-level 80/20, seed 42 | LexGLUE official splits |
| Subsampling | train-only 1:1 class balance | none | stratified ≤100/label train, ≤20/label validation, seed 42 |
| Length control | trainer-level `max_length=1024`, end-truncation; **50/6,106 train and 6/2,208 validation prompts already ≥ 1,024 tokens**, so their target is cut | same cap, never reached (longest example 652 tokens) | **input trimmed to 670 tokens** before assembly; label can never be cut |
| Output | `cuad/{train,validation}/cuad_*.jsonl` | `cuad/{train,validation}/cuad_task2_*.jsonl` | `ledgar/{train,validation}/ledgar_task3_*.jsonl` |
| Evaluation data | the same JSONL, response withheld | same | same, same `build_prompt()` |

### 3.3 Implementation

#### 3.3.1 The fine-tuning technique and the models

**Why parameter-efficient fine-tuning at all.** Full fine-tuning of an 8-billion-parameter model requires holding the weights, their gradients, and two optimiser moments in memory — far beyond 16 GB. LoRA (Hu et al., 2022) avoids this by freezing the pretrained weights and learning a low-rank update: for a frozen weight matrix *W*, it trains two small matrices *A* and *B* and uses *W + BA*, where the rank *r* of *BA* is tiny relative to *W*. Only *A* and *B* receive gradients, so optimiser state shrinks by orders of magnitude and the original model is never modified.

**Why QLoRA specifically.** QLoRA (Dettmers et al., 2023) pushes this further by also *quantising* the frozen base weights to 4 bits. It contributes three components, two of which are used here:

- **4-bit NormalFloat (NF4)** — *used.* A quantisation data type designed to be information-theoretically well matched to normally distributed weights, which neural-network weights approximately are. The weights are stored in 4 bits and dequantised block by block to a 16-bit compute dtype for every matmul; the 4-bit format is storage only.
- **Paged optimisers** — *used.* NVIDIA unified memory lets transient optimiser-memory spikes page to host RAM instead of raising an out-of-memory error.
- **Double quantisation** — *not used.* This quantises the per-block quantisation constants themselves and saves roughly 0.37 bits per parameter, about 0.3 GB on this model. None of the notebooks set `bnb_4bit_use_double_quant`, so it stays at its default of `False`. The memory problems this project actually hit were dominated by the logits tensor (*memory and numerics*, item 3, below), which double quantisation does not touch.

Dettmers et al. (2023) report that the full combination fits fine-tuning of a 65B model onto a single 48 GB GPU while preserving 16-bit fine-tuning task performance. The present work is the same idea at a smaller scale: an 8B model on a 16 GB card.

**Base model.** All three production runs adapt `meta-llama/Meta-Llama-3.1-8B` — the **base** checkpoint, not `-Instruct`. Llama 3.1 is an open-weight dense transformer family released with a 128K-token context window (Grattafiori et al., 2024); the 8B configuration has 32 layers, hidden size 4,096, `max_position_embeddings` 131,072, and a vocabulary of **128,256** tokens. That vocabulary size is not trivia — it dominates peak memory during training (§3.3.1, *memory*) because the loss materialises a `tokens × 128,256` logits tensor.

Choosing the base rather than the instruction-tuned checkpoint is a deliberate methodological decision: the research question is what *fine-tuning* teaches, so the baseline must be a model that has had no instruction tuning of its own to confound the comparison. The project's own model-selection review (`docs/MODEL_SELECTION.md`) notes that `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, stronger reported JSON adherence) and the legal-domain-pretrained `Equall/Saul-7B-Instruct-v1` (Colombo et al., 2024) are arguably better *performance* picks; neither was swapped in, so this remains an untested alternative rather than a finding (§5.2).

`meta-llama/Llama-3.2-1B` was used as a same-family **smoke test** to prove the pipeline end to end. Its artifacts survive (`kaggle_output_3.2B_Llama_smoketest/`: accuracy .897 on a 448-example subset) and are reported here only as evidence the pipeline ran, never as a result.

**Fine-tuning parameters — what each one does, and why it has this value.** The tables below list every setting passed in the training notebooks (cells 27 of T1/T2, cell 21 of T3), plus the library defaults that matter because they were *not* overridden. Values were cross-checked against each run's `train_metrics.json` and the saved `adapter_config.json`. Where T3 differs from T1/T2, both values are shown; §3.3.1, *memory and numerics*, explains why T3 differs.

*(a) Quantisation — `BitsAndBytesConfig`.* This controls how the frozen base model is stored.

| Parameter | T1 / T2 | T3 | What it does, and why this value |
| :--- | :--- | :--- | :--- |
| `load_in_4bit` | `True` | `True` | Stores every frozen linear layer in 4 bits. The 8B model's linear weights drop from ~16 GB (fp16) to ~3.6 GB, which is what lets an 8B model train on a 16 GB card at all. |
| `bnb_4bit_quant_type` | `"nf4"` | `"nf4"` | NormalFloat-4: the 16 quantisation levels are placed at quantiles of a normal distribution, so they match the shape of trained weights better than uniform `fp4` levels do. QLoRA's recommended type. |
| `bnb_4bit_compute_dtype` | `bfloat16` | `float16` | The dtype each 4-bit block is dequantised to for the matmul. It must be a dtype the GPU has tensor cores for. The T4 has fp16 tensor cores but no bf16 ones, so T1/T2 paid an emulation penalty (§5.1, item 21). |
| `bnb_4bit_use_double_quant` | not set (`False`) | not set (`False`) | QLoRA's second quantisation of the block constants. It would save ~0.3 GB but was never enabled (see above). |
| `torch_dtype` (in `from_pretrained`) | not set | `float16` | The dtype of the modules that are *not* quantised: embeddings, RMSNorms, `lm_head`. T3 sets it explicitly so these match the compute dtype. |

*(b) Model placement and memory.*

| Parameter | Value (all tasks) | What it does, and why |
| :--- | :--- | :--- |
| `CUDA_VISIBLE_DEVICES="0"` + `device_map={"": 0}` | one T4 | Kaggle's "T4 ×2" exposes two GPUs, and `device_map="auto"` splits the model across both. TRL's loss then built the label mask on `cuda:0` while `lm_head` lived on `cuda:1`, which raised a device-mismatch error. The 4-bit model fits on one card, so the second GPU is hidden and stays idle. |
| `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` | set | Lets PyTorch's allocator grow memory segments instead of reserving fixed blocks. This reduces fragmentation when tensor sizes vary from batch to batch. |
| `model.config.use_cache` | `False` in training, `True` in evaluation | The KV cache only helps step-by-step generation. In teacher-forced training it wastes memory and conflicts with gradient checkpointing, so it is switched off for training and back on for `generate()`. |

*(c) Tokenizer.*

| Parameter | Value | What it does, and why |
| :--- | :--- | :--- |
| `pad_token = eos_token` | all tasks | The Llama 3.1 base tokenizer has no pad token, so EOS is reused. Padded positions are hidden by the attention mask and excluded from the loss, so the model is not taught to emit them. |
| `padding_side` | `"right"` in training; `"left"` for T3's batched generation | Right padding is the safe choice for causal-LM training: real tokens start at position 0 in every row. For *batched* generation, the left padding T3 uses puts every row's last prompt token directly before the first generated token. T1/T2 and all baselines generate one example at a time, so their generation never pads. |

*(d) The adapter — `LoraConfig`.*

| Parameter | Value (all tasks) | What it does, and why this value |
| :--- | :--- | :--- |
| `r` | 16 | The rank of the update *ΔW = BA*. Each adapted *d_out × d_in* matrix gains *r·(d_in + d_out)* trainable parameters. 16 is a mid-range rank: the adapter has to learn an output format and a label mapping, not new world knowledge. No other rank was tried (§5.1, item 24). |
| `lora_alpha` | 16 | The update is scaled by *alpha / r* before it is added to *W*. With alpha = r the scale is 1.0, so the learning rate alone sets the step size. Changing `r` without changing `alpha` would change the effective step size. |
| `lora_dropout` | 0.05 | Dropout on the adapter's input path only; the frozen path is unaffected. It is a light regulariser for the single-epoch, few-thousand-example regime. |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Adapters are attached to all four attention projections in each of the 32 layers. The MLP projections (`gate/up/down_proj`) are not adapted. This is wider than the `q,v`-only minimum from the original LoRA paper, at negligible cost. |
| `bias` | `"none"` | No bias terms are trained. Llama has no biases in these layers anyway. |
| `task_type` | `"CAUSAL_LM"` | Tells `peft` to wrap the model for next-token prediction. |
| initialisation | `peft` default (`init_lora_weights=True`) | *A* is random and *B* is zero, so *BA = 0* at step 0. The adapted model therefore starts *exactly* equal to the baseline. This is part of why the base model is the correct baseline. |
| **Resulting size** | **13,631,488 trainable parameters** | Per layer: `q`/`o` add 16·(4096+4096) = 131,072 each, and `k`/`v` add 16·(4096+1024) = 81,920 each, because grouped-query attention gives K and V only 1,024 output dims. That is 425,984 per layer × 32 layers. This is **0.17% of the 8.03B base parameters**. The T3 log prints 0.299% because it divides by the 4.55B *stored* parameter count, in which two 4-bit weights share one byte. Stored in fp32, each adapter is a 54.6 MB file. |

*(e) Optimisation — `SFTConfig` (a `transformers.TrainingArguments` subclass).*

| Parameter | T1 / T2 | T3 | What it does, and why this value |
| :--- | :--- | :--- | :--- |
| `num_train_epochs` | 1 | 1 | One pass over the training set, set by the compute budget. No epoch sweep was run (§5.1, item 1). |
| `per_device_train_batch_size` | 1 | 2 | Rows per forward pass. bitsandbytes dequantises every weight on every forward pass whatever the batch size, so batch 1 spends most GPU time dequantising. T3 raised it to 2 to amortise that cost. |
| `gradient_accumulation_steps` | 8 | 4 | Micro-batches summed before one optimiser step. |
| → effective batch / optimiser steps | 8 / ⌈N/8⌉ | 8 / ⌈N/8⌉ | Same effective batch in all three runs, giving 764 (T1), 307 (T2) and 1,226 (T3) optimiser steps, which match `global_step` in each `train_metrics.json`. |
| `group_by_length` | `False` | `True` | Puts examples of similar length in the same batch, so less compute is spent on padding. It only matters when batch > 1. Side effect: the longest batch is scheduled first, which is why T3's pre-flight check 9 measures that batch. |
| `learning_rate` | 2e-4 | 2e-4 | The conventional LoRA/QLoRA rate, about 10× a typical full-fine-tuning rate, because only the small, zero-initialised *B·A* is being trained. |
| `lr_scheduler_type`, `warmup_ratio` | not set | not set | `TrainingArguments` defaults apply: **linear decay from 2e-4 to 0 over the run, with no warmup.** No schedule was chosen deliberately. |
| `weight_decay` | 0.001 | 0.001 | Decoupled AdamW weight decay on the trainable (adapter) parameters only. Deliberately small. |
| `max_grad_norm` | not set (1.0) | not set (1.0) | The default gradient-norm clipping at 1.0 applies. |
| `optim` | `paged_adamw_32bit` | `paged_adamw_32bit` | AdamW (default β₁ = 0.9, β₂ = 0.999, ε = 1e-8) with 32-bit moment estimates in paged memory. Optimiser state is only 2 × 13.6M × 4 bytes ≈ 109 MB, so paging is a safety net here rather than a necessity. |
| mixed precision | `bf16=True` | `fp16=True` | Autocasts the forward and backward passes to 16-bit, while the trainable adapter weights and optimiser state stay fp32. `fp16` adds dynamic loss scaling, which multiplies the loss to stop small gradients underflowing. That is why T3's pre-flight check asserts the adapter weights are fp32: fp16 gradients cannot be unscaled. |
| `gradient_checkpointing` (+ `use_reentrant=False`) | `True` | `True` | Stores only layer-boundary activations and recomputes the rest in the backward pass, trading ~30% extra compute for a large cut in activation memory (Chen et al., 2016). The non-reentrant variant is needed because the frozen base weights do not require gradients, and the older reentrant implementation can then silently drop the adapter's gradients. |
| `max_length` | 1,024 | 1,024 | The maximum tokens per example (prompt + completion). A longer example is **truncated from the end, which deletes the answer**. T3 prevents this by trimming the input text first. T1 does not, and loses the target on 0.8% of its training examples (§3.2.4, §5.1 item 11). |
| `completion_only_loss` | `True` | `True` | Gives prompt tokens the label `-100`, so the cross-entropy loss is computed only on the answer. Without it, the loss on a one-token answer like `Yes` would be swamped by the loss on reproducing hundreds of clause tokens. |
| `packing` | `False` | `False` | Each sequence holds exactly one example, so no example can attend to another. |
| `logging_steps` | 25 | 25 | Training loss is printed every 25 optimiser steps, into the run log. |
| `save_steps` / `save_total_limit` | 100 / unlimited | 250 / 1 | Intermediate checkpoints, kept only as crash insurance. T3 reduced them after a run spent output space on 173 MB checkpoints every 100 steps. Only the final adapter is evaluated. |
| `eval_strategy` | not set (`"no"`) | not set (`"no"`) | No in-training evaluation. The validation set is passed to the trainer but is used only *after* training, by the custom generation loop, so no checkpoint or early-stopping decision was made on it. |
| `seed` | not set (42) | not set (42) | The trainer's default seed fixes the LoRA *A* initialisation and the data order. The data-construction seeds (split, hard negatives, stratified sampling) are set to 42 separately in the data cells. |
| `report_to` | `"none"` | `"none"` | No external experiment tracker; everything is written to files in `WORK_DIR`. |

*(f) T3-only run controls.* These are described in full under *memory and numerics* below:

- `MAX_INPUT_TOKENS = 1024 − 341 (instruction + template) − 5 (longest label) − 8 (EOS + slack) = 670`, derived from the real tokenizer.
- `TimeBudgetCallback` with a budget of 25,200 s (7 h).
- `embed_tokens` and `lm_head` are recast to fp16 after trainer initialisation.

*(g) Generation at evaluation time.* These settings are the same for a fine-tuned run and its baseline, except where marked.

| Setting | T1 | T2 | T3 fine-tuned | T3 baseline |
| :--- | :--- | :--- | :--- | :--- |
| Decoding | greedy (`do_sample=False`) | greedy | greedy | greedy |
| `max_new_tokens` | 3 | 128 | longest label + 2 = 7 | 16 |
| Batch | 1 | 1 | 8, left-padded | 1 |
| Autocast dtype | bf16 | bf16 | fp16 | bf16 |
| Prompt length control | tokenizer truncation of the assembled prompt at 1,024 | same (never triggered) | `build_prompt()` input trimming | tokenizer truncation of the assembled prompt |
| Parse | `"yes" in completion.lower()` | strict `json.loads` + exact key set | first line, case-insensitive label lookup | same as fine-tuned |

Greedy decoding makes every prediction deterministic, so a fine-tuned-versus-baseline difference cannot be sampling noise. `max_new_tokens` is the smallest bound that can hold a complete answer, which caps how far a base model can ramble.

**Software stack.** The notebooks use:

- Hugging Face `transformers` (Wolf et al., 2020) for the model and tokenizer;
- `datasets` (Lhoest et al., 2021) for loading JSONL;
- `peft` (Mangrulkar et al., 2022) for the LoRA adapter;
- `bitsandbytes` (Dettmers et al., 2022) for 4-bit quantisation;
- `trl`'s `SFTTrainer`/`SFTConfig` (von Werra et al., 2020) for supervised fine-tuning;
- `scikit-learn` (Pedregosa et al., 2011) for the classification metrics.

On Kaggle, the first cell of every training and baseline notebook pins the core stack: `transformers==4.55.4`, `bitsandbytes==0.46.1`, `accelerate==1.7.0`, `peft==0.15.2` and `trl==0.20.0`. `transformers` is held below 4.56 because that release's threaded loader broke bitsandbytes 4-bit loading and fell back to a full fp16 load, which ran out of memory. `datasets`, `torch` and CUDA come unpinned from the Kaggle base image (§5.1, item 19).

**Prompt template.** One template serves all three tasks, in training *and* evaluation *and* the baselines — the Alpaca instruction format popularised by Stanford Alpaca (Taori et al., 2023), itself derived from Self-Instruct (Wang et al., 2023):

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

The completion the model is trained to produce is exactly the `output` field, appended after `### Response:\n`. That marker is also precisely where `completion_only_loss` stops masking, so the loss is computed on the answer and nothing else.

**Memory and numerics — the engineering that made the runs finish.** These are not incidental implementation notes; two of them are the difference between a result and a dead kernel, and they are reported because they are reusable findings about this hardware tier.

1. **`fp16`, not `bf16`, on a T4 (task 3).** bfloat16 requires compute capability ≥ 8.0 (Ampere); Turing cards such as the T4 (cc 7.5) have no bf16 tensor cores. Critically, `torch.cuda.is_bf16_supported()` nonetheless returns `True` on these cards because recent PyTorch counts emulated bf16 as support (pytorch/pytorch issues #75427, #118122), so a bf16 run proceeds silently at a large speed penalty. An earlier task-3 attempt at bf16 measured ~12.8 s per example (~35 h for one epoch) and was killed by Kaggle's 12 h wall. Switching to `fp16` everywhere (`fp16=True`, `bnb_4bit_compute_dtype=torch.float16`) with batch 2 × accumulation 4 brought the measured pace to 18.0 s/step (~6.1 h/epoch, logged at step 100). Mixed-precision training with fp16 is the standard technique here (Micikevicius et al., 2018). A pre-flight check now hard-fails bf16 on any pre-Ampere GPU.
2. **Cap the sequence by trimming the *input*, never the assembled prompt.** Because the answer sits at the *tail* of the sequence and the 341-token label menu sits at the head, truncating the assembled sequence at 1,024 tokens deletes the training signal on the longest examples. With `completion_only_loss` and a micro-batch of 1, a micro-batch whose every token is masked produces **NaN loss** — which is exactly what killed the earliest task-3 attempts. The fix derives the input budget from the real tokenizer (`MAX_SEQ_LENGTH − overhead − longest label − 8` = 670 provision tokens) and trims the provision text instead. The run log records the cost: **79/9,801 train (0.81%)** and **9/1,945 validation (0.46%)** provisions trimmed. For context, LexGLUE's own BERT baseline truncates LEDGAR at 512 tokens, so 670 is conservative. **Task 1 has the same exposure and no such guard.** Re-tokenising the T1 run's own JSONL with the Llama-3.1 tokenizer shows **50 of 6,106 training prompts (0.82%)** are already ≥ 1,024 tokens before the answer is appended (longest 2,880; median 120). On those 50 the trainer's end-truncation removes the `Yes`/`No` target, so they carry no training signal. T1 did not diverge (final loss 0.094), but those 50 examples were silently wasted. Task 2 never reaches the cap: its longest example is 652 tokens.
3. **Peak VRAM is dominated by the logits tensor, not the weights.** The loss materialises a `tokens_in_batch × 128,256` logits tensor in fp16 and again in fp32. At `max_length=2048` with batch 2, `group_by_length` deliberately schedules the longest example first, which produced a 1.81 GiB fp32 allocation on top of ~13.5 GiB and a hard OOM inside `cross_entropy`. Dropping to 1,024 fixed it; the production run's own memory probe logged *"worst-case batch (2036 tokens, ~1.6 GB of logits) peaked at 10.8 / 15.6 GB"* and *"projected training peak 11.9 / 15.6 GB"*.
4. **`embed_tokens` and `lm_head` are recast to fp16 after trainer initialisation.** `SFTTrainer` runs peft's `prepare_model_for_kbit_training` for 4-bit models, which upcasts *every* fp16 parameter to fp32 — including those two frozen 525M-parameter modules, taking each from 1.05 GB to 2.1 GB. Recasting them back (while deliberately leaving the RMSNorms in fp32 for stability, ~1 MB) frees 2 × 1.05 GB, as the production log confirms. It also removes the ~1.05 GB fp16 copy of `lm_head` that autocast otherwise caches, for ~3.2 GB in total. The widely repeated "4-bit 8B ≈ 5.6 GB" rule of thumb is wrong for this stack by roughly 3 GB.
5. **`generate()` must run inside `torch.autocast`.** The same fp32/fp16 split, seen from the other side: `prepare_model_for_kbit_training` leaves the RMSNorms in fp32, so the final norm emits fp32 hidden states into an fp16 `lm_head` → `RuntimeError: expected scalar type Float but found Half`. Training never hits it because accelerate wraps the training forward in autocast; `generate()` is wrapped by nothing. This crashed a task-3 run **after 5.5 hours of successful training**. Two pre-flight checks now exercise a real forward+backward *and* a real 2-prompt `generate()` before training starts, so a path failure costs seconds.
6. **A time-budget callback** stops task-3 training at 7 h so that adapter saving and evaluation always complete inside the 12 h kill, and writes `stopped_on_time_budget` into both metrics files so a partial epoch can never be silently compared against a full one. The production run completed its epoch with the budget untouched (`stopped_on_time_budget: false`).

**Recorded cost of the three production runs** (from each run's `train_metrics.json`):

| Run | Optimizer steps | Final training loss | Runtime | Throughput (samples/s) | Precision | Adapter |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| T1 | 764 | 0.0942 | 30,995 s (8.6 h) | 0.197 | bf16 | `llama-3.1-8B-cuad-task1` |
| T2 | 307 | 0.2427 | 10,076 s (2.8 h) | 0.244 | bf16 | `llama-3.1-8B-cuad-task2` |
| T3 | 1,226 | 0.2926 | 22,025 s (6.1 h) | 0.445 | fp16 | `llama-3.1-8B-ledgar-task3` |

All three adapters have the same shape: 13,631,488 trainable parameters, saved as a 54.6 MB `adapter_model.safetensors`. The final training losses are not comparable across tasks. Each is a mean over completion tokens only, and the completions differ in length and entropy: one token for T1, a short JSON object for T2, a one- to five-token label for T3. The T3 run finished its epoch inside the 7 h budget (`stopped_on_time_budget: false`). Its pace settled at 18.0 s per optimiser step by step 100, against the 102 s per step of the earlier bf16 attempt.

#### 3.3.2 Practical implementation — the logic of each task, and of the baseline comparison

##### Task 1 — Risk clause recognition (CUAD, binary)

**The question put to the model**, once per (contract, category) pair across the 32 Yes/No categories:

> `Is the following contract text a "Cap On Liability" clause? Answer strictly "Yes" or "No".`

with the clause excerpt as `### Input:` and the single token-string `Yes` or `No` as the target completion.

**The design decision that makes the task honest: hard negatives.** As noted in §3.1.1, raw CUAD provides no text for an absent category, so a naively built Yes/No dataset is trivially gameable. The construction here instead:

1. For each contract, collect every category that *does* have real clause text — that contract's pool of genuine legal language.
2. For a category whose expert answer is present, emit the category's own text with the label `Yes`.
3. For a category whose expert answer is `"No"`, emit a **randomly chosen real clause from a different category of the same contract**, labelled `No` (seeded `random.seed(42)`).

The result is that *both* classes are authentic contract prose from the same document, so surface cues — length, formatting, legalese density — cannot separate them. The label depends on the interaction between the category named in the instruction and the content of the input, which is the skill under test. A sanity-check cell asserts that no placeholder string survives into any `No` input, and the notebook prints sampled negatives for inspection.

A second, smaller filter is applied silently and should be stated: where the `-Answer` column says a clause is present but the matching context cell is blank (the two columns disagree), the example is skipped rather than emitted with empty input.

**The split: contract-level, 85/15, `random_state=42`.** Contracts are split *first*, then examples are built from each side. Clause-level splitting would be leakage: a hard negative in validation could be the very same text the model saw as a positive in training, since every negative is borrowed from a real clause of the same contract. The documented sanity expectation follows from this — an input-ignoring model should score near 50%, not near 100%; a near-perfect validation F1 on the first try would be a leak signal, not a success.

**Class balance is applied to train only.** The training split is downsampled to a 1:1 positive:negative ratio (3,053/3,053); validation is left at its natural 558/1,650 (25.3% positive). The intent is a clean learning signal on a rare-positive task while preserving a realistic evaluation prior. §5.1 returns to the cost of that asymmetry.

**Prediction parsing is lenient.** Greedy decoding, `max_new_tokens=3`, then `"Yes" if "yes" in completion.lower() else "No"`. The consequence is explicit in the project's own documentation and is important for reading the results: this parse **never fails**, so task 1 reports no format-validity metric and a rambling completion is silently coerced into a class rather than counted as malformed. The coercion can push either way. A 3-token ramble that happens to contain "yes", such as an echo of the instruction's `"Yes" or "No"`, becomes `Yes`; every other ramble becomes `No`. The direction of the resulting bias therefore cannot be known in advance. Tasks 2 and 3 do report validity, because their formats can genuinely break.

##### Task 2 — Structured entity extraction (CUAD, JSON)

**The question**, once per (contract, category) pair across the 9 entity categories:

> `Extract the "Governing Law" from the contract text below. Return the result as a JSON object with the single key "Governing Law". If multiple values exist, return them as a list of strings. If the value is not present, use null.`

**The expected output is a contract, not just an answer** — one JSON object, no markdown fence, no prose, whose only key is the requested category:

```json
{"Governing Law": "Nevada"}
```

with a list when several entities exist (`{"Parties": ["Acme Inc.", "Beta LLC"]}`) and `null` when absent (`{"Expiration Date": null}`).

**Why this task is scored in three layers rather than one.** A prediction can be wrong in three qualitatively different ways, and conflating them would answer no research question:

| Failure mode | Example | What it means |
| :--- | :--- | :--- |
| Broken format | ```` ```json {"Governing Law": "Nevada"} ``` ```` or prose around the object | the model did not learn the output *contract* |
| Wrong content | `{"Governing Law": "Delaware"}` when gold is Nevada | the model did not learn the *extraction* |
| Almost-right content | `{"Governing Law": "the laws of the State of Nevada"}` | the model extracts but does not *normalise* |

Fine-tuning is expected to affect these differently, so the evaluation scores each example on three gated metrics (defined in §4.1.2). The same split *discipline* as task 1 applies — contract-level, seed 42 — but **not the same ratio: task 2 splits 80/20 (408/102 contracts), where task 1 splits 85/15** (see §3.2.2, stage 5). Parsing is **strict** — `json.loads` on the entire raw string plus an exact key-set check, with no repair of fences or prose, because the premise of the task is that the output feeds a downstream program.

##### Task 3 — Provision-type classification (LEDGAR, 100-way)

**The question**: pick exactly one of 100 provision-type labels, with the full menu inlined in the instruction:

> `Classify the following contract provision. Answer with exactly one label from this list: [Adjustments, Agreements, Amendments, ..., Waivers, Warranties, Withholdings].`

**Why the whole menu is in every prompt.** It costs 331 tokens of every example — roughly a third of the budget — but omitting it would make the baseline comparison meaningless: an un-fine-tuned model cannot be expected to guess a closed taxonomy it has never been shown. The menu is what makes the base model's failures *attributable* rather than inevitable. The price is paid in the input-trimming machinery described in §3.3.1.

**Training data is stratified, not natural.** Up to 100 examples per label are drawn from the 60,000-row LexGLUE train split (9,801 total; labels below the target cap out at whatever they have, minimum 23) and up to 20 per label from the 10,000-row validation split (1,945 total). The balanced *validation* set is itself a methodological choice: it means macro-F1 is not quietly dominated by the frequent provision types.

**Prediction parsing** takes the first line of the completion, strips and lowercases it, and looks it up in the 100-label table. A hit maps back to the canonical label; a miss becomes the sentinel `__INVALID__`, which is not in the label set and therefore counts as **wrong in every metric** — never as an abstention or a free pass. Generation is greedy, batched at 8, with `max_new_tokens` = longest label + 2.

##### The baseline protocol — what "versus its untrained self" means operationally

The deliverable of this project is not an absolute score; it is a **delta** against the same model without the adapter. For that delta to be attributable to the fine-tuning, everything else must be held constant. Three separate baseline notebooks (`llama_3.1_task_{1,2,3}_no_fine_tune.ipynb`) implement this, and the following are verified identical to the fine-tuned runs:

- **Same checkpoint**: `meta-llama/Meta-Llama-3.1-8B`, loaded 4-bit NF4, **with no adapter attached**.
- **Same validation examples**: each run saves the exact JSONL it evaluated, and each comparison notebook compares them **record by record**, not merely by count — the notebooks state plainly that if these differ, every number is meaningless.
- **Same prompt**: the identical `### Instruction / ### Input / ### Response:` template, ending at the same position; for task 3 the baseline receives the same full 100-label menu.
- **Same decoding**: greedy (`do_sample=False`) in every run, so a delta cannot be sampling noise. `max_new_tokens` is set per task to the smallest generous bound (3 for task 1, 128 for task 2, 16 for the task-3 baseline).
- **Same scoring code path**: each task's metrics are computed by the same functions over both runs.
- **One directory per run**: the comparison notebooks read each arm from its own output directory (`kaggle_output_task{1,2,3}_fine_tuned/`, `kaggle_output_task{1,2,3}_baseline/`). They no longer read the shared `kaggle_output/` folder, which several downloads have written into, so its root-level metrics belong to whichever run was pulled last (Appendix B).

Two asymmetries were found during this review and are *not* resolved; they are reported as caveats in §5.1 rather than smoothed over: the task-3 baseline runs bf16 where the fine-tuned run runs fp16, and the task-3 baseline relies on tokenizer truncation of the assembled prompt where the fine-tuned run trims the input text.

The remote-execution workflow itself is documented in `docs/kaggle/kaggle_connection_guide.md`: `kaggle/kernel-metadata.json` selects which notebook is pushed, data and the Hugging Face token are mounted as read-only Kaggle datasets, and `kaggle/run.ps1 -Wait` pushes, polls and downloads the artifacts.

---

## 4. Results

### 4.1 The evaluation metrics, and why each one

Each task is scored by a different metric set because each task can fail differently. The common thread is a deliberate ordering: **first ask whether the output is an answer at all, then ask whether it is the right one.** Scores that skip that gate flatter a generative model.

#### 4.1.1 Task 1 — binary classification metrics

| Metric | The question it answers | Why it is reported |
| :--- | :--- | :--- |
| Accuracy | How often is the prediction correct? | Intuitive, but **not the headline**: validation is 74.7% `No`, so an input-ignoring "always No" model scores ~75%. |
| Precision (per class) | When it says `Yes`, how often is it right? | Low precision = false alarms a reviewer must dismiss. |
| Recall (per class) | Of the real `Yes` clauses, how many did it catch? | **The operationally important one.** In legal review, missing a risk clause is worse than over-flagging one a human can reject. |
| **F1 (per class) and macro-F1** | A single score balancing precision and recall, averaged with classes weighted equally | **Headline.** Macro-averaging prevents the abundant `No` class from hiding failure on the rare `Yes` class (Sokolova & Lapalme, 2009). |
| Support | How many real examples of each class exist | Context, not a score: an F1 over 5 examples is far weaker evidence than one over 500. |
| Confusion matrix | The raw TP / FN / FP / TN counts | Shows *how* the model fails — a large FP cell is over-flagging, a large FN cell is missing clauses. Everything above derives from these four numbers. |

No format-validity metric exists for task 1, by construction: the lenient parse cannot fail (§3.3.2).

#### 4.1.2 Task 2 — three gated metrics for structured output

| Metric | The question it answers | Why it is defined this way |
| :--- | :--- | :--- |
| **1. `json_valid`** (the gate) | Did the model obey the output contract at all? | Valid **iff** `json.loads` succeeds on the *entire* raw string **and** the result is a dict whose key set is *exactly* the requested category. Markdown fences, leading prose, bare values, wrong-cased keys and extra keys are all invalid — deliberately, including things a human would forgive — because the premise is that a downstream program consumes this output without repair code. Stripping fences before parsing would report format discipline the model does not have. |
| **2. `exact_match`** | Of well-formed outputs, is the value exactly right? | `norm(pred) == norm(gold)` under *light* normalisation only: `None` stays `None`, scalars are stripped and lowercased, lists become **sorted tuples** (party order carries no meaning). Deliberately **no** fuzzy matching (`"State of Nevada"` ≠ `"Nevada"` — that is a normalisation failure, which is what this metric is for), **no** null leniency (predicting a value for an absent entity is a hallucination, the most damaging failure mode for a legal tool, and scores 0), and **no** scalar/list coercion. |
| **3. `token_f1`** | When the value is not exactly right, how close is it? | SQuAD-style bag-of-words F1 over whitespace tokens with multiset overlap (Rajpurkar et al., 2016), so a right-span/wrong-normalisation answer earns partial credit instead of collapsing to 0. For lists, a greedy one-to-one alignment in gold order, divided by `max(len(pred), len(gold))` — so both missing items *and* padding the answer with extra plausible parties cost score. |

The metrics are **gated**: an invalid output scores `exact_match = False` and `f1 = 0.0` without its content being inspected. This yields a per-example invariant that holds in aggregate and can be used to sanity-check any report of this task:

```
exact_match  ≤  token_f1  ≤  json_valid
```

**A correction, from the preprocessing audit in §3.2.2.** The list-handling halves of metrics 2 and 3 — sorted-tuple comparison and greedy one-to-one alignment — are **never exercised on this dataset.** All 3,085 generated task-2 targets are scalars (2,972 strings) or `null` (163); not one is a JSON list, because the preprocessing cleaner removed the `;` delimiter that `normalize_answer` splits on. `Parties` is therefore scored as a single space-joined string, and its 0.886 token-F1 is an **order-insensitive bag-of-words** comparison between two blobs of the same party names, not a per-party alignment. That is a much weaker claim than list F1 would be, and §4.2.2 is read accordingly. The alignment code is correct and would work on properly delimited data; it is simply dead here.

Two *gaps* between these three numbers are themselves diagnostics, and §4.2.2 uses them: `json_valid − token_f1` approximates "well-formed but wrong content", and `token_f1 − exact_match` approximates "right span, wrong normalisation". Reporting is **per category** as well as overall, because the categories are not equally hard — dates are short and templated, `Parties` is multi-valued free text — and a single aggregate would average an easy 0.98 against a hard 0.40 and hide exactly the structure that matters. The overall figure is a **micro**-average over all 631 examples, so categories with more examples weigh more; it is not the mean of the nine rows.

#### 4.1.3 Task 3 — four metrics, read as a cascade

| Metric | The question it answers | Why it is reported |
| :--- | :--- | :--- |
| **1. `valid_label_rate`** (the gate) | Is the answer even one of the 100 allowed labels? | The model generates free text and could invent a label or ramble. Anything unparseable is `__INVALID__` and counts as wrong. A low rate caps every metric below it from above. |
| **2. Accuracy** | How often is the predicted label the gold label? | Most intuitive, but the balanced validation set makes it near-equivalent to micro-F1 here, and it says nothing about *which* labels fail. |
| **3. Macro-F1** | Is the model good at *every* label, rare ones included? | **Headline**, because of the 137× class imbalance in the source data: per-label F1 averaged with every label weighted equally, so getting *Books* wrong hurts as much as getting *Governing Laws* wrong (Sokolova & Lapalme, 2009). |
| **4. Micro-F1** | Is the model right on *most* provisions? | The number reported on the LexGLUE benchmark, so it permits comparison against published results (Chalkidis et al., 2022). |
| Per-label P/R/F1 | *Which* labels fail? | The averages hide the distribution; this table is what points at the next experiment. |
| Top-15 confusions (gold → pred) | *Which* labels get mixed up? | Distinguishes a sensible near-synonym confusion from a random error — the difference between a label-design problem and a model problem (EDA Finding 4). |

### 4.2 Results analysis

All figures below were recomputed in this review directly from the `eval_metrics.json` files the runs produced, and match the comparison notebooks' loaders. Baseline = the same `Meta-Llama-3.1-8B` with no adapter, same prompts, same examples, same greedy decoding.

#### 4.2.1 Task 1 — risk clause recognition (2,208 validation examples)

| Metric | Baseline | Fine-tuned | Δ |
| :--- | ---: | ---: | ---: |
| Accuracy | 0.5611 | **0.9615** | +0.400 |
| **Macro-F1** (headline) | 0.5355 | **0.9507** | **+0.415** |
| `Yes` F1 (rare class) | 0.4263 | **0.9277** | +0.501 |
| `Yes` precision | 0.3183 | **0.8833** | +0.565 |
| `Yes` recall | 0.6452 | **0.9767** | +0.332 |
| `No` F1 | 0.6447 | **0.9738** | +0.329 |

Confusion matrices (rows = true `[Yes, No]`, columns = predicted):

```
baseline                       fine-tuned
            pred Yes  pred No              pred Yes  pred No
true Yes       360      198     true Yes      545       13
true No        771      879     true No        72     1578
```

**Interpretation.** The base model's dominant failure is **over-prediction of `Yes`**: 771 false positives against 198 false negatives. With `Yes` precision of 0.318 on a validation set that is only 25.3% positive, the base model is close to indiscriminate — the "everything looks like a risk clause" behaviour. Its accuracy of 0.561 is *worse* than the 0.747 an input-ignoring "always No" model would achieve, which is the clearest single statement of how little the un-adapted model contributes here.

Fine-tuning inverts the error profile: 72 false positives and 13 false negatives. Recall on the rare, operationally important `Yes` class reaches 0.977 — only 13 of 558 real risk clauses missed — at a precision of 0.883. For a review-assistance tool that ordering is the right one: the residual error is 72 clauses a human reviewer would dismiss, not 72 clauses a reviewer never sees. The model is now mildly **over-sensitive rather than random**, and that over-sensitivity is consistent with having been trained on a 1:1 balanced prior while being evaluated on a 25% positive one (§5.1).

Two cautions on reading this delta. First, because the parse is lenient, part of the baseline's 771 false positives may be *format* failures coerced into a class rather than genuine legal misjudgements — task 1 cannot separate those, and that is a limitation of the task-1 metric set, not a property of the model. Second, the sanity expectation holds: the fine-tuned score is high but not suspiciously perfect, and the contract-level split means the 0.96 accuracy cannot be leakage of the kind clause-level splitting would have introduced.

#### 4.2.2 Task 2 — structured entity extraction (631 validation examples)

| Metric | Baseline | Fine-tuned | Δ |
| :--- | ---: | ---: | ---: |
| `json_valid` | 0.4723 | **0.9968** | +0.525 |
| `exact_match` | 0.0903 | **0.6910** | +0.601 |
| `token_f1` | 0.1408 | **0.8193** | +0.679 |
| *Derived:* EM conditional on validity | 0.1913 | **0.6932** | +0.502 |
| *Derived:* F1 conditional on validity | 0.2982 | **0.8219** | +0.524 |

Per category (validation count in the last column):

| Category | BL `json_valid` | FT `json_valid` | BL EM | FT EM | BL F1 | FT F1 | n |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Document Name | 0.431 | 1.000 | 0.402 | **0.980** | 0.428 | 0.998 | 102 |
| Parties | 0.725 | 0.980 | 0.000 | **0.206** | 0.115 | 0.886 | 102 |
| Agreement Date | 0.000 | 1.000 | 0.000 | **0.840** | 0.000 | 0.840 | 94 |
| Governing Law | 0.000 | 1.000 | 0.000 | **0.920** | 0.000 | 0.955 | 87 |
| Expiration Date | 0.942 | 1.000 | 0.000 | **0.395** | 0.000 | 0.395 | 86 |
| Effective Date | 0.260 | 1.000 | 0.000 | **0.792** | 0.000 | 0.792 | 77 |
| Renewal Term | 0.977 | 1.000 | 0.000 | **0.535** | 0.327 | 0.689 | 43 |
| Notice Period To Terminate Renewal | 0.931 | 1.000 | 0.552 | **0.931** | 0.667 | 0.931 | 29 |
| Warranty Duration | 0.909 | 1.000 | 0.000 | **1.000** | 0.000 | 1.000 | 11 |

**Interpretation — this is where the three-layer metric earns its keep.**

*The format story.* The base model emits a usable single-key JSON object on fewer than half of all requests (0.472), and for **Agreement Date and Governing Law it is 0.000** — on those two categories it never once produced parseable JSON with the right key, answering in prose instead. Fine-tuning essentially solves this (0.997): 629 of 631 outputs are strictly valid. This is the largest structural gain in the whole project and is unambiguously about the output *contract*, not about law.

*The content story.* Isolating content quality by conditioning on validity (the ratio `exact_match / json_valid`, both means over the same denominator) shows the gain is **not** purely cosmetic: among well-formed outputs, exact match rises from 0.191 to 0.693 and token-F1 from 0.298 to 0.822. So fine-tuning taught both the envelope and the value. But the conditional view also tempers the headline: the raw exact-match gain is 60.1 percentage points, of which 50.2 survive the conditioning — so roughly a sixth of the headline delta is the base model simply not answering in the required shape, and five sixths is genuine content improvement.

*Where the fine-tuned model is still weak.* The per-category table localises the remaining error precisely:

- **Date normalisation is the weak spot, and the preprocessing audit in §3.2.2 explains why.** `Expiration Date` reaches only 0.395 EM and `Renewal Term` 0.535, despite perfect format compliance. For the four date categories the gold target is not a date at all but an **unseparated digit string** produced by this project's own cleaner (`12/31/14` → `123114`), so the answer appears verbatim in the input text in only **1.0%** of `Expiration Date` examples. The model is not failing to read a date; it is failing to infer an arbitrary digit encoding — and 0.395 is better described as "inferred the encoding 39.5% of the time". Note too that for `Expiration Date` and `Effective Date`, F1 equals EM exactly: a digit string is one token, so there is no partial credit to earn and a near-miss scores zero. Contrast `Agreement Date` at 0.840, where the date is usually stated explicitly near the start of the clause and so is easier to re-encode.
- **The two strongest categories are the two that are closest to copying.** `Document Name` gold **is** the input string verbatim in 98.2% of examples, and its 0.980 EM is correspondingly a copy task rather than extraction; `Governing Law` gold is a substring of the input in 89.5% of examples, and scores 0.920. The ranking of the per-category table tracks "how much of the answer is already present in the input" far more closely than it tracks any notion of legal difficulty.
- **`Parties` shows "right content, wrong exactness": 0.206 EM against 0.886 token-F1** — but the gap is smaller evidence than it first appears. Per §3.2.2 and the correction in §4.1.2, the `;` separator was stripped in preprocessing, so `Parties` is a single space-joined blob and the 0.886 is an **order-insensitive bag-of-words** overlap between two such blobs. Inspection of the generated data shows the input often contains nearly the same tokens as the target in a different order, so a high bag-of-words score is close to the floor for a model that copies competently. The honest reading is: the right party *names* are being surfaced; nothing here establishes that the model identifies them as a correctly delimited list.
- **`Warranty Duration`'s perfect score is degenerate.** Its 1.000 EM is on **n = 11**, and — decisively — its gold value is the constant string `"Yes"` for all 75 examples in the dataset, because raw CUAD's `Warranty Duration-Answer` is a presence column, not a duration (§3.2.2, stage 4). The model learned to emit `{"Warranty Duration": "Yes"}` unconditionally. This row should be excluded from any summary of extraction quality.
- **Small supports demand caution generally.** `Notice Period` scores 0.931 on **n = 29**. Encouraging but statistically thin; not a headline capability.

One further detail worth noting: the baseline's `json_valid` is *high* for `Expiration Date` (0.942), `Renewal Term` (0.977) and `Warranty Duration` (0.909) while its content scores are 0.000. That combination — well-formed but always wrong — is the "knows the format, not the value" quadrant, and it is the mirror image of `Agreement Date` and `Governing Law`, where the base model is plausibly capable of the content but cannot express it. The two patterns together are why a single aggregate score would have been uninterpretable.

#### 4.2.3 Task 3 — provision-type classification (1,945 validation examples, 100 labels)

| Metric | Baseline | Fine-tuned | Δ |
| :--- | ---: | ---: | ---: |
| `valid_label_rate` | 0.4010 | **0.9964** | +0.595 |
| Accuracy | 0.0653 | **0.7686** | +0.703 |
| **Macro-F1** (headline) | 0.0655 | **0.7509** | **+0.685** |
| Micro-F1 | 0.0932 | **0.7700** | +0.677 |

**Interpretation.**

*The baseline is near-useless, and mostly for format reasons.* On **1,165 of 1,945 examples (59.9%)** the base model emitted a string that is not one of the 100 labels, despite being shown the full menu in every prompt. Those are wrong predictions by definition, so they cap accuracy and both F1 scores from above. The baseline's own top-confusion table makes this concrete: 228 of the top-15 confusion mass is the `__INVALID__` sentinel, five gold labels (*Entire Agreements*, *Non-Disparagement*, *Severability*, *Litigations*, *Waiver Of Jury Trials*) produced an invalid answer on **all 20** of their validation examples, and *Publicity* on 19 of 20. A baseline macro-F1 of 0.066 should therefore **not** be read as "the base model has 6.6% of the legal knowledge required"; it is closer to "the base model will not answer this question in the requested form." Its accuracy of 0.065 is about six times the 0.01 a uniform random guesser over 100 labels would achieve — above chance, but nowhere near usable.

*The fine-tuned model's errors are a different species.* Valid-label rate reaches 0.996 (7 invalid answers in 1,945), and **98 of 100 labels improved, 0 regressed**, with 2 unchanged. The largest single-label gain is *Closings* (0.000 → 1.000). What remains is overwhelmingly **label-taxonomy ambiguity**, not random error — every one of the top confusions is a near-synonym pair that LEDGAR keeps separate:

| Gold → predicted | Count |
| :--- | ---: |
| Applicable Laws → Governing Laws | 9 |
| Withholdings → Tax Withholdings | 9 |
| Integration → Entire Agreements | 9 |
| Jurisdictions → Consent To Jurisdiction | 8 |
| Definitions → Defined Terms | 8 |
| Successors → Binding Effects | 8 |
| Authorizations → Authority | 8 |
| Survival → Warranties | 8 |
| Defined Terms → Definitions | 7 |
| Modifications → Amendments | 7 |
| Indemnifications → Indemnity | 7 |
| General → Payments | 7 |
| Enforcements → Specific Performance | 6 |
| Waivers → No Waivers | 6 |
| Consent To Jurisdiction → Jurisdictions | 6 |

This directly confirms EDA Finding 4, which predicted these pairs *before* training. Note also the **symmetry** of several pairs (Definitions ↔ Defined Terms at 8/7; Jurisdictions ↔ Consent To Jurisdiction at 8/6): the model is not systematically preferring one label, it genuinely cannot separate the two. A meaningful share of the remaining 23.1% error is arguably irreducible without merging labels — which is a finding about the label set, not only about the model.

*Hardest labels after fine-tuning* (F1, validation support): `Assigns` 0.000 (n=3), `General` 0.250 (n=20), `Powers` 0.400 (n=15), `Enforcements` 0.412 (n=20), `Applicable Laws` 0.424 (n=20). Two patterns: ultra-rare labels (`Assigns` has only 31 rows in the full LEDGAR train split, so it capped at 31 after stratification — one of just three labels below the 100 target, alongside `Books` at 23 and `Qualifications` at 47 — and it has 3 validation examples), and semantically vague labels (`General` is almost definitionally a catch-all, and is most often confused with `Payments`).

*Versus published work.* Chalkidis et al. (2022) report BERT at ≈87.6 micro / 81.8 macro-F1 and Legal-BERT at ≈88.5 / 82.4 on this exact LexGLUE LEDGAR configuration. The fine-tuned model here reaches **77.0 micro / 75.1 macro**, i.e. roughly 10 points of micro-F1 below the encoder baselines. That gap should be read with three differences in mind, none of which favour the present setup: this is a **decoder-only generative** model asked to *write* the label rather than a classifier choosing among 100 logits; it was trained on a **stratified 9,801-example sample** rather than the full 60,000-row split; and it had **one epoch** of a rank-16 adapter rather than full fine-tuning. The result is therefore best stated as *a QLoRA-adapted 8B generative model gets within ~10 micro-F1 points of a fully fine-tuned BERT on 16% of the training data and one epoch* — not as a competitive leaderboard entry.

#### 4.2.4 Cross-task synthesis

| | T1 (binary) | T2 (JSON) | T3 (100-way) |
| :--- | ---: | ---: | ---: |
| Format/validity gate, BL → FT | n/a (lenient parse) | 0.472 → 0.997 | 0.401 → 0.996 |
| Headline metric, BL → FT | 0.536 → 0.951 (macro-F1) | 0.141 → 0.819 (token-F1) | 0.066 → 0.751 (macro-F1) |
| Headline Δ | +0.415 | +0.679 | +0.685 |

Three conclusions hold across all three tasks:

1. **Fine-tuning wins decisively everywhere**, and the margin is widest where the output *format* is itself part of the task. T2's +0.68 and T3's +0.69 are both larger than T1's +0.42, and T1 is the only task whose output format is a single word the base model can stumble into.
2. **A large share of every headline delta is output discipline, not legal reasoning.** This is the single most important caveat on the whole result set, and it is quantified rather than asserted: 59.9% of baseline T3 predictions and 52.8% of baseline T2 predictions were not valid answers at all. Where the data allows the two to be separated — T2's conditional exact match — content quality still improves substantially (0.191 → 0.693), so the gain is not *only* formatting. T1 and T3 offer no equivalent decomposition.
3. **The baselines are fair, which is what makes the deltas quotable.** Same checkpoint, same prompt including T3's full label menu, same examples verified record by record, same greedy decoding, same scoring code. The two residual asymmetries found in this review are stated in §5.1 rather than buried.

---

## 5. Conclusions

This project set out to test whether a general-purpose open-weight LLM can be made useful on three distinct legal contract-review tasks using only free-tier compute, and whether the improvement can be *attributed* rather than merely claimed. On both counts the answer is yes, with explicit limits.

**What was built.** A single environment-aware notebook pipeline per task that runs unchanged locally (data preparation on CPU) and on Kaggle (training and evaluation on one T4), driven as a batch job from a GPU-less laptop. One base checkpoint, `Meta-Llama-3.1-8B`, three QLoRA adapters, three matched no-adapter baselines, three comparison notebooks, and a persisted artifact set (`eval_metrics.json`, `eval_report.txt`, `train_metrics.json`, the generated JSONL, and the full run log) for every run.

**What was measured.** Fine-tuning improved the headline metric on all three tasks: macro-F1 0.536 → 0.951 on binary risk-clause recognition; token-F1 0.141 → 0.819 (and strict JSON validity 0.472 → 0.997) on nine-category entity extraction; macro-F1 0.066 → 0.751 (valid-label rate 0.401 → 0.996) on 100-way provision classification. On task 3, 98 of 100 labels improved and none regressed.

**What the numbers mean, stated conservatively.** Three claims are supported: (i) a rank-16 QLoRA adapter on an 8B base model is sufficient to teach each of these three output contracts almost perfectly, on free hardware, in one epoch; (ii) the residual errors are *structured and diagnosable* rather than random — over-sensitivity on task 1, date/term normalisation and list exactness on task 2, near-synonym label pairs on task 3; and (iii) the comparison is fair enough that the deltas are attributable to the weights. One claim is **not** supported: that these deltas measure acquired legal comprehension. A majority of the baseline's failures on tasks 2 and 3 are refusals to answer in the requested form, and only task 2 permits separating format from content.

**A fourth conclusion, added after auditing the pre-processing (§3.2).** Task 2's per-category results are shaped more by the data pipeline than by the model. Three of its nine categories are compromised: `Warranty Duration`'s target is a constant string, so its perfect score is meaningless; the four date categories' targets are unseparated digit strings created by this project's own text cleaner, so the model is scored on inferring an arbitrary encoding rather than on reading a date; and `Parties` lost its `;` delimiter to the same cleaner, which silently disabled the list-handling path in both the data builder and the metric. Meanwhile the two best-scoring categories, `Document Name` (0.980) and `Governing Law` (0.920), are the two closest to verbatim copying — the gold answer *is* the input in 98.2% and a substring of it in 89.5% of cases respectively. Task 2's headline figures remain valid as a fine-tuned-versus-baseline delta, because both arms were scored on identical data; but the per-category table is not yet a credible measurement of entity-extraction ability, and the cheapest high-value work in this project is repairing the pipeline rather than training anything (§5.2, Tier 0).

**What was found about the engineering, which is reusable.** The free-tier constraint produced findings worth more than the task scores to anyone repeating this: bf16 is a trap on a Turing GPU because PyTorch reports it as supported while it runs emulated; sequence caps must be enforced by trimming the *input*, because truncating an assembled prompt deletes a tail-positioned label and produces NaN loss under completion-only training; peak VRAM for an 8B model at this vocabulary size is dominated by the logits tensor, not the weights; peft's `prepare_model_for_kbit_training` silently upcasts frozen embedding and output layers to fp32, costing ~3 GB once autocast's cached copy is counted; and the dtype split it leaves behind breaks `generate()` outside autocast, which cost one run 5.5 hours of successful training. Pre-flight checks that exercise the real training *and* the real evaluation path before training starts are the cheap general lesson.

### 5.1 Failure modes — what currently limits this research

These are ordered by how much they constrain the conclusions, and every one is verified against the artifacts rather than guessed.

**A. Statistical weakness of the evidence**

1. **Single run, single seed, one epoch, per task.** Every number is from one training run at `seed=42` with `num_train_epochs=1`. Fine-tuning outcomes are known to vary materially with the random seed alone — through weight initialisation and data order — even holding everything else fixed (Dodge et al., 2020). With no seed replicates, **no variance estimate exists**, so a difference of one or two points anywhere in this report is uninterpretable, and the headline deltas have no confidence interval.
2. **No significance testing.** The fine-tuned-vs-baseline comparisons are paired by construction (same examples, same prompts), which is precisely the setting McNemar's test was recommended for in classifier comparison (Dietterich, 1998), and bootstrap resampling would give intervals on the generative metrics (Efron & Tibshirani, 1993). Neither is computed. The deltas here are large enough that significance is not seriously in doubt, but that is an argument, not a result.
3. **No test-split evaluation.** All reported figures are **validation**-set figures. For task 3 this is a real missed opportunity: LexGLUE ships a 10,000-row official test split that was held out and never used, and it is the split published results are reported on — so the comparison to BERT in §4.2.3 is already not apples-to-apples on that axis. CUAD ships no official split, so tasks 1 and 2 have only the project's own contract-level divisions (85/15 for task 1, 80/20 for task 2). In fairness to the current numbers, no selection was made on validation. There is no in-training evaluation (`eval_strategy` is left at `"no"`), no checkpoint selection and no early stopping, and the final weights of a single configuration are scored once. The validation figures are therefore not optimistically biased *yet*. That stops being true as soon as any hyperparameter is tuned against them, so the sweep in §5.2 Tier 2 needs a separate test split first.
4. **Thin supports in the per-category and per-label tables.** T2's `Warranty Duration` (n=11) and `Notice Period` (n=29) carry headline-looking scores on very few examples. T3's validation set, nominally 20 per label, actually ranges 0–20: `Assigns` n=3, `Qualifications` n=8, `Powers` n=15, `Venues` n=19.

**B. Metric and task-design artifacts**

5. **Task 1 has no format-validity metric, and its parse is lenient.** `"yes" in completion.lower()` never fails, so every base-model ramble is silently coerced into a class. This means (i) task 1 cannot distinguish a legal misjudgement from a format failure, and (ii) the baseline's 771 false positives are of unknown composition. Tasks 2 and 3 both show the base model failing the format gate ~50–60% of the time, which makes it likely that a non-trivial share of T1's baseline error is the same phenomenon measured through a lens that cannot see it. T1's +0.415 delta is therefore the *least* decomposable of the three.
6. **Task 1 trains on a balanced prior and is evaluated on a natural one.** Training is downsampled to 3,053/3,053; validation is 558/1,650 (25.3% positive). The fine-tuned model's residual error profile — 72 false positives versus 13 false negatives — is exactly what that mismatch predicts. The high `Yes` recall (0.977) may therefore be partly an artifact of the training prior rather than a property of the learned representation, and the precision/recall trade-off was never deliberately chosen or tuned (no threshold exists to tune in a generative setup).
7. **Task 3's macro-F1 is mechanically understated.** `Books` has zero validation examples (§3.1.2) but is still included in the macro average with F1 = 0.0, so macro-F1 is averaged over 100 labels when only 99 are scorable. Recomputed over the 99 labels with support > 0, the fine-tuned macro-F1 is **0.7585 rather than the reported 0.7509** — a 0.76-point understatement (the baseline's is 0.0662 versus 0.0655). The reported number is conservative, not inflated, but it is not the number it is labelled as.
8. **Task 2 measures normalisation, and exact match punishes semantically correct answers.** Gold is CUAD's expert-*normalised* value, not a quotable span. A prediction of `"the laws of the State of Nevada"` where gold is `"Nevada"` is legally correct and scores `exact_match = 0`. Token-F1 mitigates this but is bag-of-words with no semantics: `"NY"` versus `"New York"` scores 0. The `Expiration Date` figure of 0.395 should be read as "0.395 of answers are in canonical form", not "0.395 of answers are right".
9. **Task 2's per-category ranking tracks input overlap, not legal difficulty.** Measured in §3.2.2, stage 4: the gold value *is* the input verbatim for 98.2% of `Document Name` examples and a substring of it for 89.5% of `Governing Law` examples — the two top-scoring categories. Conversely `Expiration Date`'s target appears in the input 1.0% of the time. The per-category table is therefore substantially a measurement of how much copying each category permits, and should not be presented as a difficulty ordering over legal concepts.
10. **Task 2's list-valued path is dead code, so `Parties` is scored as a blob.** None of the 3,085 targets is a JSON list, so the sorted-tuple comparison and greedy one-to-one alignment never run; `Parties`' 0.886 token-F1 is an order-insensitive bag-of-words overlap. The metric design anticipated a case the data never presents, and the instruction's list clause is never demonstrated to the model. Any future claim about multi-party extraction quality requires rebuilding the data first (§5.2, Tier 0 items 1–2).
11. **Silent data loss in task 1.** There are two separate mechanisms:
    - Where CUAD's `-Answer` column says a clause is present but the paired context cell is blank, the example is dropped. The number dropped is not logged, so the realised positive rate is not fully accounted for.
    - **50 training prompts (0.82%) and 6 validation prompts (0.27%) are already ≥ 1,024 tokens**, measured with the Llama-3.1 tokenizer on the run's own JSONL. In training, `max_length=1024` truncates from the end, so those 50 examples lose their `Yes`/`No` target and contribute nothing. In evaluation, both arms truncate the assembled prompt at 1,024 tokens, so on those 6 examples the model never sees the `### Response:` cue.

    This is exactly the failure that produced NaN loss in task 3 (§3.3.1, item 2), and task 3's fix — trimming the input text, not the assembled prompt — was never back-ported. The effect on the reported numbers is small and applies equally to both arms, but the T1 pre-flight checks that would have reported it are commented out in the notebook.

**B2. Pre-processing defects (surfaced by the §3.2 audit; these are the most actionable items in this list)**

12. **The CUAD cleaner strips every non-alphanumeric character from every cell, including the ground truth.** `scripts/preprocess_values_cuad.py` applies `re.sub(r'[^a-zA-Z0-9\s]', '', ...)` across the whole dataframe. The three measured consequences are: (a) every date target becomes an unseparated, ambiguous digit string (`12/31/14` → `123114`), which is the direct cause of task 2's weakest scores; (b) 99.2% of `Parties-Answer` values lose the `;` delimiter that the downstream `normalize_answer` function depends on, silently disabling all list handling; and (c) clause text loses `.` `,` `%` `$` `(` `"` entirely — `0.0%` of a 400-example sample of task-1 inputs contains any of them, so `10%` reads as `10`. For a corpus where a percentage, a currency symbol and a quoted defined term all carry legal meaning, (c) is a real loss of signal on both CUAD tasks, and it is **not** applied to task 3, so the three tasks do not feed the model comparable text. Nothing downstream detects or restores any of this.
13. **`Warranty Duration` is not an extraction task as implemented.** Its gold value is the constant string `"Yes"` for all 75 examples, because raw CUAD's `Warranty Duration-Answer` is a presence column (75 Yes / 435 No) and the builder keeps only rows with non-empty context. The reported 1.000 exact match is a constant-target artifact and must not be quoted as an extraction result. The same column is, correctly, a Yes/No signal — it belongs in task 1's category set, not task 2's.
14. **12 of 32 task-1 category names are mangled in the prompt.** Header cleaning strips hyphens and slashes, so the model is asked about `NonCompete`, `AntiAssignment`, `RevenueProfit Sharing`, `RofrRofoRofn` and `UnlimitedAllYouCanEatLicense`. This is applied identically to both arms, so it does not bias the comparison, but it degrades prompt legibility for both and is trivially fixable.
15. **The two CUAD tasks use different split ratios without a stated reason.** Task 1 splits 85/15 (433/77 contracts), task 2 splits 80/20 (408/102). Nothing in the code or docs justifies the difference, and three task-2 documents (`TASK2_EVALUATION_METRICS.md`, `TASK2_LOGIC.md`, `TASK2_ENTITY_EXTRACTION_PLAN.md`) incorrectly describe task 2 as 85/15. The ratios are internally consistent within each task, so no result is invalidated, but cross-task comparisons of validation-set size carry an unexplained difference.
16. **`normalize_answer`'s self-tests pass on inputs the pipeline never produces.** Its assertions use hand-written strings that still contain `;` and `/`, so they validate the function in isolation while the real upstream data has had both removed. This is the mechanism by which defect 12(b) went unnoticed, and it is a general lesson: unit tests on synthetic fixtures do not substitute for an assertion over the generated dataset.

**C. Fairness asymmetries between fine-tuned and baseline runs (task 3)**

17. **Different compute dtype.** The T3 fine-tuned run uses `fp16` throughout (`bnb_4bit_compute_dtype=torch.float16`, fp16 autocast at generation); the T3 baseline notebook uses `bnb_4bit_compute_dtype=torch.bfloat16` and bf16 autocast. On a cc-7.5 card this is an unequal numerical path between the two arms being compared. The effect on a greedy argmax is very likely negligible, but it is unmeasured, and it violates the "change one thing" rule the rest of the protocol upholds.
18. **Different sequence-capping mechanism.** The fine-tuned run trims the *provision text* via the shared `build_prompt()`; the baseline notebook builds the prompt inline and relies on `tokenizer(..., truncation=True, max_length=MAX_SEQ_LENGTH)` on the *assembled* prompt. For any prompt exceeding 1,024 tokens the baseline therefore loses its tail — which includes the `### Response:` cue the model is supposed to complete from. The fine-tuned run logged **9 of 1,945 validation provisions (0.46%)** as over-budget, so at most ~0.5% of baseline prompts are affected; the bound is small and verified, but the defect is real and the project's own documentation explicitly required one shared prompt builder "otherwise the comparison silently drifts."

**D. Reproducibility gaps**

19. **The environment is only partly pinned.** The core training stack *is* pinned: every Kaggle notebook installs `transformers==4.55.4`, `bitsandbytes==0.46.1`, `accelerate==1.7.0`, `peft==0.15.2` and `trl==0.20.0` in its first cell. Three gaps remain:
    - `datasets`, `torch` and the CUDA/driver versions come from the Kaggle base image unpinned and are not logged, so a later image can change them silently.
    - The local `requirements.txt` is entirely unpinned: 21 bare package names, with `pandas` and `scikit-learn` listed twice. The local environment has since drifted to `trl` 1.5.1, `transformers` 5.9.0 and `peft` 0.19.1, so local experiments do not run the same library code as the Kaggle runs.
    - Code comments in all training notebooks say "trl 1.x" while the pinned version is 0.20.0.

    This falls short of the project's own stated principle of pinning the environment.
20. **Metrics were computed by the in-process trained model, never by the saved adapter.** All three evaluations score `trainer.model` (or the live `model` object) immediately after training. The adapters were saved and retrieved, but no run has demonstrated that loading a saved adapter from disk reproduces the reported metrics. Every published number is therefore un-reconfirmed against the artifact that was actually shipped.
21. **Tasks 1 and 2 ran bf16 on a T4** — recorded as `"bf16": true` in both `train_metrics.json` files, on the same accelerator the T3 post-mortem identified as lacking bf16 tensor cores. They almost certainly paid the same emulation penalty; recorded throughput is 0.197 and 0.244 samples/s against T3's 0.445 at fp16, a 1.8–2.3× gap. Batch size and length-grouping also differ between the runs, so this is consistent with the penalty rather than an isolation of it — but it means **T1 and T2 cost roughly twice what they needed to**, and it makes re-running them cheap enough to be worth doing. Relatedly, `docs/task_3/TASK3_NEXT_STEPS.md` states that T1/T2's bf16 setting applied to "the 1B smoke model"; the production `train_metrics.json` for both shows `model_name: meta-llama/Meta-Llama-3.1-8B`, so that note is wrong and should be corrected.
22. **Stale project documentation.** `README.md` and `CLAUDE.md` both still describe T2 and T3 as "training run pending" although completed adapters, baselines, comparison notebooks and eval metrics exist for both. `CLAUDE.md` also states CUAD's `master_clauses.csv` holds 545 contracts; the file in this repository has **510 rows**, matching the published CUAD figure (Hendrycks et al., 2021). `docs/task_3/TASK3_EVAL_METRICS.md` gives the label-menu length as 341 tokens where the EDA and the run log distinguish 331 tokens of instruction from 341 including scaffolding. `docs/task_2/TASK2_EVALUATION_METRICS.md` states a contract-level 85/15 split for task 2, where the code uses `test_size=0.2`; `TASK2_LOGIC.md` and `TASK2_ENTITY_EXTRACTION_PLAN.md` repeat the 85/15 figure (only `TASK2_NOTEBOOK_CELL_GUIDE.md` correctly says 80/20). `TASK2_EVALUATION_METRICS.md` also describes list-valued targets and greedy list alignment as live behaviour, which the generated data shows they are not. Three more documents are out of date:
    - `docs/task_3/TASK3_NEXT_STEPS.md` is dated 2026-07-26 and still lists the evaluation cells, the T3 baseline and the comparison notebook as not done, although all three exist with results.
    - `docs/llm_finetuning_parameters.md` describes an early configuration that no run used: `q_proj`/`v_proj` targets only, batch 4 × accumulation 1, `fp16`, `device_map="auto"` and `max_seq_length=2048`. It should not be read as the configuration behind any result here; the parameter tables in §3.3.1 replace it.
    - `docs/task_1/TASK1_EVALUATION_METRICS_ASSESSMENT.md` predicts that the lenient parse pushes the baseline towards `No`, but the measured baseline over-predicts `Yes` (771 FP vs 198 FN). As §3.3.2 notes, the coercion can go either way.
23. **Versioned data does not fully match the runs.** Several items are now in git: `docs/RESULTS_OVERVIEW.md`, the LEDGAR JSONL under `ledgar/`, and the task-2 JSONL under `cuad/`. The tracked LEDGAR and task-2 files were checked and are **record-for-record identical** to the files each run consumed. One mismatch remains: the tracked **task-1** files `cuad/train/cuad_train.jsonl` (1,282 rows) and `cuad/validation/cuad_validation.jsonl` (448 rows) are the *smoke-test* data, not the 6,106 / 2,208 rows the production run trained and evaluated on. Those exist only inside the git-ignored `kaggle_output_task1_fine_tuned/`. More generally, every production `kaggle_output*/` directory is git-ignored by design, so no metrics file or adapter behind §4.2 is versioned alongside the code that produced it. The only run output in git is the one that matters least: `kaggle_output_3.2B_Llama_smoketest/` is tracked, including its adapter and checkpoints.

**E. Scope limits**

24. **One model family, one adapter configuration.** Only `Meta-Llama-3.1-8B` at `r=16` with `q/k/v/o_proj` targets was trained. No rank sweep, no learning-rate sweep, no epoch sweep, and no comparison against the alternatives the project's own model-selection review recommended (`Qwen2.5-7B-Instruct`, or the legal-domain-pretrained `Saul-7B`; Colombo et al., 2024). Nothing here separates "QLoRA works" from "this particular configuration works".
25. **No comparison against cheaper baselines.** A linear classifier over TF-IDF features is a strong and often-forgotten baseline for exactly this kind of provision classification, and encoder fine-tuning is the published state of the art on LEDGAR (Chalkidis et al., 2022). Neither was run, so the *cost-effectiveness* of an 8B generative model for T1 and T3 is unestablished — these are classification tasks being solved with a text generator.
26. **Clause-level inputs only.** Both CUAD tasks consume pre-extracted clause excerpts, not whole contracts. The practically important step — locating the clause in a 50-page agreement before classifying it — is outside the current scope, so none of these numbers describe end-to-end contract review.
27. **No measurement of what the adapters broke.** LoRA is reported to forget less than full fine-tuning while also learning less (Biderman et al., 2024), but no general-capability or cross-task evaluation was run, so the trade-off is unmeasured here. In particular, nothing tests whether the T1 adapter still performs T2, or whether any adapter retains general instruction-following.

### 5.2 Next steps

Ordered so that the cheapest items that most strengthen the existing claims come first.

**Tier 0 — repair the pre-processing (no GPU time at all; highest value per hour in the whole list)**

These items come first because §3.2's audit shows they currently bound what task 2 can possibly score, and because every one of them is a data-build change that can be validated locally on a CPU before any GPU is booked.

1. **Stop cleaning the ground-truth columns.** Apply `clean_text` to the clause *context* columns only, and leave the `-Answer` columns untouched — or, better, replace the blanket `[^a-zA-Z0-9\s]` strip with a targeted cleaner that preserves `.` `,` `%` `$` `/` `;` `(` `)` and `"`. This single change restores real dates (`12/31/14` instead of `123114`), restores the `;` party delimiter, and stops deleting percentages and currency symbols from legal text. Then re-generate the JSONL and re-run tasks 1 and 2. Expect the largest single movement in the whole result set on the four date categories, whose targets currently appear in the input ~1–10% of the time.
2. **Add a generated-dataset assertion, not just a unit test.** After building the JSONL, assert over the *real* data what the unit tests currently assert over fixtures: that at least one target is a list for `Parties`, that no category has a single constant target, and that each category's target appears in its input at some minimum rate (or is explicitly flagged as a normalisation target). Defect 16 in §5.1 exists precisely because `normalize_answer`'s self-tests were run on hand-written strings containing delimiters the pipeline had already removed.
3. **Move `Warranty Duration` out of task 2.** As implemented it is a presence flag with a constant `"Yes"` target, not an entity. Either relabel it from the CUAD column that actually holds a duration, or drop it from the task-2 category set and add it to task 1's Yes/No categories where it belongs — and re-report task 2's overall figures without it, since it currently contributes a free 1.000.
4. **Un-mangle the category names in the prompt.** Keep the raw CUAD category strings for display in the instruction while using the cleaned names internally as dictionary keys, so the model is asked about `Non-Compete` rather than `NonCompete`. Apply to both arms so the comparison stays matched.
5. **Reconcile the two split ratios** (85/15 for task 1, 80/20 for task 2) to a single documented value, and correct `TASK2_EVALUATION_METRICS.md`, `TASK2_LOGIC.md` and `TASK2_ENTITY_EXTRACTION_PLAN.md`, which state 85/15.
6. **Back-port task 3's length guard to task 1.** Trim the clause text to a tokenizer-derived budget inside one shared prompt builder, as T3's `build_prompt()` does, so the 50 over-length training prompts keep their target. Then re-enable T1's commented-out pre-flight checks, at least the sequence-length check, which would have reported this. Both changes are local, CPU-only edits.

**Tier 1 — make the current results defensible (hours of GPU time, no new design)**

7. **Close the two task-3 fairness asymmetries and re-run the baseline.** Switch the baseline notebook to `fp16` and to the shared `build_prompt()`, so the only difference between the arms is the adapter. Cost: one baseline run. This removes failure modes 17 and 18 outright.
8. **Evaluate the saved adapters from disk.** Load each `llama-3.1-8B-*-task*` adapter onto the base model in a fresh kernel and re-score. This converts every number in §4.2 from "what the training process reported" into "what the shipped artifact does", and closes failure mode 20.
9. **Report on the held-out test split for task 3.** The 10,000-row LexGLUE test split exists and is the basis of published numbers. Stratify it the same way as validation and re-score both arms; report validation and test separately. This makes the comparison to Chalkidis et al. (2022) legitimate.
10. **Add confidence intervals and a paired significance test.** Bootstrap intervals over the validation examples for every headline metric (Efron & Tibshirani, 1993), plus McNemar's test on the paired fine-tuned/baseline predictions for T1 and T3 (Dietterich, 1998). Both are pure post-processing of predictions already generated — if the per-example predictions are persisted alongside the metrics, which they currently are not and should be.
11. **Fix the macro-F1 denominator for task 3** (exclude zero-support labels, or report both figures explicitly). Have each run write `pip freeze`, `torch.__version__` and the CUDA version into `train_metrics.json`, so the unpinned Kaggle-image packages are at least recorded. Pin `requirements.txt` to the same versions. Finally, correct the stale statements identified in §5.1 (items 21–23): in `README.md`, `CLAUDE.md`, `TASK3_NEXT_STEPS.md` and `llm_finetuning_parameters.md`, and in the tracked task-1 JSONL, which is still the smoke-test data.
12. **Add a format-validity metric to task 1.** Replace the lenient substring parse with a strict one (`completion.strip().lower() in {"yes","no"}`) scored alongside the lenient one. This costs nothing, makes T1 decomposable like T2 and T3, and would settle how much of the baseline's 771 false positives are format failures.

**Tier 2 — seed and sweep (days of GPU time)**

13. **Three seeds per task, minimum.** Re-run each fine-tune at seeds 42/43/44 and report mean ± range. This is the single largest upgrade to the strength of the evidence (Dodge et al., 2020) and is the prerequisite for any claim about a difference smaller than several points.
14. **Re-run T1 and T2 at `fp16` with batch 2 × accumulation 4 and `group_by_length`,** matching T3. On the recorded throughput gap this should roughly halve their cost, making the seed replicates in item 13 affordable.
15. **A small, pre-registered hyperparameter sweep**: LoRA rank {8, 16, 32} with `alpha` scaled to keep *alpha/r* fixed, epochs {1, 2}, target modules {attention-only, attention + MLP}, and a short linear warmup (e.g. 3% of steps) against the current none. Fix everything else and report the full grid, so the configuration choice stops being arbitrary. Select on a development split carved out of the *training* data, and keep the current validation sets (and, for T3, LexGLUE's test split) untouched. Tuning on the validation sets would make them optimistically biased (§5.1, item 3).

**Tier 3 — attack the identified error modes directly**

16. **Task 2 date and term normalisation — *after* Tier 0 item 1.** `Expiration Date` (0.395 EM) and `Renewal Term` (0.535 EM) are the two weakest categories, but the diagnosis has changed with the §3.2 audit: for the date categories the dominant cause is the digit-string target, not relative phrasing. Restore real dates first, then re-measure; only the residual error after that is worth attacking with oversampling, explicit normalisation examples in the instruction, or a date parser applied as post-processing with both raw and post-processed scores reported. `Renewal Term` is the one case where the original diagnosis still stands — its targets are phrases like `"successive 1 year"`, genuinely requiring normalisation of a relative expression.
17. **Task 2 list-valued `Parties` — *after* Tier 0 items 1–2.** The 0.206 EM / 0.886 F1 gap cannot be diagnosed as an exactness problem until the `;` delimiter is restored and `Parties` targets are actually lists; at present the F1 is a bag-of-words overlap between two blobs and the list metric is dead code. Once the data carries real lists: add a list-level precision/recall breakdown so "missed a party" and "invented a party" are counted separately (the latter being the safety-critical one), and test whether constrained or grammar-guided decoding (Willard & Louf, 2023) raises exactness.
18. **Task 3 taxonomy.** Define a defensible merge map over the confirmed near-synonym pairs (Applicable Laws/Governing Laws, Definitions/Defined Terms, Jurisdictions/Consent To Jurisdiction, Modifications/Amendments, Indemnifications/Indemnity, Withholdings/Tax Withholdings) and report macro-F1 both on the official 100 labels and on the merged set. This is the honest way to quantify the "irreducible without merging labels" claim in §4.2.3 instead of asserting it.
19. **Task 1 operating point.** Since the balanced training prior plausibly drives the 72-FP/13-FN profile, train one variant at the natural class ratio and one at 1:1, and report both — then state which operating point is being recommended for review assistance, rather than inheriting one by accident.

**Tier 4 — widen the comparison**

20. **Cheap baselines, for cost-effectiveness.** A TF-IDF + linear classifier and a fine-tuned BERT-base on T1 and T3. If an encoder matches an 8B QLoRA adapter at a fraction of the cost on these two tasks, that is an important negative result and belongs in the write-up.
21. **Alternative base models.** `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, reportedly stronger JSON adherence) and the legal-domain-pretrained `Equall/Saul-7B-Instruct-v1` (Colombo et al., 2024), both at the same adapter configuration. This tests directly whether a legal pretraining prior beats a stronger general model — the question the project's own model-selection review posed and never answered.
22. **Multi-task and retention evaluation.** Train one adapter on all three tasks jointly and compare against the three single-task adapters; and evaluate each adapter off-task and on a general instruction benchmark to measure what fine-tuning cost (Biderman et al., 2024).
23. **Position against the field's own benchmarks.** Score the T1 adapter against ContractEval, which benchmarks 4 proprietary and 15 open-source models on clause-level legal risk identification using CUAD (Liu et al., 2025), and consider the relevant LegalBench tasks (Guha et al., 2023). Without an external benchmark, "+0.415 macro-F1 over its own base model" says nothing about whether the result is good in absolute terms.
24. **Move toward end-to-end review.** Add a retrieval or segmentation stage so the system takes a whole contract rather than a pre-extracted clause, and report the compounded error. This is the step between a benchmark result and a tool.

---

## 6. References

*Note on very large author lists: papers with more than twenty authors are abbreviated here as the first authors followed by the collaboration name, in place of APA's nineteen-plus-ellipsis form, to avoid misattributing names.*

Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=aloEru2qCG

Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., & Aletras, N. (2022). LexGLUE: A benchmark dataset for legal language understanding in English. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 4310–4330). Association for Computational Linguistics. https://aclanthology.org/2022.acl-long.297/

Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). *Training deep nets with sublinear memory cost* (arXiv:1604.06174). arXiv. https://arxiv.org/abs/1604.06174

Colombo, P., Pires, T. P., Boudiaf, M., Culver, D., Melo, R., Corro, C., Martins, A. F. T., Esposito, F., Raposo, V. L., Morgado, S., & Desa, M. (2024). *SaulLM-7B: A pioneering large language model for law* (arXiv:2403.03883). arXiv. https://arxiv.org/abs/2403.03883

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. In *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2208.07339

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. In *Advances in Neural Information Processing Systems 36*. https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197

Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). *Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping* (arXiv:2002.06305). arXiv. https://arxiv.org/abs/2002.06305

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.

Grattafiori, A., Dubey, A., Jauhri, A., & the Llama Team at Meta AI. (2024). *The Llama 3 herd of models* (arXiv:2407.21783). arXiv. https://arxiv.org/abs/2407.21783

Guha, N., Nyarko, J., Ho, D. E., Ré, C., Chilton, A., & the LegalBench collaboration. (2023). LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2308.11462

Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. In *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/6ea9ab1baa0efb9e19094440c317e21b-Paper-round1.pdf

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*. https://openreview.net/forum?id=nZeVKeeFYf9

Lhoest, Q., Villanova del Moral, A., Jernite, Y., Thakur, A., von Platen, P., Patil, S., Chaumond, J., Drame, M., Plu, J., Tunstall, L., Davison, J., Šaško, M., Chhablani, G., Malik, B., Brandeis, S., Le Scao, T., Sanh, V., Xu, C., Patry, N., & Wolf, T. (2021). Datasets: A community library for natural language processing. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 175–184). https://aclanthology.org/2021.emnlp-demo.21/

Liu, S., Li, Z., Ma, R., Zhao, H., & Du, M. (2025). ContractEval: Benchmarking LLMs for clause-level legal risk identification in commercial contracts. In *Proceedings of the Natural Legal Language Processing Workshop 2025*. Association for Computational Linguistics. https://aclanthology.org/2025.nllp-1.19/

Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2022). *PEFT: State-of-the-art parameter-efficient fine-tuning methods* [Computer software]. Hugging Face. https://github.com/huggingface/peft

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., García, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed precision training. In *International Conference on Learning Representations*. https://openreview.net/forum?id=r1gs9JgRZ

NVIDIA Corporation. (2025). *CUDA C++ programming guide: Compute capabilities*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

PyTorch. (2022). *`torch.cuda.is_bf16_supported()` seem to not work properly* (Issue No. 75427) [GitHub issue]. https://github.com/pytorch/pytorch/issues/75427

PyTorch. (2024). *`torch.cuda.is_bf16_compatible()` output inconsistent with TorchInductor support* (Issue No. 118122) [GitHub issue]. https://github.com/pytorch/pytorch/issues/118122

Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing* (pp. 2383–2392). https://aclanthology.org/D16-1264/

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management, 45*(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). *Stanford Alpaca: An instruction-following LLaMA model* [Computer software]. Stanford Center for Research on Foundation Models. https://github.com/tatsu-lab/stanford_alpaca

Tuggener, D., von Däniken, P., Peetz, T., & Cieliebak, M. (2020). LEDGAR: A large-scale multi-label corpus for text classification of legal provisions in contracts. In *Proceedings of the Twelfth Language Resources and Evaluation Conference* (pp. 1235–1241). European Language Resources Association. https://aclanthology.org/2020.lrec-1.155/

von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., & Gallouédec, Q. (2020). *TRL: Transformer reinforcement learning* [Computer software]. Hugging Face. https://github.com/huggingface/trl

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirzi, H. (2023). Self-Instruct: Aligning language models with self-generated instructions. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 13484–13508). https://aclanthology.org/2023.acl-long.754/

Willard, B. T., & Louf, R. (2023). *Efficient guided generation for large language models* (arXiv:2307.09702). arXiv. https://arxiv.org/abs/2307.09702

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., … Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 38–45). https://aclanthology.org/2020.emnlp-demos.6/

---

### Appendix A — Provenance of every reported figure

| Figure | Source file |
| :--- | :--- |
| T1 fine-tuned metrics, confusion matrix | `kaggle_output_task1_fine_tuned/eval_metrics.json` |
| T1 baseline metrics, confusion matrix | `kaggle_output_task1_baseline/no_finetune_baseline/eval_metrics.json` |
| T1 training cost, hyperparameters | `kaggle_output_task1_fine_tuned/train_metrics.json` |
| T2 fine-tuned metrics (overall + per category) | `kaggle_output_task2_fine_tuned/eval_metrics.json` |
| T2 baseline metrics (overall + per category) | `kaggle_output_task2_baseline/no_finetune_baseline_task2/eval_metrics.json` |
| T2 training cost, hyperparameters | `kaggle_output_task2_fine_tuned/train_metrics.json` |
| T3 fine-tuned metrics, per label, confusions | `kaggle_output_task3_fine_tuned/eval_metrics.json` |
| T3 baseline metrics, per label, confusions | `kaggle_output_task3_baseline/eval_metrics.json` |
| T3 training cost, hyperparameters, time budget | `kaggle_output_task3_fine_tuned/train_metrics.json` |
| T3 prompt budget, trimmed-row counts, memory probe, recast savings, trainable-parameter count, pace | `kaggle_output_task3_fine_tuned/cuad-task3-finetune.log` |
| Fine-tuning parameters (§3.3.1 tables) | cell 27 of `llm_fine_tuning_LORA_task1_v2.ipynb` / `llm_fine_tuning_LORA_task2.ipynb`, cell 21 of `llm_fine_tuning_LORA_task3.ipynb`; `adapter_config.json` in each adapter directory; library defaults for settings not passed |
| Pinned library versions | first code cell of every training and baseline notebook |
| T1/T2 token-length counts (50/6,106 and 6/2,208 prompts ≥ 1,024 tokens; T2 max 652) | measured in this revision by re-tokenising each run's own JSONL with the cached `meta-llama/Meta-Llama-3.1-8B` tokenizer and the training prompt template |
| Tracked-vs-run JSONL comparison | record-level comparison of `cuad/` and `ledgar/` against the JSONL inside each run directory |
| Example counts, label balance, category counts | the `*.jsonl` files inside each run's output directory |
| CUAD table shape (510 × 83) | `data/CUAD_v1/master_clauses.csv`, `master_clauses_cleaned.csv` |
| Pre-processing logic (§3.2) | `scripts/preprocess_values_cuad.py`, `scripts/download_cuad.py`, `scripts/download_ledgar.py`; data cells of `llm_fine_tuning_LORA_task{1_v2,2,3}.ipynb` |
| Raw-vs-cleaned answer comparison, punctuation-survival rates, gold-vs-input overlap, value-type census, category-name mangling, split ratios | measured in this review by diffing `master_clauses.csv` against `master_clauses_cleaned.csv` and scanning all 3,085 task-2 and 8,314 task-1 generated examples |
| Derived figures (conditional EM, improved/regressed label counts, macro-F1 over scorable labels, invalid-prediction totals) | recomputed in this review from the files above |
| Smoke-test figures | `kaggle_output_3.2B_Llama_smoketest/eval_metrics.json` |

### Appendix B — Reported project artifacts not used in this paper

`kaggle_output/` is a shared download target that several `kaggle kernels output` pulls have written into. It holds the T1, T2 and 1B smoke-test adapters, the logs of three different runs, and a root-level `eval_metrics.json` / `train_metrics.json` that belong to whichever run was pulled last (currently T2 fine-tuned, byte-identical to `kaggle_output_task2_fine_tuned/`). For that reason no figure is read from it, and the comparison notebooks now point at the per-run directories. `kaggle_output/no_finetune_baseline/` duplicates the T1 baseline metrics found in `kaggle_output_task1_baseline/`; the two files are identical, and the task-1 directory is treated as canonical. `kaggle_output_3.2B_Llama_smoketest/` holds the `Llama-3.2-1B` pipeline smoke test (accuracy 0.897 on a 448-example subset) and is cited only as evidence that the pipeline ran end to end, never as a result.
