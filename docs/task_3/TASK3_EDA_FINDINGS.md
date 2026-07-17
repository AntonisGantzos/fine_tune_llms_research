# Task 3 (LEDGAR) — EDA Findings & What They Mean for Fine-Tuning

This document explains, in plain language, what the data-exploration notebook
[`LEDGAR_dataset_exploration.ipynb`](../../LEDGAR_dataset_exploration.ipynb)
found, and — for each finding — **what we should do about it when we train the
model.** Every number here comes from actually running that notebook.

For the step-by-step build plan see [TASK3_PLAN.md](TASK3_PLAN.md).

---

## 30-second background

**The task:** show the model one paragraph from a contract (a "provision") and
have it answer with **one label** saying what kind of provision it is — e.g.
*Governing Laws*, *Notices*, *Terminations*. There are **100 possible labels**,
and exactly one is correct for each provision. This is simple multi-class
classification (pick 1 of 100).

**The data:** the LexGLUE version of LEDGAR — a clean, standard research dataset
already split into three piles for us:

| Split | Rows | What it's for |
| :--- | ---: | :--- |
| train | 60,000 | teach the model |
| validation | 10,000 | check progress while training |
| test | 10,000 | final unseen exam |

Each row is just `text` (the provision) + `label` (a number 0–99) + `label_name`
(the human-readable name). We answered four questions before spending any GPU
time.

---

## Finding 1 — The labels are very unbalanced

**What we measured:** how many training examples exist for each of the 100 labels.

**What we found:**

- The most common label, **Governing Laws, has 3,167 examples.**
- The rarest, **Books, has just 23.**
- That is a **137× gap** between the biggest and smallest class.
- Typical (median) label has ~426 examples; average is ~600.
- The 10 rarest labels (Books 23, Assigns 31, Qualifications 47, Anti-Corruption
  Laws 106, Sanctions 118, Powers 125, Venues 131, Costs 168, Consent To
  Jurisdiction 175, Approvals 178) are the ones to watch.

**Why this matters (the simple version):** imagine a class where 90% of students
are named "Governing Laws." A lazy model can score well just by guessing the
popular labels and never learning the rare ones — and plain **accuracy** would
make it *look* good while it quietly fails every rare provision type.

**What we do about it in training:**

1. **Headline metric = macro-F1, not accuracy.** Macro-F1 scores each of the 100
   labels separately and then averages them, so getting *Books* wrong hurts just
   as much as getting *Governing Laws* wrong. This is the honest number to
   optimize and report. (We also report micro-F1 because that's what the public
   LEDGAR leaderboard uses, so we can compare to other people's results.)
2. **Balance the training data by sampling.** Instead of training on all 60k rows
   (which would drown the model in Governing Laws examples), we take a
   **stratified sample** — roughly the same number of examples *per label* (e.g.
   ~60–100 each). That gives rare labels a fairer share of the model's attention.
3. **Accept a floor for the rarest labels.** A few labels have fewer examples than
   our per-label target (Books = 23, Assigns = 31), so they simply cap out at
   whatever they have. The model will always be weakest on these; that's expected,
   and macro-F1 will honestly reflect it.

---

## Finding 2 — The provisions are short

**What we measured:** how long each provision is, counted in **tokens** (the
word-pieces the model actually reads), using the real Llama-3.1 tokenizer.

**What we found:**

| | tokens |
| :--- | ---: |
| half of provisions are under (median) | 104 |
| 90% are under | 290 |
| 95% are under | 379 |
| 99% are under | 585 |
| the single longest | 1,749 |

Only **26 provisions** (out of 70,000) are longer than our 1,024-token training
window all by themselves.

**Why this matters:** the model can only read a fixed amount of text at once (our
limit is `max_length = 1024` tokens). Anything past that gets chopped off. If
provisions were routinely huge, we'd lose important text and the model would be
guessing from fragments.

**What we do about it in training:** basically **nothing special — we're fine.**
Provisions are short, so in almost every case the model sees the whole thing. The
handful of monster provisions are so rare they won't affect learning. We keep
`max_length = 1024` as-is.

---

## Finding 3 — The instruction is big, but the whole prompt still (mostly) fits

**What we measured:** the model doesn't just see the provision — it sees a full
**prompt** we build around it. Because the model has to *choose from 100 labels*,
we must list all 100 labels inside the instruction (otherwise it can't know its
options). We measured how big that makes the total prompt.

**What we found:**

- The instruction — with all 100 label names spelled out — is **331 tokens.**
  That's a fixed "tax" paid on *every single* example.
- The prompt scaffolding (`### Instruction / ### Input / ### Response`) adds ~9
  tokens.
- So each prompt is roughly **340 tokens of overhead + the provision itself.**
- Result: 95% of full prompts are under 719 tokens, 99% under 925 — comfortably
  inside 1,024.
- **But not everything fits:** the longest prompts reach ~2,089 tokens, and
  **327 prompts (0.47%) go over 1,024** and get their tail truncated.

**Why this matters:** two things. First, that 331-token label list is unavoidable
— the model genuinely needs to see its menu of choices, and this is also what
makes the base-model baseline a *fair* comparison (it gets the same menu).
Second, because the instruction already eats a third of the budget, long
provisions have less room before they hit the ceiling.

**What we do about it in training:**

1. **Keep the full 100-label instruction.** It's the point of the task, and it's
   only ~330 tokens — a price worth paying.
2. **Accept the 0.47% truncation.** Fewer than 1 in 200 prompts lose any text, and
   those are the longest, rambliest provisions where the label is usually still
   obvious from the start. Not worth engineering around.
3. **Reuse the exact same instruction + template everywhere** — training,
   validation, and the base-model baseline — so all comparisons are apples-to-apples.
4. **A caution for later:** if we ever shrink `max_length` to save memory, that
   fixed 330-token instruction means we'd start truncating provisions much sooner.
   Don't lower it without re-checking this.

---

## Finding 4 — The labels make sense (with a few tricky look-alikes)

**What we measured:** we printed real example provisions for the most common and
rarest labels, to make sure the data is clean and the labels are sensible.

**What we found:** the examples are clean, readable contract English and the
labels clearly fit the text. For instance a *Governing Laws* provision literally
says *"...shall be governed by... the law of the State of New York,"* and a
*Counterparts* provision talks about signing in separate counterparts. Good.

**The catch:** some labels are genuinely close cousins and will be easy for the
model to confuse:

- *Governing Laws* vs *Jurisdictions* vs *Submission To Jurisdiction*
- *Assigns* vs *Successors* vs *Assignments*
- *Amendments* vs *Modifications*
- *Waivers* vs *No Waivers*

**Why this matters:** even a good model will mix these up, and that's not really a
"bug" — the categories overlap in real legal writing. If we only looked at one
overall score we'd never see *which* pairs it confuses.

**What we do about it in training/eval:** in the evaluation step, alongside the
headline numbers, we produce a **"top confusions" list** — the label pairs the
model most often swaps. That tells us whether errors are silly (random) or
sensible (these known look-alikes), which is far more informative than a single
accuracy number.

---

## One open decision to make before training

The notebook surfaced a real wrinkle: **all 100 labels appear in the train and
test splits, but the validation split is missing one — `Books`** (it has 23
training examples and 0 validation examples).

The original plan wanted a sanity check that "every label appears in both the
train and validation sets." That **can't be literally true** here. Two clean
options:

| Option | What it means |
| :--- | :--- |
| **A. Relax the check (recommended)** | Require full label coverage in *train* only. Accept that `Books` (and any other ultra-rare label) simply won't be scored on validation. Simplest, and validation is only a progress gauge — the real exam is the test set, which does have all 100. |
| **B. Rebuild the validation split** | Carve a fresh validation set out of train so every label is present. More faithful to the plan, but throws away LexGLUE's official split and hurts comparability with published results. |

**Recommendation: Option A.** Keep the official LexGLUE splits (that's the whole
reason we can compare to published scores), and just soften the sanity check to
train-only coverage.

---

## The bottom line for the fine-tuning step

| Decision | Value | Because of |
| :--- | :--- | :--- |
| Headline metric | **macro-F1** (micro-F1 secondary) | Finding 1 (imbalance) |
| Training data | **stratified sample**, ~60–100 per label | Finding 1 (imbalance) |
| `max_length` | **keep 1024** | Findings 2 & 3 (short text, mostly fits) |
| Instruction | **keep full 100-label list** (~331 tokens) | Finding 3 (needed & fair) |
| Truncation | **accept 0.47% over-limit** | Finding 3 (negligible) |
| Extra eval output | **top-confusions list** | Finding 4 (look-alike labels) |
| Validation coverage | **relax to train-only** (Option A) | `Books` missing from val |

None of these are surprises for a classification task — which is the good news.
The dataset is clean, the text is short, and the only real judgment calls are how
to handle the class imbalance (sample + macro-F1) and the one missing validation
label. Next up is Step 3 of the plan: turn this into the training JSONL files.
