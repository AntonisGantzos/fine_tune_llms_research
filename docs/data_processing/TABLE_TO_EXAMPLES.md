# From a Table to Training Examples (Task 1)

How `build_examples` in [llm_fine_tuning_LORA_task1.ipynb](../llm_fine_tuning_LORA_task1.ipynb)
turns the `master_clauses` table into a list of separate yes/no questions — and why
that reshaping is what lets the model actually learn, instead of just handing it the
raw DataFrame.

---

## 1. What the raw table looks like

`master_clauses_cleaned.csv` is **one row per contract** and **two columns per
category** — a text column and its `_Answer` column. With 32 Task 1 categories
that is 64 columns side by side:

| Filename | NonCompete | NonCompete_Answer | Insurance | Insurance_Answer | … |
|---|---|---|---|---|---|
| contract_A | "Seller shall not compete within…" | Yes | "" | No | … |
| contract_B | "" | No | "The Venture may acquire insurance…" | Yes | … |

A model can't be trained directly on this shape. One row mixes 32 unrelated
questions together, the columns are wide and sparse, and "the thing to predict" is
smeared across 32 different `_Answer` cells. We need to break it into many small,
self-contained questions.

---

## 2. The reshape: one row → many examples

`build_examples` walks the table **one contract at a time** and, for **each of the
32 categories**, emits a single training example shaped like a question:

```json
{
  "instruction": "Is the following contract text a \"NonCompete\" clause? Answer strictly \"Yes\" or \"No\".",
  "category": "NonCompete",
  "input": "Seller shall not compete within the territory for three years…",
  "output": "Yes"
}
```

So **one contract with 32 categories becomes up to 32 separate examples.** Each
example is fully self-contained: it carries its own question (`instruction`), the
text to judge (`input`), and the answer to learn (`output`). The wide table is
"unpivoted" into a long list of (question, text, answer) triples.

```
            ┌─────────────── one contract row ───────────────┐
 TABLE      │ NonCompete | Insurance | Audit Rights | … (×32) │
            └────────────────────────┬───────────────────────┘
                                     │  build_examples()
                                     ▼
 EXAMPLES   { NonCompete?,   text, Yes }
            { Insurance?,    text, No  }
            { Audit Rights?, text, No  }
            …  (one example per category)
```

---

## 3. Where the `input` text comes from (the important part)

For each category the example's `input` is chosen as follows:

- **Answer is `Yes`** → use that category's **own real clause text**.
- **Answer is `No`** → **borrow a real clause from a *different* category in the
  same contract** (a *hard negative*). If the contract has no other clause to
  borrow, skip the example.

This is the whole point. Both the `Yes` and the `No` inputs are now **genuine legal
text**. A `No` example for `NonCompete` might contain a real Insurance clause — real
language that simply *isn't* a non-compete. The only way to answer correctly is to
recognize what each clause type actually looks like.

| Category asked | `input` (text shown) | `output` |
|---|---|---|
| NonCompete | "Seller shall not compete within the territory…" | `Yes` |
| NonCompete | "The Venture may acquire insurance on behalf of…" *(borrowed)* | `No` |

---

## 4. Why this helps the model learn better than the raw table

**1. The model is trained on the task it will actually do.**
At inference time you hand the model *one* clause and ask *one* yes/no question.
The examples match that exactly — one question, one passage, one answer — so there
is no gap between how it is trained and how it is used.

**2. It closes the data leak.** In the raw table the text column is non-empty
*exactly when* the answer is `Yes`. If you trained on that directly, the model would
learn a cheap shortcut: *"any text → Yes, empty/placeholder → No"*, never reading a
clause. By giving `No` examples **real** borrowed text, that shortcut disappears —
the model must learn the actual difference between, say, a non-compete and an
insurance clause.

**3. Each example is a clean, isolated signal.** One row of the table bundles 32
questions and one shared `Filename`. Splitting it into 32 separate examples means
the loss for "is this a non-compete?" is computed on just that one decision, with no
interference from the other 31 categories.

**4. It produces balanceable, countable data.** Once every row is an independent
`{question, text, answer}` example, we can count `Yes` vs `No`, downsample the
majority class on the train split, and measure per-class precision/recall/F1 — none
of which is possible while the labels are locked inside a wide DataFrame.

---

## 5. One safeguard: split first, then expand

Because one contract explodes into up to 32 examples, we must **split contracts into
train/validation _before_ calling `build_examples`**. If we split the examples
afterwards, clauses from the same contract could land in both sets and the model
could "memorize" a contract in training and be re-tested on it — inflated, dishonest
scores. Splitting at the contract level keeps every contract entirely on one side of
the train/val line.

---

## 6. In one sentence

> We unpivot the wide contract table into a long list of self-contained yes/no
> questions, give the `No` cases real (borrowed) clause text so the model can't cheat,
> and split by contract first — turning an untrainable spreadsheet into clean,
> leak-free, balanceable examples that teach the model to recognize clause types.

See also: [TASK1_HARD_NEGATIVES_PLAN.md](TASK1_HARD_NEGATIVES_PLAN.md) for the full
implementation plan.
