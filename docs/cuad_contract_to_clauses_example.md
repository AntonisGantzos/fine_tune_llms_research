# From Full Contract to Cleaned Master Clause Row

A walkthrough of how one raw CUAD contract becomes a single row in `master_clauses_cleaned.csv`.

**Example contract:** `MARKETING AFFILIATE AGREEMENT` between *Birch First Global Investments Inc* and *Mount Knowledge Holdings Inc* (Cybergy Holdings filing).

---

## Step 1 — The Raw Contract

The original is a long PDF/TXT document (thousands of words). A trimmed excerpt:

> **MARKETING AFFILIATE AGREEMENT**
> This agreement is made this **8th day of May, 2014**, between Birch First Global Investments Inc. ("Company") and Mount Knowledge Holdings Inc. ("Marketing Affiliate" or "MA")...
> This agreement shall begin upon the date of its execution by MA and acceptance in writing by Company, and shall remain in effect **until the end of the current calendar year**, and shall be automatically renewed for successive one (1) year periods...
> This Agreement is accepted by Company in the **State of Nevada** and shall be governed by and construed in accordance with the laws thereof...

**Why this is hard for a model:** the whole document is too long for a context window, and the useful facts (dates, parties, governing law) are buried in legal prose.

---

## Step 2 — Annotators Extract Clauses

Legal experts read the contract and, for each of the ~41 categories, pull out:

1. **The text span** (the sentence containing the answer) → the `[Category]` column.
2. **The normalized answer** (the clean fact) → the `[Category]-Answer` column.

If a category is not present, the answer is left empty / `"No"`.

---

## Step 3 — One Contract = One Row

All extracted clauses for this contract collapse into **a single row**. Each category becomes a **pair of columns**:

| Column (`[Category]`) — text span | Column (`[Category]-Answer`) — clean fact |
| :--- | :--- |
| `Document Name`: MARKETING AFFILIATE AGREEMENT | MARKETING AFFILIATE AGREEMENT |
| `Parties`: BIRCH FIRST GLOBAL INVESTMENTS INC ... MOUNT KNOWLEDGE HOLDINGS INC ... | Birch First Global Investments Inc; Mount Knowledge Holdings Inc |
| `Agreement Date`: 8th day of May 2014 | 5/8/14 |
| `Governing Law`: ...accepted by Company in the State of Nevada and shall be governed by... | Nevada |
| `Expiration Date`: ...shall remain in effect until the end of the current calendar year... | 12/31/14 |
| `Effective Date`: This agreement shall begin upon the date of its execution... | *(empty — no fixed date)* |

> Note: the *cleaned* file also strips punctuation, so `5/8/14` is stored as `5814` and `12/31/14` as `123114`.

---

## Step 4 — The Cleaning Pass

`master_clauses.csv` → `master_clauses_cleaned.csv` applies:

- Removes brackets/quotes/extra punctuation from values.
- Normalizes spacing and stray characters.
- Keeps the `[Category]` / `[Category]-Answer` pairing intact.

---

## Step 5 — Row Becomes Training Examples

Each category pair in the row turns into one instruction example:

```json
{
  "instruction": "Extract the Governing Law from the contract text. Return in JSON format.",
  "input": "{\"Governing Law\": \"This Agreement is accepted by Company in the State of Nevada and shall be governed by...\"}",
  "output": "{\"Governing Law\": \"Nevada\"}"
}
```

The `[Category]` text is the **input**; the `[Category]-Answer` is the **target output**.

---

## Summary

| Stage | What it is | Size |
| :--- | :--- | :--- |
| Full contract | Raw legal document (PDF/TXT) | Thousands of words |
| Annotation | Experts extract span + clean answer per category | ~41 categories |
| CSV row | One contract → one row of `[Category]` / `[Category]-Answer` pairs | 1 row |
| Cleaning | Strip punctuation/brackets, normalize | same row |
| Training data | Each pair → one instruction/input/output JSONL example | ~41 examples |

**Key idea:** a messy, page-long contract is compressed into one tidy row, then unfolded into many small input→output pairs the model can actually learn from.
