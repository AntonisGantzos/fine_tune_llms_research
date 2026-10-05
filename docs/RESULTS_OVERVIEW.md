# Fine-Tuning vs. Baseline — Cross-Task Results Report

High-level assessment of the model, the three task designs, and the measured
fine-tuned vs. baseline results. All numbers come from the run artifacts in
`kaggle_output*/` (`eval_metrics.json`, `train_metrics.json`), not from prose.

## 1. The Model & Training Setup

One model backs every task: **`meta-llama/Meta-Llama-3.1-8B`** (base, not Instruct),
adapted per task with **QLoRA** — 4-bit NF4 frozen weights plus a small trainable
LoRA adapter. `Llama-3.2-1B` served only as a same-family pipeline smoke-test.

| Setting | Value |
| :--- | :--- |
| Quantization | 4-bit NF4 (`BitsAndBytesConfig`) |
| LoRA | `r=16`, `alpha=16`, `dropout=0.05`, targets `q/k/v/o_proj` |
| Optimizer / LR | `paged_adamw_32bit`, 2e-4, weight decay 0.001 |
| Sequence length | 1024 (all three tasks) |
| Epochs | 1 (all three tasks) |
| Loss | completion-only, no packing |
| Hardware | Kaggle free **T4×2**, batch kernels driven from a CPU-only laptop |

Shared too: the `### Instruction / ### Input / ### Response:` template, greedy
decoding at eval, and a held-out **validation** split — fine-tuned and baseline runs
score the *same* examples with the *same* prompt, so the delta is the weights.


| Run | Steps | Final train loss | Runtime | Adapter |
| :--- | ---: | ---: | ---: | :--- |
| T1 | 764 | 0.094 | 8.6 h | `llama-3.1-8B-cuad-task1` |
| T2 | 307 | 0.243 | 2.8 h | `llama-3.1-8B-cuad-task2` |
| T3 | 1226 | 0.293 | 6.1 h (budget not hit) | `llama-3.1-8B-ledgar-task3` |

## 2. Task Logic

### T1 — Risk Clause Recognition (CUAD, binary)
**Question:** *"Is this text a `[Category]` clause? Answer strictly Yes or No."*
32 CUAD risk categories (the 41 column-pairs minus the 9 entity categories and
`Filename`, keeping T1/T2 disjoint). Input = the clause excerpt, output = one token.

The design decision that makes the task honest: **negatives are real, unrelated
clause text** (`scripts/preprocess_values_cuad.py`). Raw CUAD leaves the context
column empty/placeholder when a clause is absent, so a model could score high by
learning "short/empty input → No" without reading anything; hard negatives force the
answer to depend on content. Split is **contract-level 85/15** so no contract's
clauses appear in both sets. 6,106 train / 2,208 validation examples. **Metrics:**
macro-F1 headline (positives ~25% of validation, rare categories <10%), plus
accuracy, per-class P/R/F1 and a confusion matrix.

### T2 — Structured Entity Extraction (CUAD, JSON)
**Question:** *"Extract the `[Category]` … return a JSON object with the single key
`[Category]`; a list if multiple; `null` if absent."* 9 entity categories: Document
Name, Parties, Agreement/Effective/Expiration Date, Renewal Term, Notice Period To
Terminate Renewal, Governing Law, Warranty Duration. 2,454 train / 631 validation.

Two things are measured at once — *format discipline* and *value correctness* — so
the metric is three-layered and gated: `json_valid` (parses **and** the key set is
exactly the requested category) → `exact_match` on the normalized value → SQuAD-style
**token-F1** for partial credit, with greedy one-to-one matching for list values
(Parties). An invalid output scores 0 on the other two by construction, so
`exact_match ≤ token_f1 ≤ json_valid` always holds. Ground truth is CUAD's
expert-*normalized* value ("5/8/2014", not the sentence it came from) — a
normalization test, not a span-copy test.

### T3 — Provision Classification (LEDGAR, 100-way)
**Question:** pick exactly one of **100** provision-type labels. Data is LexGLUE's
LEDGAR subset (not raw LEDGAR's 12k noisy labels), sampled to **100 examples per
label** for train (9,801) and **20 per label** for validation (1,945) — a
deliberately balanced eval so macro-F1 isn't dominated by frequent provision types.

The full 100-label list must sit in every prompt (331 tokens) or the baseline can't
know its options and the comparison is meaningless. That drives the implementation:
the label is at the *tail* of the sequence, so the prompt is bounded by **trimming
the provision text**, never by truncating the assembled prompt. **Metrics:**
`valid_label_rate` (is the output even one of the 100 labels?), accuracy, macro-F1
(headline), micro-F1, and a top-confusions table.

## 3. Results — Fine-Tuned vs. Baseline

Baseline = the same `Meta-Llama-3.1-8B` with **no adapter**, same prompt and examples.

### T1 — 2,208 validation examples
| Metric | Fine-tuned | Baseline | Δ |
| :--- | ---: | ---: | ---: |
| Accuracy | **0.9615** | 0.5611 | +0.400 |
| Macro-F1 | **0.9507** | 0.5355 | +0.415 |
| "Yes" F1 (rare class) | **0.9277** | 0.4263 | +0.501 |
| "Yes" precision / recall | 0.883 / 0.977 | 0.318 / 0.645 | — |

Confusion matrix goes from `[[360,198],[771,879]]` to `[[545,13],[72,1578]]`. The
base model over-predicts "Yes" (771 false positives) — the classic "everything looks
like a risk clause" failure. Residual fine-tuned error is 72 false positives: the
model is now slightly *over*-sensitive rather than random.

### T2 — 631 validation examples
| Metric | Fine-tuned | Baseline | Δ |
| :--- | ---: | ---: | ---: |
| JSON valid | **0.9968** | 0.4723 | +0.525 |
| Exact match | **0.6910** | 0.0903 | +0.601 |
| Token-F1 | **0.8193** | 0.1408 | +0.679 |

The largest structural gain of the three: the base model produces a usable JSON
envelope less than half the time (0.00 valid for Agreement Date and Governing Law —
it answers in prose). Fine-tuning essentially solves the format problem; the
remaining gap is *value* quality. Per-category: Warranty Duration 1.00 EM, Document
Name 0.98, Notice Period 0.93, Governing Law 0.92 — but **Expiration Date 0.395** and
**Renewal Term 0.535** EM (date/term normalization is the weak spot), and **Parties
0.206 EM vs 0.886 F1**: the right parties are found, the full list rarely exact. Small
supports (Warranty Duration n=11, Notice Period n=29) make those figures noisy.

### T3 — 1,945 validation examples (100 labels, ~20 each)
| Metric | Fine-tuned | Baseline | Δ |
| :--- | ---: | ---: | ---: |
| Valid-label rate | **0.9964** | 0.4010 | +0.595 |
| Accuracy | **0.7686** | 0.0653 | +0.703 |
| Macro-F1 | **0.7509** | 0.0655 | +0.685 |
| Micro-F1 | **0.7700** | 0.0932 | +0.677 |

The baseline is near-useless here — 60% of the time it emits a string that isn't one
of the 100 labels, so most of its "errors" are format failures, not legal
misjudgments. Fine-tuned errors are overwhelmingly **label-taxonomy ambiguity**: the
top confusions are near-synonym pairs LEDGAR keeps separate — Applicable
Laws→Governing Laws (9), Withholdings→Tax Withholdings (9), Integration→Entire
Agreements (9), Definitions↔Defined Terms (8/7), Jurisdictions↔Consent To
Jurisdiction (8/6). A meaningful slice of the remaining 23% error is arguably
irreducible without merging labels.

## 4. Assessment & Caveats

- **Fine-tuning wins decisively on all three tasks**, and the gain is largest where
  output *format* is part of the task (T2 JSON, T3 closed label set). A sizeable
  share of the headline delta is the base model failing to answer in the required
  shape at all — worth stating explicitly rather than reading it all as legal skill.
- **The baselines are fair**: same base checkpoint, prompt, examples and greedy
  decoding, scored by the same code — that is what makes these deltas citable.
- **All numbers are validation-set, single-run, 1 epoch, seed 42.** No test-split
  evaluation and no seed variance yet — treat differences of a point or two, and the
  small-support T2 categories, as noise.
- **`README.md` and `CLAUDE.md` are stale**: both still call T2 and T3 "training run
  pending", but completed adapters and eval metrics exist for both.
- **Clearest next targets:** T2 date/term normalization (Expiration Date, Renewal
  Term) and list-valued Parties; for T3, accept the near-synonym confusions or
  collapse the taxonomy and re-report.
