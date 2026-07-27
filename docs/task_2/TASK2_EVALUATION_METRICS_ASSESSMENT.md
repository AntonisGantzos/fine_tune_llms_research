# Task 2 Evaluation — Assessment

**Overall rating: 3 / 5.** The metrics are a sound skeleton but not sufficient on their own to
accurately measure fine-tuned model performance. Sections 1–3 below list what works, what breaks,
and what to do next.

---

## 1. What this evaluation gets right

| # | Strength | Why it matters |
|---|---|---|
| 1 | **Gate-then-content design** (validity → exact match → F1) | Separates "didn't format" from "didn't extract". Matches how modern structured-output benchmarks score (parse check → schema check → zero the semantic scores on failure). |
| 2 | **Strict validity, no repair** | The stated deliverable is machine-consumable JSON. Stripping fences before parsing would report discipline the model doesn't have. |
| 3 | **Three metrics instead of one accuracy number** | A generative extraction task fails in ≥3 distinct ways; one scalar hides all of them. |
| 4 | **Deterministic decoding** (`do_sample=False`) | Fine-tuned vs. baseline deltas can't be sampling noise. Rerunning reproduces the numbers exactly. |
| 5 | **Prompt identical to training template, decode only post-prompt tokens** | No accidental prompt leakage into the score. (But see failure mode F3 — this also creates a confound.) |
| 6 | **Value-level comparison, not string comparison** | Gold is parsed to a real Python value, so `null`, scalars, and lists are compared as types, not text. |
| 7 | **Order-insensitive list matching with `max(len)` denominator** | Party order carries no meaning; padding the list with extra guesses costs score instead of gaming it. |
| 8 | **Zero partial credit for hallucination** | Predicting a value when gold is `null` is the worst failure for legal review. Correctly hard-zeroed. |
| 9 | **Per-category reporting with example counts** | Dates are easy, `Parties` is hard. A single aggregate would hide exactly the structure the write-up needs. |
| 10 | **`EM ≤ F1 ≤ valid` invariant** | A cheap, verifiable sanity check on any reported table. (Verified: holds under the stated normalization.) |
| 11 | **Micro vs. macro explicitly flagged** | Prevents the classic error of comparing overall numbers across runs with different category mixes. |
| 12 | **Machine-readable + human-readable artifacts** | `eval_metrics.json` + `eval_report.txt` makes the run auditable and re-analyzable without a rerun. |

---

## 2. Failure modes of the current pipeline

Ordered by severity.

### F1 — Null base rates are invisible (critical)

Nulls are averaged into the same means as real values. If a category is 80% gold-null, a model
that **always** predicts `null` scores EM 0.80 / F1 0.80 / validity 1.00 — and the report cannot
distinguish it from a good model.

- No null prevalence is reported anywhere.
- No trivial (majority-class) baseline exists to compare against.
- CUAD is a sparse "needle in a haystack" dataset, so this is not a corner case.

### F2 — No uncertainty quantification (critical)

- 631 examples ÷ 9 categories ≈ 70 per cell; several cells will be much smaller.
- No confidence intervals, no significance test on the fine-tuned − baseline delta.
- Worse: examples are **clustered by contract** (9 categories per contract), so they are not
  independent. Even a naive item-level bootstrap would understate the variance.
- Determinism ≠ precision. The number is repeatable but its error bar is unknown.

### F3 — The baseline comparison is confounded (critical)

The eval prompt is byte-identical to the training template. The fine-tuned model is therefore
evaluated in-distribution and the base model out-of-distribution.

- The JSON-validity gain measures **fine-tuning + prompt-format alignment**, not fine-tuning.
- Grammar-constrained decoding or 2–4 in-context examples close most of the format gap with zero
  training — neither is measured, so the "QLoRA buys format discipline" claim isn't identified.
- Format restrictions can also *depress* base-model content quality, so the base model's extraction
  ability may be understated by the JSON framing itself.

### F4 — Normalization bias favours the fine-tuned model

The defence "training data fixes one canonical surface form" holds for the fine-tuned model and
fails for the base model.

- `"November 1, 2013"` vs `"11/1/2013"` → EM 0 **and** F1 0, despite a perfect extraction.
- Affects 3 of 9 categories (all dates) plus the three duration categories.
- Net effect: EM and F1 deltas are biased **in the direction of the thesis**.

### F5 — All-or-nothing gate destroys diagnostic information

A perfectly correct answer wrapped in ```` ```json ```` fences and total gibberish both score
`0 / 0 / 0`. The error analysis cannot separate "cosmetic wrapper" from "no format understanding".

### F6 — "JSON validity" is key-set validity, not schema validity

`{"Parties": {"x": 1}}` passes the check. No type assertion on the value
(`str | list[str] | None`), so the metric doesn't fully match its name.

### F7 — Truncation is silently counted as a format failure

`max_new_tokens=128` with no EOS check. A non-terminating generation is indistinguishable from a
formatting error. Most likely to bite on `Parties`.

### F8 — Single gold reference

SQuAD and the official CUAD scorer both take the max over multiple acceptable references. Here
gold is one value, so correct answers are penalized wherever the cleaned CSV collapsed multiple
annotator spans.

### F9 — No confidence signal, so no operating point

Greedy decoding discards likelihood, so there is no precision/recall knob and nothing comparable
to CUAD's own AUPR / Precision@80% Recall. The deployment question — "what fraction can be
auto-accepted?" — cannot be answered.

### F10 — No forgetting check and no label-noise ceiling

- The adapter is only evaluated in-domain; general capability regression is assumed, not measured.
- Gold-label error rate is unmeasured, so an EM of 0.75 can't be placed against the annotation ceiling.

---

## 3. Next steps

### Tier 1 — cheap, fixes the critical holes

1. **Report null rate per category** in `per_category_counts` (one line of code).
2. **Add an always-null trivial baseline row** to the results table, next to base and fine-tuned.
3. **Stratify EM and F1** by `gold is null` vs `gold is not null`.
4. **Surface two named rates:** hallucination rate = P(pred ≠ null | gold = null);
   miss rate = P(pred = null | gold ≠ null).
5. **Contract-level paired bootstrap** (B ≈ 10,000) → 95% CIs on every metric and on every delta,
   plus a two-sided p-value on the delta.
6. **Report macro-average alongside micro-average.**
7. **Add a schema type check** to the validity metric.
8. **Log `truncation_rate` / EOS emitted** separately from format failures.
9. **Error taxonomy with counts:** fence / prose / wrong key / extra key / truncated / empty.

### Tier 2 — moderate cost, fixes the causal claim

10. **Baseline ladder** instead of a single baseline:
    - (a) base, zero-shot, same prompt *(current)*
    - (b) base, k-shot (2–4 in-context JSON examples)
    - (c) base, constrained decoding / JSON mode (Outlines, XGrammar, GBNF)
    - (d) base, lenient parser (strip fences, regex the first JSON object)

    Row (d) decomposes validity into "cosmetically wrapped" vs. "genuinely non-compliant".
    Keep the strict metric as the headline; report the lenient one as a diagnostic.
11. **Add a `normalized_match` metric** (fourth column): ISO-8601 date parsing, numeric+unit
    canonicalization for durations/notice periods, corporate-suffix stripping for `Parties`.
    Report it *beside* strict EM, never instead of it.
12. **Content quality among valid outputs:** restrict EM/F1 to examples where **both** models
    produced valid JSON, and report a paired (McNemar) table. This is a fairer read than the
    `EM / json_valid` ratio, which suffers from selection effects.

### Tier 3 — research extensions

13. **Confidence + risk–coverage.** Emit mean token log-probability per prediction, plot accuracy
    vs. coverage, report AURC and EM@80% coverage. This is the CUAD-native metric family and the
    metric a deployed reviewer actually needs.
14. **Semantic equivalence tier.** BEM or a constrained LLM-as-judge on a sampled subset, to bound
    how much token-F1 under-credits correct-but-differently-phrased answers. Document judge
    caveats (prompt sensitivity, self-preference bias).
15. **Forgetting check.** Small MMLU / instruction-following slice, base vs. adapter, to confirm
    the task gain didn't cost general capability.
16. **Label-noise audit.** Manually review ~50 gold values to estimate the annotation ceiling.
17. **Robustness sweep.** 3–5 paraphrases of the instruction line; report mean ± spread. Establishes
    whether the format gain is prompt-specific or genuinely learned.
18. **Scale to more of CUAD.** 9 of 41 categories were used; extending coverage tests whether the
    format/normalization finding generalizes beyond short, highly-templated fields.

---

## Summary

| | |
|---|---|
| **Question** | Do these metrics accurately measure fine-tuned model performance? |
| **Answer** | Partly. They measure the *right three things* but lack the controls needed to interpret the numbers. |
| **Rating now** | 3 / 5 |
| **Rating after Tier 1 + Tier 2** | ~4.5 / 5 |
| **Single biggest risk** | Null base rates (F1) — the aggregate can reward a degenerate always-null model. |
| **Single biggest scientific gap** | Confounded baseline (F3) — the headline claim is not identified by the current design. |
