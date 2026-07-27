# Task 1 — Evaluation Review

A critique of the metrics defined in [TASK1_EVAL_METRICS.md](./TASK1_EVAL_METRICS.md).

**Overall rating: 3 / 5.** The data design is sound and the metrics chosen are the right *kind*. But as specified, the pipeline cannot separate "the model learned the task" from "the model learned the output format" — which is exactly the claim the research deliverable rests on.

---

## 1. What this evaluation gets right

### 1.1 Contract-level split (the most important decision)

Splitting 85/15 at the **contract** level, not the clause level, is correct. Contracts reuse boilerplate internally; a clause-level split would leak near-duplicate text across the boundary and inflate scores. The doc's own sanity check — be suspicious of near-perfect F1 — is the right instinct.

### 1.2 Hard negatives

Every `No` example is a real clause borrowed from a different category of the *same* contract. This removes the trivial "real legal text vs. filler" shortcut and forces discrimination between clause *types*. This is the strongest feature of the dataset construction.

### 1.3 Per-class metrics instead of pooled accuracy

Reporting precision, recall and F1 **separately for `Yes` and `No`**, with accuracy explicitly demoted, is correct for an asymmetric task. It matches how the CUAD authors frame the problem: they focus on precision/recall measures because of severe class imbalance, and note that recall matters more than precision since contract review is a needle-in-a-haystack task [1].

### 1.4 Support is reported

A small thing most write-ups skip. It is what lets a reader discount an F1 computed over a handful of examples.

### 1.5 Both machine- and human-readable artifacts

`eval_metrics.json` + `eval_report.txt` — one for downstream comparison scripts, one for eyeballing.

### 1.6 A stated baseline comparison

Framing the deliverable as a **delta** against the un-fine-tuned base model, rather than an absolute score, is the right research framing. The problem is how that delta is currently measured (§2.1).

---

## 2. Failure modes of the current pipeline

### 2.1 The answer parser biases the baseline comparison — **critical**

```python
"Yes" if "yes" in completion.lower() else "No"
```

This rule is not neutral. **Every unparseable completion silently becomes a `No`.** The base model "answers in a sentence" and `max_new_tokens=3` truncates it before it reaches a verdict — so the baseline is systematically pushed toward `No`: depressed recall, inflated specificity.

The measured delta is therefore a **mixture of legal-reasoning gain and format-compliance gain**, and the pipeline cannot decompose them. This is a documented artifact: generation-based scoring fails when a model embeds the right answer in conversational text that a rigid rule misses; log-likelihood ranking removes answer parsing entirely [2].

**Consequence:** the headline research number is not currently interpretable.

### 2.2 No format-validity metric for Task 1

The doc argues Task 1 needs no validity rate because the parse "never fails." That is backwards — the parse never fails *because it coerces silently*. Tasks 2 and 3 report validity; Task 1 hides the same information.

### 2.3 A single operating point, despite a stated recall priority

The doc declares recall on `Yes` matters most, then reports one greedy-argmax point with no threshold to tune. An asymmetric cost function is asserted and the only knob that acts on it is discarded.

CUAD's native metrics exist for this: thresholding confidence yields a PR curve, whose area (AUPR) summarises performance across thresholds, and fixing recall gives "Precision @ X% Recall" [1].

### 2.4 Aggregate-only metrics hide per-category collapse

2,208 validation examples over **32 categories ≈ 69 per category**. A pooled micro-F1 is dominated by easy, frequent categories (Governing Law, Parties, Agreement Date) and conceals failure on rare or subtle ones (Most Favored Nation, Third Party Beneficiary, Uncapped Liability). For a legal-review tool, per-category behaviour *is* the product.

### 2.5 No uncertainty quantification on a deliverable that is a difference

- At n = 2,208, a 95% CI on accuracy is ≈ **±1.5 pp**.
- Per category (n ≈ 69), it is ≈ **±8.5 pp** — wide enough to make most per-category comparisons meaningless without intervals.

Comparing two point estimates with no test is not a result. McNemar's test is the standard choice for two classifiers on the same test set because predictions are paired, and is recommended where models are too expensive to retrain repeatedly [3]. Bootstrap CIs should follow standard NLP protocol [4].

**Subtlety:** examples are **not i.i.d.** — many share a contract. A naive bootstrap over *examples* understates uncertainty; resample **contracts** (cluster bootstrap).

### 2.6 Precision will not survive deployment prevalence

Hard negatives create an artificially controlled class prior. **Recall and specificity are prevalence-invariant; precision and F1 are not.** Real contract review has low single-digit base rates for many categories.

```
Precision(pi) = TPR*pi / (TPR*pi + FPR*(1 - pi))
```

A detector scoring 0.82 precision at pi = 50% can collapse to ~0.02 precision at pi = 5% [5]. Metrics computed under artificial balance should not be read as operational estimates [6].

### 2.7 Single prompt, single seed

- **Prompt:** the fine-tuned model is evaluated on its own training template, which flatters it. Single-prompt evaluation is unreliable — absolute scores *and* model rankings shift across paraphrased instructions [7], and formatting alone has produced accuracy swings of tens of points on Llama-family models [8].
- **Seed:** one LoRA run at 764 steps. Multi-seed studies show identical configurations producing qualitatively different outcomes, so single-seed results underestimate true variance [9].

### 2.8 No non-LLM floor

The only comparison is base-LLM vs fine-tuned-LLM. Without TF-IDF + logistic regression and a majority-class baseline, there is no evidence the LoRA run earns its compute.

### 2.9 No calibration measurement

Greedy argmax discards confidence. This matters because SFT degrades calibration: pre-trained LMs are reasonably calibrated on next-token prediction, and instruction tuning / preference alignment systematically induce overconfidence [10, 11]. The fine-tuned model may be *more accurate* and *less trustworthy* at a review threshold.

### 2.10 Validation set doubles as the reporting set

There is no held-out test split. If any checkpoint or hyperparameter decision touched validation, the reported numbers are optimistically biased.

### 2.11 Two internal inconsistencies in the doc

| Claim A | Claim B | Problem |
|---|---|---|
| "Most clauses are absent (`No`) — always-`No` scores 80%+" | "A model ignoring the input should score ~50%" | Mutually exclusive. With 1:1 hard negatives the prior is ~50/50 and always-`No` scores 50%. **Print the actual class counts and derive the threshold from them.** |
| "Contract-level 85/15 split" | 6.1k train vs 2,208 val = **73/27** | Contract-level splitting causes some slack, not 12 points. Either training was truncated before a full epoch, or the split is not behaving as documented. Audit before publishing. |

---

## 3. Next steps

### P0 — do before reporting any numbers

| # | Action | Fixes |
|---|---|---|
| 1 | **Switch primary scoring to log-probabilities.** Compare `logP(" Yes")` vs `logP(" No")` at the first completion position, for *both* models. Standard rank-classification protocol [2]. | 2.1, 2.2, 2.3, 2.9 — and yields a continuous score for free |
| 2 | **Print actual `Yes`/`No` counts**; fix the two contradictions in §2.11; audit the split ratio. | 2.11 |
| 3 | **Per-category P/R/F1 + macro-average** across the 32 categories, next to the micro figure. The divergence between them is itself a finding. | 2.4 |
| 4 | **Add a format-validity rate for Task 1** and a three-way `Yes` / `No` / unparseable breakdown; report metrics with and without coercion. | 2.1, 2.2 |

> Item 1 is the highest-leverage change here. It removes format from the model comparison *and* unlocks AUPR, threshold tuning and calibration in one move.

### P1 — needed for a defensible claim

| # | Action | Fixes |
|---|---|---|
| 5 | **McNemar's test** on the fine-tune vs baseline delta [3], plus **cluster-bootstrap 95% CIs resampling contracts** [4]. | 2.5 |
| 6 | **AUPR, Precision@80%Recall, Precision@90%Recall** from the `P(Yes)` scores [1]. | 2.3 |
| 7 | **Add MCC** as a co-headline number. It scores high only when all four confusion-matrix cells are good, and unlike F1 it accounts for both class sizes [12]. | 2.6 |
| 8 | **Carve out a true test split**, or state explicitly that no selection touched validation. | 2.10 |

### P2 — strengthens the research contribution

| # | Action | Fixes |
|---|---|---|
| 9 | **Non-LLM floors:** TF-IDF + logistic regression, majority class, random. If bag-of-words lands within a few points, that is a legitimate finding. | 2.8 |
| 10 | **3–5 paraphrased prompt templates**; report mean ± spread for both models [7, 8]. | 2.7 |
| 11 | **2–3 training seeds**; report mean ± std [9]. | 2.7 |
| 12 | **Prevalence-projected precision** at realistic per-category base rates [5, 13]; report **ECE** [10]. | 2.6, 2.9 |
| 13 | **Error analysis:** manually audit ~50 FPs. CUAD categories overlap semantically (e.g. Cap on Liability vs Uncapped Liability), so some "hard negatives" may be genuine positives — label noise, not model error. | data quality |
| 14 | **Shortcut probe:** check whether clause length alone predicts the label. If a length-only classifier does well, the negatives are not as hard as assumed. | data quality |

### Reporting caveat to add to the doc

Metrics from this **binary reformulation are not comparable to the CUAD leaderboard**, which evaluates span extraction (reference: DeBERTa at 47.8% AUPR / 44.0% P@80%R / 17.8% P@90%R [1, 14]). Present them as internally-comparable numbers only.

---

## References

1. Hendrycks, D., Burns, C., Chen, A., Ball, S. (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS Datasets & Benchmarks. arXiv:2103.06268 — AUPR / Precision@X%Recall; recall prioritised over precision under class imbalance.
2. Gao, L. et al. *A Framework for Few-Shot Language Model Evaluation* (EleutherAI lm-evaluation-harness) — log-likelihood rank classification removes answer-parsing failure and decoding stochasticity.
3. Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.* Neural Computation 10(7) — McNemar's test for paired classifier comparison on a single test set.
4. Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing.* ACL 2018, pp. 1383–1392.
5. Precision-vs-prevalence stress testing — precision collapse at low base rates under a fixed operating point. arXiv:2601.18552.
6. Balanced-class evaluation overstates deployment-relevant performance; F1 under artificial balance is not an operational estimate. arXiv:2607.00553.
7. Mizrahi, M. et al. (2024). *State of What Art? A Call for Multi-Prompt LLM Evaluation.* TACL — scores and rankings shift substantially across paraphrased instructions.
8. Sclar, M., Choi, Y., Tsvetkov, Y., Suhr, A. (2024). *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design* (FormatSpread). ICLR — large accuracy swings from formatting alone.
9. *Analyzing the Effect of Noise in LLM Fine-tuning* — multi-seed stability analysis; identical configs produce divergent outcomes across seeds. arXiv:2604.12469.
10. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). *On Calibration of Modern Neural Networks.* ICML — ECE.
11. Xiao, J. et al. (2025). *Restoring Calibration for Aligned Large Language Models.* ICML; see also Zhu et al. (2023) — instruction tuning and preference alignment systematically degrade calibration.
12. Chicco, D., Jurman, G. (2020). *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation.* BMC Genomics 21:6.
13. Brabec, J. et al. (2020). *On Model Evaluation under Non-constant Class Imbalance.* arXiv:2001.05571 — Positive-Prevalence Precision (P³) curves.
14. HuggingFace `evaluate-metric/cuad` metric card — documents AUPR, P@80%R, P@90%R and the reported DeBERTa figures.
