# Task 3 — Evaluation Review

A short assessment of the metrics defined in `TASK3_EVAL_METRICS.md`.

**Overall rating: 3 / 5** — the right headline metric and honest handling of malformed output, but the numbers it produces can't yet support the conclusions drawn from them.

---

## 1. What this evaluation gets right

| # | Decision | Why it's correct |
|---|---|---|
| 1 | **Macro-F1 as the headline** | LEDGAR is 137× imbalanced. Macro weights every label equally, so weak performance on rare provision types can't hide behind *Governing Laws*. This is the standard choice in the LexGLUE literature. |
| 2 | **Unparseable output counted as wrong** | `__INVALID__` is never a free pass. Many generative-classification evals quietly drop unparseable rows, which inflates scores. This one doesn't. |
| 3 | **Valid-label rate reported separately** | Format compliance and task competence are different failures. Measuring them separately is the right instinct. |
| 4 | **Eval prompt is byte-identical to the training prompt** | No train/eval skew from prompt drift. Same `build_prompt()`, same 670-token trim. |
| 5 | **Deterministic decoding** | Greedy generation removes sampling noise as a confound. |
| 6 | **Per-label table + confusion pairs** | Averages hide *where* the model fails. Sorting worst-F1-first and listing gold→pred pairs is the correct diagnostic layer. |
| 7 | **Named look-alike label pairs** | Flagging Governing Laws ↔ Jurisdictions, Assigns ↔ Successors etc. shows awareness that some errors are label artifacts, not model failures. |
| 8 | **Rebalanced eval set** | Reasonable *for measuring capability* — it stops the metric being dominated by a handful of frequent labels. (It does break leaderboard comparability; see F2.) |

---

## 2. Failure modes of the current pipeline

Ordered by severity.

### F1 — Micro-F1 is a duplicate of accuracy
In single-label multi-class with one prediction per example, **micro-F1 = micro-precision = micro-recall = accuracy**. Metrics 2 and 4 are one number reported twice.

The only thing separating them is the `__INVALID__` sentinel: invalid predictions become false negatives but never false positives, so micro-F1 lands *fractionally above* accuracy. That gap is a parsing artifact and will be misread as signal.

> **Fix:** drop micro-F1, or relabel it. Report accuracy once.

### F2 — The leaderboard comparison doesn't hold
Published LEDGAR figures (~87–88 micro, ~82 macro) are computed on the **full, naturally imbalanced 10,000-example test set**. Yours is a stratified ~20-per-label validation subset.

F-scores are prevalence-dependent — change the class balance and the score changes even if the model didn't. On a near-uniform set, accuracy ≈ macro-recall, so the "micro-F1" here is really **balanced accuracy**. Comparing it to 87–88 is apples-to-oranges by construction.

> **Fix:** either stop claiming comparability, or run one extra eval on the untouched natural-distribution split.

### F3 — Reporting validation numbers against test-set benchmarks
The test set is held out entirely, so every number is a *validation* number. If validation was also used for checkpoint selection, early stopping, or the 670-token trim decision, these are biased and are not a generalization estimate.

> **Fix:** final numbers come from the held-out test set, run once.

### F4 — No baselines
A macro-F1 in isolation cannot tell you whether fine-tuning did anything. Missing:

- **Base model, no LoRA, same prompt** — isolates the effect of fine-tuning
- **TF-IDF + linear SVM** — scores 87.0 / 81.4 on LEDGAR, essentially matching BERT-base (87.6 / 81.8). The task is close to saturated by bag-of-words.
- **Legal-BERT / DeBERTa on your split** — 88.2 / 83.0 and 88.2 / 83.1
- **Majority class / random** — the floor

For scale: a 6B fine-tuned generative legal LLM (LexGPT) reaches 83.9 / 74.0 — still below a 110M Legal-BERT.

### F5 — No uncertainty quantification
At ~20 examples per class, per-label recall has a **95% CI of roughly ±17 points**. The bottom of the "worst-F1-first" table is substantially noise.

Aggregated: macro-F1 CI ≈ **±2 points**; accuracy CI ≈ **±1.6 points** on n=1,945. Two configurations within ~4 macro points are indistinguishable. Nothing in the pipeline reports this, and there are no multiple seeds and no significance tests.

### F6 — Macro-F1 is under-specified
Three ambiguities that change the number:

- **99 vs 100 labels.** *Books* is absent from validation. Averaging over 100 with `zero_division=0` silently costs up to ~0.8 macro points.
- **Which formula.** Mean-of-per-class-F1 vs. F1-of-macro-P-and-macro-R are different metrics and can rank models differently.
- **Invalid handling in the denominator.** Implied, not stated.

### F7 — Format and competence stay entangled
Valid-label rate is one scalar covering several distinct failures — empty output, near-miss (`Governing Law` vs `Governing Laws`), extra prose, hallucinated label, repetition. A near-miss is a normalization bug; a hallucination is a knowledge failure. They shouldn't share a number.

### F8 — No confidence signal
Greedy generation discards it entirely. So the pipeline cannot answer the actual product question: *which predictions should a lawyer check?* No calibration, no abstention, no top-k.

### F9 — Label noise is treated as ground truth
LEDGAR labels were scraped from formatting patterns in SEC filings — assigned by contract drafters, not annotators. The source corpus is 16.4% multi-label, and the original authors flag surviving synonym clusters (*withholding taxes* / *withholding of tax* / *tax withholding*).

Independent work confirms it: *Indemnity* is always predicted as *Indemnifications*; (*Taxes, Tax Withholdings, Withholdings*) collapses to one label; *Applicable Laws* scores 0–20% while *Governing Laws* hits 90%.

**There is an accuracy ceiling well below 100 and it is currently unmeasured.** The watchlist should also include Taxes ↔ Withholdings and Indemnity ↔ Indemnifications.

### F10 — Prompt robustness never tested
The 341-token label menu is a fixed design choice. Label ordering can move classification performance substantially. One ordering = one sample.

### F11 — Truncation impact unmeasured
Provisions are trimmed to 670 tokens. The % truncated and the accuracy gap between truncated and untruncated rows are not reported.

### F12 — No cost metrics
LEDGAR is a task where a 110M encoder is state of the art. Matching 83 macro at 50× the inference cost is a finding — and no accuracy metric will surface it.

---

## 3. Next steps

### Tier 1 — required before any result is quotable
1. **Delete micro-F1** (or rename it *balanced accuracy*).
2. **Move final reporting to the held-out test set**, run once.
3. **Add baselines on the same split**: base model zero-shot, TF-IDF+SVM, Legal-BERT, majority class.
4. **Add bootstrap 95% CIs** on all headline metrics + **Wilson intervals** per label.
5. **Run ≥3 seeds**, report mean ± std (LexGLUE itself uses 5).
6. **Pin down macro-F1**: average over the 99 present labels, state the formula, state invalid handling.

### Tier 2 — makes the evaluation genuinely informative
7. **Split valid-label rate into a failure taxonomy** — empty / near-miss / extra prose / hallucinated / repetition.
8. **Add constrained or rank-classification scoring** alongside free generation. Score all 100 labels by likelihood, or mask invalid tokens during decoding. **The gap between the two is your clean measure of format compliance**, and the constrained score upper-bounds what the model actually knows.
9. **Add a confidence score** (sequence log-prob or label margin) and report **ECE/Brier + a risk–coverage curve** — accuracy at 70/80/90% coverage. This is the deployment-relevant number for human-in-the-loop contract review.
10. **Add top-3 accuracy / MRR.** A shortlist of three is a usable product.
11. **Paired significance testing** vs. each baseline: McNemar's test on paired predictions + bootstrap ΔF1 CI.

### Tier 3 — research directions
12. **Establish the label-noise ceiling.** Manually audit ~100 errors; report what fraction are annotation artifacts rather than model errors.
13. **Report a coarse (merged-label) macro-F1** alongside the fine-grained one, collapsing known synonym clusters. Fine + coarse together is far more informative than either alone.
14. **Collapse the confusion table into one scalar**: *% of all errors falling into known synonym clusters.*
15. **Prompt-order robustness.** Rerun with 2–3 shuffled label orderings; report the spread. If the spread exceeds your CI, the result isn't stable.
16. **Truncation slice.** Report % truncated and the accuracy delta.
17. **Out-of-corpus slice.** Label semantics shift across contract types — *Consequences of Termination* means different things in an NDA vs. a construction agreement.
18. **Efficiency table.** Tokens/example, latency, $/1k provisions, next to macro-F1. Note that the 341-token label menu is a large fixed share of every prompt.
19. Optional: **multiclass MCC** as a chance-corrected sanity check next to macro-F1.

---

## References

- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL. https://aclanthology.org/2022.acl-long.297/ — leaderboard: https://github.com/coastalcph/lex-glue
- Tuggener et al. (2020). *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts.* LREC. https://aclanthology.org/2020.lrec-1.155/
- Jayakumar, Farooqui & Farooqui (2023). *Large Language Models are legal but they are not.* arXiv:2311.08890
- Chalkidis (2023). *ChatGPT may Pass the Bar Exam soon, but has a Long Way to Go for the LexGLUE benchmark.* arXiv:2304.12202
- Opitz & Burst (2019). *Macro F1 and Macro F1.* arXiv:1911.03347
- Brandl et al. (2022). *Bag-of-Words vs. Graph vs. Sequence in Text Classification*, App. C — micro-F1 ≡ accuracy proof. arXiv:2109.03777
- Brabec et al. (2020). *On Model Evaluation under Non-constant Class Imbalance.* arXiv:2001.05571
- Reinke et al. (2023). *Understanding metric-related pitfalls in image analysis validation.* arXiv:2302.01790
- Dror, Baumer, Shlomov & Reichart (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP.* ACL. https://aclanthology.org/P18-1128/
- Lu, Bartolo, Moore, Riedel & Stenetorp (2022). *Fantastically Ordered Prompts and Where to Find Them.* ACL. arXiv:2104.08786
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML
- Geifman & El-Yaniv (2017). *Selective Classification for Deep Neural Networks.* NeurIPS
