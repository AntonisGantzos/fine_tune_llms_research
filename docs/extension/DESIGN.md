# Legal Model Extension — Design

Step-by-step plan: [LEGAL_MODEL_EXTENSION_PLAN.md](../LEGAL_MODEL_EXTENSION_PLAN.md).

## Question

Does legal-domain pretraining help contract review once the model has been task-adapted?
Saul-7B-Base is Mistral-7B continued-pretrained on legal text, so plain Mistral-7B is the
control that isolates the effect of that pretraining. Llama-3.1-8B is the existing reference.

## The 3×2 grid

|                | Zero-shot (`baseline`) | QLoRA (`finetune`)                          |
|----------------|------------------------|---------------------------------------------|
| Llama-3.1-8B   | re-run through harness | existing adapters re-scored (`adapter` mode) |
| Mistral-7B-v0.1| new                    | new                                         |
| Saul-7B-Base   | new                    | new                                         |

Each cell is run for each task (T1 risk clauses, T2 entity extraction, T3 LEDGAR labels).

**Current scope (2026-10-06): Saul only, Task 1 first.** The Mistral arms are deferred. Without
them, Saul results can *rank* Saul against Llama (zero-shot vs zero-shot, QLoRA vs QLoRA) but
cannot attribute a difference to legal pretraining, because Saul and Llama also differ in base
model, size, tokenizer and pretraining data. Zero-shot Saul is never compared against fine-tuned
Llama: that would mostly measure answer-format adherence, not legal knowledge.

**Headline comparison:** Saul-QLoRA vs Mistral-QLoRA (same architecture and tokenizer, so
any difference comes from the legal pretraining). Zero-shot arms show how much of the gap is
format adherence vs domain knowledge.

## What is held fixed across every arm

- Validation sets: the canonical JSONL below, byte for byte.
- Prompt template (`### Instruction / ### Input / ### Response:`), greedy decoding,
  `max_new_tokens` 3 / 128 / 16 (T1 / T2 / T3), max length 1024.
- Scoring: the same functions, extracted from the training notebooks into `scripts/task*_metrics.py`.
- Quantisation: 4-bit NF4, no double quantisation, fp16 compute (Kaggle T4 has no bf16).
- Training: 1 epoch, lr 2e-4, weight decay 0.001, `paged_adamw_32bit`, LoRA r=16 / α=16 /
  dropout 0.05 on q/k/v/o, completion-only loss, no packing.

Known asymmetry: the original Llama T1/T2 runs trained and evaluated in bf16; the harness
evaluates every arm, including the re-scored Llama adapters, in fp16 (see Phase 8).

## Models and licences

Checked 2026-10-05 against the Hugging Face API with the account behind the `hf-token` dataset
(config download only, no weights).

| Key | Repo | Licence | Gated | Architecture / vocab | Weights on disk |
|---|---|---|---|---|---|
| `llama` | `meta-llama/Meta-Llama-3.1-8B` | Llama 3.1 Community | yes (accepted) | Llama / 128,256 | bf16 |
| `mistral` | `mistralai/Mistral-7B-v0.1` | Apache-2.0 | no | Mistral / 32,000 | 14.5 GB bf16 safetensors |
| `saul` | `Equall/Saul-7B-Base` | MIT | no | Mistral / 32,000 | **29.0 GB fp32** safetensors |

All three load as 4-bit NF4 with fp16 compute, so the on-disk dtype only affects download time
and disk use. Keep the Hugging Face cache on the container disk (the default), never under
`/kaggle/working` (about 20 GB, and persisted as output). The exact repo commit is logged per run as
`model_revision` in `run_config.json`.

## Contamination caveat (applies to every Saul result)

SaulLM-7B (Colombo et al., 2024, arXiv:2403.03883, Table 1) lists **SEC EDGAR (about 5B tokens)**
among its legal pretraining sources, next to FreeLaw, the English MultiLegal Pile, EuroParl,
GovInfo, USPTO and others (about 94B raw tokens, deduplicated to about 30B). The paper reports no
decontamination against CUAD, LEDGAR or LexGLUE. Both CUAD contracts and LEDGAR provisions are
taken from EDGAR filings, so Saul may have seen the **raw contract text** of our validation
examples during pretraining. It cannot have seen the **labels** (CUAD annotations, LEDGAR
categories), which are not part of EDGAR. Any Saul advantage may partly reflect memorised
text rather than better legal reasoning. Llama-3.1's pretraining data is undisclosed web text and
may also include EDGAR filings, so this risk is not unique to Saul. It is documented for Saul
because there it is confirmed.

## Harness (`scripts/legal_model_extension.py`)

- Prompts and scoring come only from `scripts/task{1,2,3}_metrics.py`, extracted from the
  notebooks. `scripts/test_extension_scoring.py` checks them on CPU; for T1 it rebuilds
  per-example labels from both saved Kaggle confusion matrices and reproduces both
  `eval_metrics.json` files exactly (accuracy 0.9615 fine-tuned, 0.5611 baseline).
- **Sequence-length cap: input trimming in every task.** The T3 rule (trim the input text so
  prompt + completion fit `max_seq_len`; never truncate the assembled sequence) is applied to all
  tasks. The T1/T2 notebooks truncated the assembled sequence instead. Measured on CPU with the
  real tokenizers at `max_seq_len=1024` for T1:

  | Tokenizer | Input budget | Train trimmed | Val trimmed |
  |---|---:|---:|---:|
  | Llama-3.1 | 977 | 50 / 6,106 | 6 / 2,208 |
  | Mistral-7B = Saul-7B (same tokenizer) | 969 | 73 / 6,106 | 8 / 2,208 |

  So the published Llama T1 run trained on 50 examples whose label was truncated away, and
  scored 6 validation prompts that had lost `### Response:`. The harness's Llama T1 numbers can
  therefore differ from the notebook's on at most those 6 validation examples (≤ 0.27 pp).
  Prompts that need no trimming are byte-identical to the notebooks'.

  For T2 (same 1024 cap) trimming is negligible: Llama budget 663 tokens, 0 / 2,454 train and
  0 / 631 val trimmed; Mistral/Saul budget 587, 1 / 2,454 train and 0 / 631 val. The budgets are
  smaller because the longest gold JSON (a `Parties` list) is 283 / 347 tokens. One validation
  gold answer exceeds `max_new_tokens` = 128 for both tokenizers, so no arm can get it right;
  this ceiling is inherited from the notebooks.

  For T3 the 100-label instruction dominates: 331 tokens for Llama (341 with the template, as in
  the notebook log) and 406 for Mistral/Saul (424 with the template).

  | Tokenizer | Input budget | Train trimmed | Val trimmed |
  |---|---:|---:|---:|
  | Llama-3.1 | 670 | 79 / 9,801 | 9 / 1,945 |
  | Mistral-7B = Saul-7B | 584 | 210 / 9,801 | 32 / 1,945 |

  The Llama row matches the T3 notebook's log exactly. Saul trims about 3.5× more examples, but
  only 1.6 % of validation (≤ 1.6 pp of accuracy, in practice far less, because a provision's
  opening usually names its topic). `max_seq_len` therefore stays 1024 for every arm. The plan's
  1536 fallback was rejected: it changes the cap relative to the published Llama run, and longer
  sequences put Saul's T3 fine-tune at risk of the 7 h budget (Llama's T3 fine-tune used 6.1 h of it).
- Tokenizers: none of the three ships a pad token; all use `pad_token = eos_token`, exactly as the
  Llama notebooks did. Mistral and Saul share a tokenizer, so the headline comparison has no
  tokenization confound.
- Eval batch size follows the notebooks: 1 for T1/T2 (one prompt per `generate()`), 8 for T3.
- Mid-run checkpoints are disabled (`save_strategy="no"`); only the final adapter is saved. This
  does not change training.
- T1 adds a **strict-validity** diagnostic (first line exactly Yes/No). Accuracy keeps the
  notebooks' lenient rule ("yes" anywhere → Yes, else No) so published numbers reproduce.
- T2 keeps the notebooks' **strict** rule: the whole completion must `json.loads` to
  `{category: value}`, otherwise it is invalid (EM 0, F1 0). An invalid completion is stored as
  the `__INVALID__` sentinel, so it is never confused with a valid `null` (which is correct when
  the gold value is null). T2 adds a **lenient** diagnostic, `lenient_json_valid_rate`: the
  completion *starts with* a valid object, ignoring trailing text. The gap between the two rates
  measures "right JSON, didn't stop", which is the expected zero-shot failure for base models
  (Llama zero-shot JSON-valid 0.47). The raw completions in `predictions.jsonl` let lenient EM/F1
  be computed later if needed. McNemar uses exact match; the bootstrap uses mean value-F1.
  The T2 runs saved no per-example predictions, so the tests instead check that the repo
  validation set has the saved runs' n (631) and per-category counts, and that a perfect
  prediction scores 1.0 on every metric.
- T3 keeps the notebooks' rule: the first line of the completion, stripped, must equal a label
  (case-insensitive), otherwise `__INVALID__`. Metrics use sklearn with `labels=LABELS`, so an
  invalid answer is a miss and never a false positive. Macro-F1 averages over all 100 labels,
  including the one label missing from validation (99 / 100 covered), so the scorer needs
  `labels.json` (`--labels_file` on `run` and `compare`). T3 adds a **lenient** diagnostic,
  `lenient_valid_label_rate`: the first line *starts with* a label. McNemar uses exact-label
  accuracy; the bootstrap uses macro-F1. `max_new_tokens` is 16, as in the baseline notebook;
  the fine-tune notebook used longest label + 2 = 7 for Llama, but Saul's longest label is 8
  tokens. The parser reads only the first line, so the larger budget does not change a
  fine-tuned model's answers. The tests rebuild per-example pairs from the saved per-label
  precision, recall and support. Both saved Llama T3 runs reproduce exactly (accuracy, valid
  rate, macro- and micro-F1; per-label values to 1e-12, because the local sklearn computes F1 in
  a different but equivalent form).
- Timing reference: Llama T1 QLoRA took 30,995 s (8.6 h) in **bf16** on the T4 — over the 7 h
  budget. fp16 should be several times faster; the first Phase 9 run measures it.

## Run log

| Date | Run | Result |
|---|---|---|
| 2026-10-05 | `saul_t1_baseline_smoke` (`--limit 40`) | Pipeline OK end to end on a T4. Saul download 2.0 min (29 GB), load 1.8 min, eval 13.8 s for 40 examples (0.35 s/example → about 13 min for all 2,208). Answered "No" to all 40; 100% strictly valid. The 0.95 accuracy is meaningless: the first 40 rows contain only 2 positives (558 / 2,208 overall). Kaggle mounted the datasets at `/kaggle/input/datasets/antonisgantzos/<slug>/` — the runner resolves by file name, so this needed no change. Kaggle stack: `datasets==4.8.5`, torch 2.11.0+cu128, other pins as requested. |
| 2026-10-05 | `saul_t1_finetune_smoke` (`--train_limit 200 --limit 200`) | All pre-flight checks passed: worst-case batch (1,017 tokens) peaks at 6.0 GB, projected 6.1 / 15.6 GB; eval-path `generate()` OK. 25 steps in 338.6 s = **13.5 s/step in fp16** → full T1 epoch (764 steps) ≈ **2.9 h**, inside the 7 h budget (Llama bf16 needed 8.6 h). Loss 0.302. Eval on first 200 val rows (47 positives): accuracy 0.905, Yes-F1 0.82, 100% strictly valid, 0.82 s/example with the adapter → about 30 min for all 2,208. Adapter saved (54.6 MB, base `Equall/Saul-7B-Base`). Trimmed: 2/200 train, 1/200 val. Bug found: `epochs_completed` was `null` (fixed to read `trainer.state.epoch`). |
| 2026-10-06 | `saul_t1_baseline` (full, 2,208) | **Degenerate: "No" for every example** (no category ever gets a Yes). Accuracy 0.747 = the majority-class rate (1,650 / 2,208), Yes-F1 0.00, **macro-F1 0.43**. 100% strictly valid (raw: "No" 2,119, "no" 35, plus trailing newlines). Eval 882 s (0.40 s/example). 8/2,208 val inputs trimmed, as predicted on CPU. For comparison, Llama zero-shot (notebook, bf16): accuracy 0.561, macro-F1 0.535, Yes-recall 0.65. **Saul's higher accuracy is not a better model**: on this imbalanced set "always No" scores 0.747, so T1 zero-shot arms must be compared on macro-F1 (what `compare` bootstraps), not accuracy. |
| 2026-10-06 | `saul_t1_finetune` (full: 6,106 train, 2,208 val) | Pre-flight checks all OK (peak 6.0 GB, projected 6.1 / 15.6 GB). 764 steps, 1.0 epoch, **3.06 h** (about 14 s/step), not stopped on budget. Final loss 0.088. **Accuracy 0.968, macro-F1 0.959**, Yes P/R/F1 0.901 / 0.980 / 0.939, confusion `[[547, 11], [60, 1590]]`, 100% strictly valid. Eval 1,896 s (0.86 s/example). 73 train / 8 val inputs trimmed. |

### T1 summary so far (validation, n = 2,208)

| Arm | Accuracy | Macro-F1 | Yes-F1 | Errors | Source |
|---|---:|---:|---:|---:|---|
| Saul zero-shot | 0.747 | 0.428 | 0.000 | 558 | harness, fp16 |
| Llama zero-shot | 0.561 | 0.535 | 0.426 | 969 | original notebook, bf16 |
| Saul QLoRA | **0.968** | **0.959** | **0.939** | 71 | harness, fp16 |
| Llama QLoRA | 0.962 | 0.951 | 0.928 | 85 | original notebook, bf16 |

| Llama QLoRA, re-scored | 0.962 | 0.951 | 0.929 | 84 | harness, fp16 (`llama_t1_adapter_adapter1`) |

**Llama re-score (Phase 8, 2026-10-06).** The original adapter evaluated through the harness in fp16
gives accuracy 0.9620 vs 0.9615 in the notebook (bf16): one more correct example, confusion
`[[547, 11], [73, 1577]]` vs `[[545, 13], [72, 1578]]`. The bf16→fp16 switch and the input
trimming of 6 val prompts change essentially nothing. This run was also the first to load a saved
adapter from disk (`adapter` mode), and it works.

**Paired tests (`compare`, 2,000 bootstrap resamples):**

| A → B | Discordant (only A / only B correct) | McNemar p | Macro-F1 diff B−A, 95% CI |
|---|---|---:|---|
| Llama QLoRA → Saul QLoRA | 27 / 40 | 0.142 | +0.007 [−0.002, +0.016] |
| Saul zero-shot → Saul QLoRA | 60 / 547 | 2.4e-99 | +0.531 [+0.518, +0.543] |

**Conclusion for T1:** fine-tuning is what makes Saul useful (zero-shot it answers "No" to
everything). After fine-tuning, Saul and Llama are **statistically indistinguishable** on T1:
Saul is ahead by 13 examples, but p = 0.14 and the macro-F1 CI includes 0. This is not evidence that
legal pretraining helps here. The contamination caveat above applies to the Saul rows.

### T2 runs

| Date | Run | Result |
|---|---|---|
| 2026-10-06 | `saul_t2_baseline` (full, 631) | **JSON-valid 0.10, EM 0.00, F1 0.07**; lenient JSON-valid 0.24. Eval 2,125 s (3.4 s/example: most completions run to the 128-token cap). 0/631 val inputs trimmed. Failure modes in the raw output: 425/631 start with a bare JSON array (`["SPONSORSHIP AGREEMENT"]`, no object); 93 give a correct-looking object and then keep generating (`\n\n### Instruction: ...`), which the notebooks' strict rule counts as invalid; 52 open a Markdown code fence. Every valid object wraps the value in a list (`{"Document Name": ["X"]}`), which the notebook rule scores EM 0 but F1 1 against a scalar gold, hence Document Name F1 0.39 with EM 0.00. **Format is not the whole story**: a generous re-score for the record only (first object, unwrap one-item lists) gives EM 0.079 / F1 0.087, still below Llama zero-shot under the strict rule. |
| 2026-10-06 | `saul_t2_finetune` (full: 2,454 train, 631 val) | Pre-flight checks all OK (worst-case batch 757 tokens, peak 5.7 GB, projected 5.8 / 15.6 GB). 307 steps, 1.0 epoch, **0.99 h** (11.6 s/step), not stopped on budget. Final loss 0.185. **JSON-valid 0.995, EM 0.724, F1 0.853**; lenient JSON-valid also 0.995 (it always stops after the object now). The 3 invalid outputs are long `Parties` answers cut off by the 128-token cap before the JSON closed. Weakest categories are the same as Llama's: Parties EM 0.19 (F1 0.89), Expiration Date 0.40, Renewal Term 0.56. Eval 1,381 s (2.2 s/example). 1 train / 0 val inputs trimmed. |

### T2 summary so far (validation, n = 631)

| Arm | JSON-valid | EM | F1 | Source |
|---|---:|---:|---:|---|
| Saul zero-shot | 0.097 | 0.000 | 0.069 | harness, fp16 |
| Llama zero-shot | 0.472 | 0.090 | 0.141 | original notebook, bf16 |
| Saul QLoRA | 0.995 | **0.724** | **0.853** | harness, fp16 |
| Llama QLoRA | 0.997 | 0.691 | 0.819 | original notebook, bf16 |
| Llama QLoRA, re-scored | 0.997 | 0.691 | 0.818 | harness, fp16 (`llama_t2_adapter_adapter2`) |

Per category (EM), Saul QLoRA vs Llama QLoRA (notebook): Agreement Date 0.95 vs 0.84, Effective
Date 0.91 vs 0.79, Notice Period 0.97 vs 0.93, Governing Law 0.94 vs 0.92; the rest within
±0.03. The gain is concentrated in the date categories.

**Paired tests (`compare`, 2,000 bootstrap resamples; McNemar on exact match, CI on mean F1):**

| A → B | Discordant (only A / only B correct) | McNemar p | F1 diff B−A, 95% CI |
|---|---|---:|---|
| Saul zero-shot → Saul QLoRA | 0 / 457 | 5.4e-138 | +0.784 [+0.753, +0.814] |
| Llama QLoRA (re-scored) → Saul QLoRA | 19 / 40 | 0.0086 | +0.036 [+0.017, +0.056] |

**Llama re-score (Phase 8, 2026-10-06).** `llama_t2_adapter_adapter2`: JSON-valid 0.997, EM 0.6910
(identical to the notebook), F1 0.818 vs 0.819. 0/631 val inputs trimmed; eval 953 s. As for T1,
the bf16→fp16 switch changes essentially nothing.

**The Saul advantage is confined to the two date categories, and those labels are a preprocessing
artefact.** `scripts/preprocess_values_cuad.py` (`re.sub(r'[^a-zA-Z0-9\s]', '', text)`) strips
punctuation from every answer, so CUAD's `5/8/14` becomes the gold `"5814"`; 167 of 171
Agreement/Effective Date golds are such digit strings. The input still reads e.g. "Nov 02 2019"
while the gold is `"11219"`, so the model must learn a lossy, sometimes ambiguous
(`11219` = 11/2/19 or 1/12/19) date→digits rewrite. Mistral/Saul tokenize digits one by one
(`1 1 2 1 9`), Llama 3 in chunks of up to three (`112 19`), and Llama's date errors are exactly
digit insertions/drops (gold `11219` → Llama `112219`; `32002` → `3202`).

| Subset | n | Only Llama / only Saul correct | McNemar p | EM Llama / Saul |
|---|---:|---|---:|---|
| Agreement + Effective Date | 171 | 2 / 21 | 6.6e-05 | 0.819 / 0.930 |
| All other categories | 460 | 17 / 19 | 0.87 | 0.643 / 0.648 |

**Conclusion for T2:** fine-tuned Saul beats fine-tuned Llama overall (p = 0.009), but the whole
difference comes from reproducing punctuation-stripped date strings, which is most plausibly a
tokenizer effect on a label artefact, not legal knowledge. On the other seven categories the two
are indistinguishable, as on T1. The artefact affects the original Llama T2 results too
(the label format is the same in every arm, so the comparison is fair, but the date EM numbers
measure digit-string reproduction rather than date extraction). The contamination caveat
applies to the Saul rows.

## Canonical data (SHA-256)

Hashes are of the **LF-normalised** bytes: what git stores and what Kaggle generated.
On this Windows checkout `core.autocrlf=true` writes CRLF to the working tree, so hashing the
working copy directly gives different values. When staging Kaggle datasets, upload LF bytes
(the `kaggle_output_task*_fine_tuned/` copies are LF).

| File | Rows | SHA-256 |
|---|---:|---|
| `cuad/train/cuad_train.jsonl` (T1) | 6,106 | `5385dc55281636077f2b3ae0198487598a6b948a7c2efc3542720e9fd4ed06b1` |
| `cuad/validation/cuad_validation.jsonl` (T1) | 2,208 | `97294400c4fa6a2e6997803397aba0dbebc6260b93127f006c1f473f989efbd0` |
| `cuad/train/cuad_task2_train.jsonl` (T2) | 2,454 | `88f1092397015e47925ef35947689e4700471c59554e6d14bed1063c0b7390d2` |
| `cuad/validation/cuad_task2_validation.jsonl` (T2) | 631 | `0a9c203065c15c6352f56cae430313ec561201d3e939da97877a3a6aa079c775` |
| `ledgar/train/ledgar_task3_train.jsonl` (T3) | 9,801 | `9fc6b58fdb898fc697bd094c9badd99182544eac081f8303e9837a17679f1190` |
| `ledgar/validation/ledgar_task3_validation.jsonl` (T3) | 1,945 | `dbfb450a695dccb82a2eba871afb15bad05b360de898e4b4564482d6121af4d1` |

Each repo file is identical (after LF normalisation) to the copy written by the corresponding
Kaggle fine-tuning run in `kaggle_output_task{1,2,3}_fine_tuned/`. The T1 files were replaced
on this branch: the previous repo copies (1,282 / 448 rows) were stale.

Recompute (Git Bash): `tr -d '\r' < FILE | sha256sum`
