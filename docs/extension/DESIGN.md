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
- Tokenizers: none of the three ships a pad token; all use `pad_token = eos_token`, exactly as the
  Llama notebooks did. Mistral and Saul share a tokenizer, so the headline comparison has no
  tokenization confound.
- Eval batch size follows the notebooks: 1 for T1/T2 (one prompt per `generate()`), 8 for T3.
- Mid-run checkpoints are disabled (`save_strategy="no"`); only the final adapter is saved. This
  does not change training.
- T1 adds a **strict-validity** diagnostic (first line exactly Yes/No). Accuracy keeps the
  notebooks' lenient rule ("yes" anywhere → Yes, else No) so published numbers reproduce.
- Timing reference: Llama T1 QLoRA took 30,995 s (8.6 h) in **bf16** on the T4 — over the 7 h
  budget. fp16 should be several times faster; the first Phase 9 run measures it.

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
