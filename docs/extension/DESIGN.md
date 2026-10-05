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
