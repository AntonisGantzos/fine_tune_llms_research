# Task 3 (LEDGAR) — Progress So Far & Next Steps

Snapshot of where Task 3 stands against [TASK3_PLAN.md](TASK3_PLAN.md), and the
ordered list of what remains until the task is complete (trained, evaluated,
compared against baseline, documented).

Status date: 2026-07-26.

---

## Where we are: plan steps 1–5 built, no successful Kaggle run yet

| Plan step | Status | Evidence |
| :--- | :--- | :--- |
| 1. Download script | ✅ Done | `scripts/download_ledgar.py`; `data/LEDGAR/` holds `labels.json` + `ledgar_train.csv` / `ledgar_validation.csv` / `ledgar_test.csv` |
| 2. EDA notebook | ✅ Done | `LEDGAR_dataset_exploration.ipynb`; findings written up in [TASK3_EDA_FINDINGS.md](TASK3_EDA_FINDINGS.md) (all four questions answered) |
| 3. Preprocess → JSONL | ✅ Done | Data cells 3.0–3.6 of `llm_fine_tuning_LORA_task3.ipynb`; 9,801 train + 1,945 validation rows, built on Kaggle from the staged CSVs (confirmed in the v11 log) |
| 3b. Stage data to Kaggle | ✅ Done | `kaggle/stage_data.ps1` stages LEDGAR flat; mounts at `/kaggle/input/ledgar-lexglue` |
| 4. Training notebook | ✅ **Ran to completion in v14** | 1,226 steps / 5.54 h / final loss 0.2931; adapter + `train_metrics.json` recovered to `kaggle_output_task3_fine_tuned/` (see run history below) |
| 5. Evaluation cells | ✅ Built, ❌ crashed on first execution (v14) | fp32-norm → fp16-`lm_head` dtype mismatch under `generate()`; fixed by an autocast wrapper + pre-flight check 10. No `eval_metrics.json` yet |
| 6. Baseline + comparison | ❌ Not done | No `llama_3.1_task_3_no_fine_tune.ipynb`, no `task3_finetune_vs_baseline_comparison.ipynb` |
| 7. Documentation | 🟡 Partial | `TASK3_PLAN.md` + `TASK3_EDA_FINDINGS.md` exist; `TASK3_LOGIC.md`, `TASK3_NOTEBOOK_CELL_GUIDE.md`, `TASK3_EVALUATION_METRICS.md` missing; README/CLAUDE.md not yet updated |

## Kaggle run history

| Version | Outcome | Root cause | Fix |
| :--- | :--- | :--- | :--- |
| ≤ v10 | Died immediately after `Starting training...` | NaN loss: `max_length=1024` truncated the **assembled sequence**, taking the tail label off the 70 longest prompts (the 100-label menu is 331 tokens of every prompt), so a `batch_size=1` micro-batch had every token masked under `completion_only_loss` | first raised to 2048 (→ v11); properly fixed in v13 by trimming the **input text** instead, so the label can never be cut |
| v11 | Cancelled at 43,200.4 s — "max allowed execution duration", exit 137. 750 MB output = mid-run checkpoints only, no adapter, no metrics | **Throughput, not a hang.** ~12.8 s/example (~102 s per optimizer step) ⇒ ~35 h for one epoch. Dominant factor: `bf16` on a **T4**, which is Turing (cc 7.5) and has *no bf16 tensor cores* — `torch.cuda.is_bf16_supported()` returns `True` anyway, so every matmul silently fell back to fp32 kernels (~8 vs ~65 TFLOPS). Compounded by `per_device_train_batch_size=1`, which pays bitsandbytes' NF4 dequantization cost per weight for a single row | fp16 everywhere (`fp16=True`, `bnb_4bit_compute_dtype=torch.float16`) + batch 2 × grad-accum 4 + `group_by_length=True`; batched eval generation; `TimeBudgetCallback` (pace report at steps 5/25/100, hard stop at 7 h so save + eval always run) |
| v12 | `OutOfMemoryError: Tried to allocate 1.81 GiB` inside `cross_entropy`, 239 s in (first training step) | Peak VRAM is dominated by the **logits tensor** — `tokens_in_batch × 128,256 vocab`, materialized in fp16 and *again in fp32* by the loss. `group_by_length` deliberately schedules the longest example in batch 0, which at `max_length=2048` meant 2 × 1,898 tokens = 1.81 GiB of fp32 logits on top of ~13.5 GiB already in use on a 14.6 GiB card. The fp16 fix from v11 **was** confirmed working in this log | `MAX_SEQ_LENGTH = 1024` enforced by trimming the provision text in `build_prompt()` (worst case 2 × 1,024 = 2,048 tokens), plus pre-flight check **9**: a real forward+backward on that worst-case batch reporting measured peak GB |

| v13 | Stopped by pre-flight check 9 before training: worst-case batch peaked at **14.0 / 15.6 GB** | peft's `prepare_model_for_kbit_training` (run inside `SFTTrainer` for 4-bit models) upcasts **every** fp16 parameter to fp32 — including `embed_tokens` and `lm_head` at 525M params each. 2.1 GB apiece for *frozen* weights that came from an fp16 checkpoint, plus ~1.05 GB for the fp16 copy autocast caches to run the lm_head matmul. The "4-bit 8B ≈ 5.6 GB" rule of thumb is wrong for this stack by ~3 GB | cell **8b** recasts those two modules back to fp16 (LayerNorms deliberately stay fp32), guarded by pre-flight check **2b**: no frozen fp32 tensor over 100M params. Check 9 also now adds the fp32 logits copy accelerate makes under fp16, which a direct-call probe never allocates |
| v14 | **Training succeeded** — all 1,226 steps, 5.54 h, final loss **0.2931**, adapter saved. Crashed 14 s later in the eval cell: `RuntimeError: expected scalar type Float but found Half` at `lm_head` inside `generate()` | Same fp32/fp16 split that check **2b** and cell **8b** created, seen from the *other* side. `prepare_model_for_kbit_training` leaves the RMSNorms in **fp32**, so the final norm emits fp32 `hidden_states`, while cell 8b put `lm_head` back in **fp16** → `F.linear` gets mismatched dtypes. Training never hit it because accelerate wraps the training forward in **fp16 autocast**, which casts both operands; `generate()` runs outside any autocast. Pre-flight checks 1–9 only exercised the training path, so 5.5 h of GPU time was spent before the eval path was tried for the first time | wrap the eval `generate()` in `torch.autocast("cuda", dtype=COMPUTE_DTYPE)` — identical to what training does, and free, whereas recasting `lm_head` to fp32 would cost 2.1 GB. New pre-flight check **10** runs the real eval call (2 prompts, 2 new tokens) *before* training, so an eval-path break costs seconds instead of hours |

Kaggle published the failed run's output, so v14's adapter was recoverable: `python -m kaggle kernels output antonisgantzos/cuad-task3-finetune` → saved to `kaggle_output_task3_fine_tuned/` (adapter + `train_metrics.json` + the generated JSONL). A re-run of the fixed notebook retrains from scratch; evaluating the recovered adapter directly would skip the 5.5 h.

The 1024 cap was confirmed working in v13's log: instruction + template = **341 tokens**, `MAX_INPUT_TOKENS = 670`, **79/9,801 train (0.81%)** and **9/1,945 validation (0.46%)** provisions trimmed, longest assembled sequence 1,017 tokens (mean 490).

Estimated but **unverified** post-fix cost: the next run's check-9 peak-memory line and the first `[pace]` line are the two pieces of real evidence — read them early instead of waiting hours.

Decisions locked in by the v11 / v12 post-mortems:

- **fp16, never bf16, on Kaggle's T4.** Pre-flight check 1b hard-fails bf16 on any
  GPU with compute capability < 8.0. (T1/T2 still use `bf16=True` — they ran the 1B
  smoke model where the penalty was affordable. Do not copy that setting into T3.)
- **Effective batch stays 8**, expressed as 2 × 4 rather than 1 × 8.
- **Sequences are capped at 1,024 tokens by trimming the provision text**, never by
  truncating the assembled prompt. `MAX_INPUT_TOKENS` is derived from the tokenizer
  (`MAX_SEQ_LENGTH` − instruction − scaffolding − longest label − slack), and the
  number of trimmed rows is printed for both splits. ~0.7% of train rows are affected;
  LexGLUE's own BERT baseline truncates LEDGAR at 512 tokens, so this is conservative.
- **One prompt builder.** `build_prompt()` is used by training and evaluation, and the
  baseline notebook must use it too — otherwise the comparison silently drifts.
- **`embed_tokens` and `lm_head` stay fp16** (cell 8b), LayerNorms stay fp32. Budget
  VRAM as ~3.6 GB of NF4 linears + ~2.1 GB of fp16 embed/lm_head + ~6 bytes per
  (token × 128,256 vocab) of logits traffic — not the "0.5 GB per billion params"
  rule of thumb, which is off by ~3 GB here.
- **A partial epoch is an acceptable result**, a killed kernel is not:
  `stopped_on_time_budget` is written into both metrics files so partial runs are
  never silently compared against full ones.

Decisions already locked in (don't re-litigate):

- **Stratified sample sizes:** ~100/label train (9,801 after rare-label caps),
  ~20/label validation (1,945). Within the plan's 6k–10k / 1k–2k targets.
- **`Books` validation gap → Option A** from the EDA doc: the sanity-check cell
  requires full 100-label coverage in *train only*
  (`require_full_coverage=False` for validation). Official LexGLUE splits kept.
- **Instruction:** full 100-label menu — **331 tokens** measured with the real
  Llama-3.1 tokenizer (EDA Finding 3), + ~9 tokens of `### Instruction/Input/Response`
  scaffolding, i.e. ~340 tokens of fixed overhead in every prompt.
  `max_length=1024` with **zero** truncation of the assembled sequence — the label
  sits at the tail, so the provision text is trimmed instead (see run history).
- The notebook's Kaggle config cell already expects the LEDGAR dataset to mount
  at **`/kaggle/input/ledgar-lexglue`** — the Kaggle dataset slug must match.
- `kaggle/kernel-metadata.json` already points `code_file` at the T3 notebook
  (kernel id `antonisgantzos/cuad-task3-finetune`), but `dataset_sources` still
  lists only the CUAD dataset — must be updated in step 1 below.

---

## Next steps, in order

### 1. Stage LEDGAR to Kaggle (blocker for everything remote)

The notebook builds the JSONL *on Kaggle* from the CSVs (cell 3.1 reads
`labels.json` + `ledgar_train.csv` + `ledgar_validation.csv` from `DATA_DIR`),
so stage the **raw split files**, not the generated JSONL:

1. Create a new Kaggle dataset whose mounted folder resolves to
   `/kaggle/input/ledgar-lexglue` (i.e. dataset slug `ledgar-lexglue`, owner
   `antonisgantzos`). Mirror the CUAD flow: a small staging script (extend
   `kaggle/stage_data.ps1` or add `kaggle/stage_ledgar.ps1`) plus a
   `dataset-metadata.json`, then `python -m kaggle datasets create` /
   `datasets version`. Payload: `labels.json`, `ledgar_train.csv`,
   `ledgar_validation.csv` (~test CSV optional, unused for now).
2. Watch the known `--dir-mode zip` flattening quirk — either stage the files
   flat (no subfolder) so `DATA_DIR / "labels.json"` resolves directly, or add
   the same fallback-search the CUAD notebooks use.
3. Add the new slug to `dataset_sources` in `kaggle/kernel-metadata.json`
   (alongside or replacing the CUAD source for this kernel).

### 2. Finish the training notebook (plan step 4)

Port the training half of `llm_fine_tuning_LORA_task2.ipynb` (cells 24–30
there) into `llm_fine_tuning_LORA_task3.ipynb`, changing as little as possible:

- The GPU-pin cell (`CUDA_VISIBLE_DEVICES` before any torch import) and UTF-8
  check cell from the top of the T2 notebook — T3 currently lacks both.
- HF-token diagnostic cell (Kaggle secret `HF_TOKEN`, gated-repo access check).
- Model + tokenizer load with the shared QLoRA config: 4-bit NF4, r=16,
  alpha=16, targets `q/k/v/o_proj`, `paged_adamw_32bit`,
  `completion_only_loss=True`, same `### Instruction / ### Input / ### Response`
  template. Keep the `meta-llama/Llama-3.2-1B` smoke-test switch.
- **Pre-flight checks cell**, with the T2 "completion parses as JSON" assert
  swapped for "completion is exactly one of the 100 labels".
- Train + save adapter to `WORK_DIR / "llama-3.1-8B-ledgar-task3"`.

### 3. Add the evaluation cells (plan step 5)

Same notebook, after training — generate on the 1,945-row validation JSONL and
compute, in this order:

1. **Valid-label rate** — exact match against the 100-label list after
   strip/lowercase (T3's analogue of T2's JSON-validity).
2. **Accuracy**.
3. **Macro-F1** — the headline number (imbalance, per EDA Finding 1).
4. **Micro-F1** — for LexGLUE-leaderboard comparison (published BERT-class
   yardstick: ~87–88 micro / ~82 macro).
5. **Per-label precision/recall/F1** + **top ~15 confused label pairs** (watch
   the known look-alikes: Governing Laws/Jurisdictions, Assigns/Successors,
   Amendments/Modifications, Waivers/No Waivers).

Write `eval_metrics.json` + `eval_report.txt` to `WORK_DIR` so they come back
as Kaggle artifacts, same filenames as T1/T2.

### 4. Re-run on Kaggle (steps 1–3 are built; this is the open blocker)

Two runs have been attempted; both died for the reasons in the run-history table
above. For the next attempt:

1. Save the notebook file, then `.\kaggle\run.ps1 -Wait`.
2. **Check the log within ~10 minutes of `Starting training...`.** The
   `[pace] step 5: Xs/step -> full epoch ~= Y h` line is the go/no-go signal: if
   `Y` is anywhere near 12, cancel and cut the work (lower `TRAIN_PER_LABEL`, or
   drop to `meta-llama/Llama-3.2-1B` for a smoke proof) rather than burning
   another GPU day.
3. Outputs land in `kaggle_output/` — keep them (adapter, metrics, report), e.g.
   in a `kaggle_output_task3_fine_tuned/` copy like the T1/T2 runs. Check
   `stopped_on_time_budget` in `train_metrics.json` before quoting any number as
   a one-epoch result.

### 5. Baseline run (plan step 6a)

- Create `llama_3.1_task_3_no_fine_tune.ipynb` by mirroring
  `llama_3.1_task_2_no_fine_tune.ipynb`: same validation JSONL, same
  100-label instruction (fairness requires the base model to see the menu),
  base model with no adapter, same metrics, writes to a
  `no_finetune_baseline/`-style directory.
- Switch `code_file` in `kaggle/kernel-metadata.json` to it and run on Kaggle;
  keep the outputs (e.g. `kaggle_output_task3_baseline/`).

### 6. Comparison notebook (plan step 6b)

`task3_finetune_vs_baseline_comparison.ipynb`, mirroring
`task2_finetune_vs_baseline_comparison.ipynb`: side-by-side valid-label rate,
accuracy, macro/micro-F1, per-label deltas, top confusions for both models.

### 7. Documentation + housekeeping (plan step 7)

- Write `docs/task_3/TASK3_LOGIC.md`, `TASK3_NOTEBOOK_CELL_GUIDE.md` (once the
  notebook is final), and `TASK3_EVALUATION_METRICS.md`, mirroring
  `docs/task_2/`.
- Update the README status table and CLAUDE.md (T3 → implemented; add the
  LEDGAR data-layout, JSONL paths, adapter name, and staging script).
- Commit the currently untracked/modified T3 work: `ledgar/` JSONL,
  `llm_fine_tuning_LORA_task3.ipynb`, `kaggle/kernel-metadata.json`, and the
  new docs.

---

## Suggested immediate next action

Step 1 (Kaggle staging) and step 2 (training cells) are independent — the
notebook can be completed locally while the dataset uploads. Everything after
that is strictly sequential: smoke test → production run → baseline →
comparison → docs.
