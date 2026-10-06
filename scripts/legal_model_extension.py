"""Legal-model extension harness: run any {Llama, Mistral, Saul} x {zero-shot, QLoRA} arm on a
CUAD/LEDGAR task with the exact prompts, decoding and scoring of the Llama notebooks, and
compare two arms with paired tests.

Design and rationale: docs/extension/DESIGN.md. Plan: docs/LEGAL_MODEL_EXTENSION_PLAN.md.

    # Kaggle GPU (via legal_model_extension_runner.ipynb)
    python legal_model_extension.py run --task 1 --model saul --mode finetune \
        --train_file .../cuad_train.jsonl --val_file .../cuad_validation.jsonl \
        --out_dir /kaggle/working/runs/saul_t1_finetune [--limit 40] [--train_limit 200]
    python legal_model_extension.py run --task 1 --model llama --mode adapter \
        --adapter_dir .../llama-3.1-8B-cuad-task1 --val_file ... --out_dir ...

    # local CPU
    python legal_model_extension.py compare --task 1 --a RUN_DIR_A --b RUN_DIR_B

Modes: `baseline` = zero-shot base model; `finetune` = QLoRA train then evaluate the new
adapter; `adapter` = evaluate an existing adapter (re-scores the original Llama runs).

Outputs per run (in --out_dir): run_config.json, eval_metrics.json, eval_report.txt,
predictions.jsonl (one line per validation example, needed by `compare`), and for
`finetune` also train_metrics.json and the adapter in adapter/.

Every arm runs in fp16 with the T3 notebook's safeguards (see CLAUDE.md, "T3 deviates
deliberately"): fp16 compute because the T4 has no bf16 tensor cores; embed_tokens/lm_head
recast to fp16 after trainer init; generate() under fp16 autocast; a 7 h training budget;
pre-flight checks 9 (worst-case-batch memory probe) and 10 (eval-path generate).

Torch / transformers are imported inside `run` only, so `compare` and the scoring tests
work on a CPU-only machine.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import task1_metrics

MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "saul": "Equall/Saul-7B-Base",
}

# Per-task settings copied from the Llama notebooks. T2/T3 are registered when their
# scorers are extracted (plan Phase 1); until then only Task 1 can run.
TASKS = {
    1: {"metrics": task1_metrics, "batch_size": 1, "grad_accum": 8,
        "group_by_length": False, "eval_batch_size": 1},
}

SEED = 42
TRAIN_TIME_BUDGET_S = 7 * 3600   # of Kaggle's 12 h wall; leaves room for save + eval
TRIM_SLACK_TOKENS = 8            # EOS appended by TRL + margin, as in the T3 notebook


# ----------------------------------------------------------------------------- helpers

def read_jsonl(path, limit=None):
    """Read a JSONL file into a list of dicts; `limit` keeps only the first N rows."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def lf_sha256(path):
    """SHA-256 of the file with CR bytes removed — the convention in docs/extension/DESIGN.md,
    so the hash matches whether the file was staged from a CRLF or an LF checkout."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r", b"")).hexdigest()


def make_trimmer(tokenizer, build_prompt, max_seq_len, max_completion_tokens):
    """Return trim(instruction, text) -> (text_that_fits, was_trimmed).

    Ported from the T3 notebook's truncate_input: the sequence-length cap is enforced by
    trimming the INPUT text, never by truncating the assembled prompt, because the
    completion sits at the tail and sequence truncation would delete it (all-masked
    micro-batch -> NaN loss). Budget = max_seq_len - prompt scaffolding - longest completion
    - slack, measured with the run's own tokenizer (Mistral/Saul tokenize differently from
    Llama). When nothing is trimmed, prompts are byte-identical to the notebooks'.
    """
    overhead_cache = {}

    def trim(instruction, text):
        if instruction not in overhead_cache:
            overhead_cache[instruction] = len(tokenizer(build_prompt(instruction, ""))["input_ids"])
        budget = max_seq_len - overhead_cache[instruction] - max_completion_tokens - TRIM_SLACK_TOKENS
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return text, False
        return tokenizer.decode(ids[:budget]), True

    trim.min_budget = lambda: max_seq_len - max(overhead_cache.values()) \
        - max_completion_tokens - TRIM_SLACK_TOKENS
    return trim


# ----------------------------------------------------------------------------- run

def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    task = TASKS[args.task]
    metrics_mod = task["metrics"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transformers.set_seed(SEED)
    random.seed(SEED)

    if args.mode == "finetune" and not args.train_file:
        raise SystemExit("--mode finetune needs --train_file")
    if args.mode == "adapter" and not args.adapter_dir:
        raise SystemExit("--mode adapter needs --adapter_dir")

    hf_token = os.environ.get("HF_TOKEN")   # set by the runner notebook; never printed
    model_id = MODELS[args.model]
    compute_dtype = torch.float16

    # Same as every notebook: CUDA, and fp16 on pre-Ampere (the T4 is cc 7.5).
    assert torch.cuda.is_available(), "CUDA not available — this harness needs a GPU."
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB, cc {torch.cuda.get_device_capability(0)})", flush=True)

    # ---- tokenizer + data
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    pad_note = f"kept existing pad_token {tokenizer.pad_token!r}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_note = f"tokenizer had no pad_token; set pad_token = eos_token ({tokenizer.eos_token!r})"
    tokenizer.padding_side = "right"
    print(pad_note, flush=True)

    val_rows = read_jsonl(args.val_file, args.limit)
    train_rows = read_jsonl(args.train_file, args.train_limit) if args.mode == "finetune" else []
    completions = [r["output"] for r in train_rows + val_rows]
    max_completion_tokens = max(len(tokenizer(c, add_special_tokens=False)["input_ids"])
                                for c in completions)
    trim = make_trimmer(tokenizer, metrics_mod.build_prompt, args.max_seq_len, max_completion_tokens)

    def prompts_for(rows):
        prompts, n_trimmed = [], 0
        for r in rows:
            text, trimmed = trim(r["instruction"], r["input"])
            n_trimmed += trimmed
            prompts.append(metrics_mod.build_prompt(r["instruction"], text))
        return prompts, n_trimmed

    train_prompts, train_trimmed = prompts_for(train_rows)
    val_prompts, val_trimmed = prompts_for(val_rows)
    budget = trim.min_budget()
    print(f"Input budget (tightest instruction): {budget} tokens | trimmed: "
          f"train {train_trimmed}/{len(train_rows)}, val {val_trimmed}/{len(val_rows)}", flush=True)

    run_config = {
        "task": args.task, "model": args.model, "model_id": model_id, "mode": args.mode,
        "adapter_dir": args.adapter_dir, "seed": SEED,
        "max_seq_len": args.max_seq_len, "max_new_tokens": metrics_mod.MAX_NEW_TOKENS,
        "limit": args.limit, "train_limit": args.train_limit,
        "train_file": args.train_file, "val_file": args.val_file,
        "train_file_sha256_lf": lf_sha256(args.train_file) if args.train_file else None,
        "val_file_sha256_lf": lf_sha256(args.val_file),
        "n_train": len(train_rows), "n_val": len(val_rows),
        "input_token_budget_min": budget, "max_completion_tokens": max_completion_tokens,
        "trimmed_train": train_trimmed, "trimmed_val": val_trimmed,
        "pad_token": pad_note, "gpu": gpu_name,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }

    # ---- model: 4-bit NF4, no double quantisation, fp16 compute (= the notebooks)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, torch_dtype=compute_dtype,
        device_map={"": 0}, token=hf_token)
    run_config["model_revision"] = getattr(model.config, "_commit_hash", None)

    train_metrics = None
    if args.mode == "finetune":
        model, train_metrics = finetune(model, tokenizer, train_prompts, train_rows, val_prompts,
                                        task, metrics_mod, args, out_dir, run_config, vram_gb)
    elif args.mode == "adapter":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    import accelerate, bitsandbytes, datasets, peft, trl
    run_config["versions"].update(peft=peft.__version__, trl=trl.__version__,
                                  bitsandbytes=bitsandbytes.__version__,
                                  accelerate=accelerate.__version__,
                                  datasets=datasets.__version__)
    write_json(out_dir / "run_config.json", run_config)

    evaluate(model, tokenizer, val_rows, val_prompts, task, metrics_mod, args, out_dir,
             run_config, train_metrics)


def generate(model, tokenizer, prompts, max_new_tokens, max_seq_len):
    """Greedy decode, notebook settings. autocast is required: after QLoRA prep the RMSNorms
    are fp32 while lm_head is fp16 (run v14 crashed here after 5.5 h of training)."""
    import torch
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_seq_len).to(model.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def finetune(model, tokenizer, train_prompts, train_rows, val_prompts, task, metrics_mod,
             args, out_dir, run_config, vram_gb):
    """QLoRA fine-tune with the notebooks' hyperparameters; returns (peft_model, train_metrics)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model.config.use_cache = False
    dataset = Dataset.from_list([{"prompt": p, "completion": r["output"]}
                                 for p, r in zip(train_prompts, train_rows)])

    peft_config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=task["batch_size"],
        gradient_accumulation_steps=task["grad_accum"],
        group_by_length=task["group_by_length"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        logging_steps=25,
        save_strategy="no",   # mid-run checkpoints never get used; the final adapter is saved below
        optim="paged_adamw_32bit",
        max_length=args.max_seq_len,
        completion_only_loss=True,
        packing=False,
        report_to="none",
        seed=SEED,
    )

    class TimeBudgetCallback(TrainerCallback):
        """Print pace early; stop cleanly at the budget so save + eval still run."""

        def __init__(self, budget_seconds):
            self.budget_seconds = budget_seconds
            self.start = None
            self.stopped_early = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.start
            if state.global_step in (5, 25, 100):
                per_step = elapsed / state.global_step
                print(f"[pace] step {state.global_step}: {per_step:.1f}s/step -> full epoch "
                      f"({state.max_steps} steps) ~= {per_step * state.max_steps / 3600:.1f} h",
                      flush=True)
            if elapsed > self.budget_seconds:
                print(f"[time budget] {elapsed / 3600:.2f} h at step {state.global_step}/"
                      f"{state.max_steps} — stopping early.", flush=True)
                self.stopped_early = True
                control.should_training_stop = True
            return control

    time_budget = TimeBudgetCallback(TRAIN_TIME_BUDGET_S)
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset,
                         peft_config=peft_config, processing_class=tokenizer,
                         callbacks=[time_budget])

    # Undo peft's fp32 upcast of the two big frozen modules (2.1 GB each on Llama).
    for name, module in trainer.model.named_modules():
        if name.endswith(("embed_tokens", "lm_head")):
            if any(p.dtype == torch.float32 for p in module.parameters()):
                module.to(torch.float16)
                print(f"Recast {name} to fp16", flush=True)

    preflight(trainer, model, tokenizer, sft_config, val_prompts, args, vram_gb)

    print("Starting training...", flush=True)
    result = trainer.train()
    loss = result.training_loss
    assert loss is not None and math.isfinite(loss) and loss > 0, f"Bad training loss: {loss}"
    print(f"Training finished — loss {loss:.4f}, {result.global_step} steps, "
          f"{result.metrics.get('train_runtime', 0) / 3600:.2f} h", flush=True)

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    assert (adapter_dir / "adapter_config.json").exists(), f"Adapter save failed: {adapter_dir}"

    train_metrics = {
        "model_id": run_config["model_id"],
        "final_training_loss": float(loss),
        "train_runtime_seconds": result.metrics.get("train_runtime"),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "global_step": result.global_step,
        "epochs_completed": trainer.state.epoch,   # not in result.metrics (Trainer adds it only to logs)
        "stopped_on_time_budget": time_budget.stopped_early,
        "train_time_budget_seconds": TRAIN_TIME_BUDGET_S,
        "hyperparameters": {
            "num_train_epochs": sft_config.num_train_epochs,
            "per_device_train_batch_size": sft_config.per_device_train_batch_size,
            "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
            "group_by_length": sft_config.group_by_length,
            "learning_rate": sft_config.learning_rate,
            "weight_decay": sft_config.weight_decay,
            "lr_scheduler_type": str(sft_config.lr_scheduler_type),
            "warmup_ratio": sft_config.warmup_ratio,
            "warmup_steps": sft_config.warmup_steps,
            "max_length": sft_config.max_length,
            "optim": sft_config.optim,
            "fp16": sft_config.fp16,
            "bf16": sft_config.bf16,
            "bnb_4bit_compute_dtype": "torch.float16",
            "bnb_4bit_use_double_quant": False,
            "completion_only_loss": sft_config.completion_only_loss,
            "packing": sft_config.packing,
            "seed": sft_config.seed,
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
            "lora_target_modules": sorted(peft_config.target_modules),
        },
    }
    write_json(out_dir / "train_metrics.json", train_metrics)

    eval_model = trainer.model
    eval_model.gradient_checkpointing_disable()
    return eval_model, train_metrics


def preflight(trainer, model, tokenizer, sft_config, val_prompts, args, vram_gb):
    """Fail in seconds, not hours: the T3 notebook's checks, generalised to any task/model."""
    import torch

    def ok(msg):
        print(f"  [OK]   {msg}", flush=True)

    print("Pre-flight checks:", flush=True)
    cc_major, _ = torch.cuda.get_device_capability(0)
    if cc_major < 8:
        assert sft_config.fp16 and not sft_config.bf16, "Pre-Ampere GPU: must train in fp16."
    ok(f"fp16={sft_config.fp16}, bf16={sft_config.bf16}")

    big_fp32 = [n for n, p in trainer.model.named_parameters()
                if p.dtype == torch.float32 and not p.requires_grad and p.numel() > 1e8]
    assert not big_fp32, f"Frozen fp32 tensors over 100M params: {big_fp32}"
    ok("no frozen fp32 tensor over 100M params")

    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    assert trainable and {p.dtype for p in trainable} == {torch.float32}, \
        "LoRA params must exist and be fp32 (GradScaler cannot unscale fp16 grads)."
    ok(f"LoRA trainable params: {sum(p.numel() for p in trainable):,} (fp32)")

    ds = trainer.train_dataset
    lengths = [len(ids) for ids in ds["input_ids"]]
    assert max(lengths) <= args.max_seq_len, f"Example of {max(lengths)} tokens > max_seq_len"
    ok(f"token lengths: max {max(lengths)}, mean {sum(lengths) // len(lengths)} <= {args.max_seq_len}")

    # Check 9: one real forward+backward on the WORST-CASE batch (longest example repeated).
    worst = max(range(len(lengths)), key=lengths.__getitem__)
    row = {k: ds[worst][k] for k in ("input_ids", "completion_mask") if k in ds.column_names}
    batch = trainer.data_collator([row] * sft_config.per_device_train_batch_size)
    unmasked = int((batch["labels"] != -100).sum())
    assert 0 < unmasked < batch["labels"].numel(), "Completion-only masking is broken."
    ok(f"completion-only loss: {unmasked} unmasked of {batch['labels'].numel()} label tokens")

    probe = {k: batch[k].to(model.device) for k in ("input_ids", "attention_mask", "labels") if k in batch}
    tokens = probe["input_ids"].numel()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        out = trainer.model(**probe)
    out.loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    trainer.model.zero_grad(set_to_none=True)
    del probe, out
    torch.cuda.empty_cache()
    # accelerate's ConvertOutputsToFp32 adds one more fp32 logits copy during real training.
    projected = peak_gb + tokens * model.config.vocab_size * 4 / 1e9
    assert projected < 0.85 * vram_gb, f"Worst-case batch projects to {projected:.1f} GB of {vram_gb:.1f}"
    ok(f"memory probe ({tokens} tokens): peak {peak_gb:.1f} GB, projected {projected:.1f} / {vram_gb:.1f} GB")

    # Check 10: the exact eval-path generate() on 2 prompts, before paying for training.
    tokenizer.padding_side = "left"
    trainer.model.eval()
    trainer.model.config.use_cache = True
    try:
        decoded = generate(trainer.model, tokenizer, val_prompts[:2], 2, args.max_seq_len)
    finally:
        tokenizer.padding_side = "right"
        trainer.model.config.use_cache = False
        trainer.model.train()
        torch.cuda.empty_cache()
    ok(f"eval-path generate works (untrained output {[d.strip() for d in decoded]!r})")


def evaluate(model, tokenizer, val_rows, val_prompts, task, metrics_mod, args, out_dir,
             run_config, train_metrics):
    """Greedy-decode the validation set, score it, and write eval_metrics.json,
    eval_report.txt and predictions.jsonl."""
    model.config.use_cache = True
    model.eval()
    tokenizer.padding_side = "left"   # generation continues from the prompt, not from padding

    bs = task["eval_batch_size"]
    start = time.time()
    predictions = []
    for i in range(0, len(val_rows), bs):
        raws = generate(model, tokenizer, val_prompts[i:i + bs], metrics_mod.MAX_NEW_TOKENS,
                        args.max_seq_len)
        for j, raw in enumerate(raws):
            row = val_rows[i + j]
            pred, valid = metrics_mod.parse_prediction(raw)
            predictions.append({"idx": i + j, "category": row["category"], "gold": row["output"],
                                "raw": raw, "pred": pred, "valid": valid,
                                "correct": metrics_mod.is_correct(row["output"], pred)})
        if len(predictions) % 200 < bs:
            print(f"  evaluated {len(predictions)}/{len(val_rows)} ({time.time() - start:.0f}s)", flush=True)
    eval_runtime = time.time() - start

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    metrics, report = metrics_mod.score([p["gold"] for p in predictions],
                                        [p["pred"] for p in predictions],
                                        [p["valid"] for p in predictions])
    metrics.update({
        "model": args.model, "model_id": run_config["model_id"], "mode": args.mode,
        "eval_runtime_seconds": eval_runtime, "max_seq_len": args.max_seq_len,
        "trimmed_val": run_config["trimmed_val"], "limit": args.limit,
        "stopped_on_time_budget": train_metrics["stopped_on_time_budget"] if train_metrics else None,
    })
    write_json(out_dir / "eval_metrics.json", metrics)
    with open(out_dir / "eval_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{args.model} ({run_config['model_id']}) task {args.task} mode {args.mode}\n\n{report}")
    print(report, flush=True)
    print(f"Wrote eval_metrics.json, eval_report.txt, predictions.jsonl to {out_dir}", flush=True)


# ----------------------------------------------------------------------------- compare

def compare(args):
    """Paired comparison of two runs on the same validation set.

    McNemar's exact test on per-example correctness (binomial test on the discordant pairs)
    and a paired bootstrap 95% CI for the difference (B - A) in the task's headline F1.
    """
    import numpy as np
    from scipy.stats import binomtest

    metrics_mod = TASKS[args.task]["metrics"]
    a = read_jsonl(Path(args.a) / "predictions.jsonl")
    b = read_jsonl(Path(args.b) / "predictions.jsonl")
    assert len(a) == len(b), f"Different number of predictions: {len(a)} vs {len(b)}"
    assert all(x["idx"] == y["idx"] and x["gold"] == y["gold"] for x, y in zip(a, b)), \
        "Runs were not evaluated on the same validation examples in the same order."

    gold = [x["gold"] for x in a]
    pred_a = [x["pred"] for x in a]
    pred_b = [y["pred"] for y in b]
    only_a = sum(x["correct"] and not y["correct"] for x, y in zip(a, b))
    only_b = sum(y["correct"] and not x["correct"] for x, y in zip(a, b))
    p_value = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0

    rng = np.random.default_rng(args.seed)
    n = len(gold)
    diffs = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        g = [gold[i] for i in idx]
        diffs.append(metrics_mod.headline_f1(g, [pred_b[i] for i in idx])
                     - metrics_mod.headline_f1(g, [pred_a[i] for i in idx]))

    result = {
        "task": args.task, "a": str(args.a), "b": str(args.b), "n": n,
        "accuracy_a": sum(x["correct"] for x in a) / n,
        "accuracy_b": sum(y["correct"] for y in b) / n,
        "mcnemar": {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p_value},
        "headline_f1_a": metrics_mod.headline_f1(gold, pred_a),
        "headline_f1_b": metrics_mod.headline_f1(gold, pred_b),
        "headline_f1_diff_b_minus_a_ci95": [float(np.percentile(diffs, 2.5)),
                                            float(np.percentile(diffs, 97.5))],
        "n_boot": args.n_boot, "seed": args.seed,
    }
    out = Path(args.out) if args.out else \
        Path(args.b).parent / f"compare_t{args.task}_{Path(args.a).name}_vs_{Path(args.b).name}.json"
    write_json(out, result)
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")


# ----------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="evaluate (and optionally fine-tune) one arm")
    r.add_argument("--task", type=int, choices=sorted(TASKS), required=True)
    r.add_argument("--model", choices=sorted(MODELS), required=True)
    r.add_argument("--mode", choices=["baseline", "finetune", "adapter"], required=True)
    r.add_argument("--val_file", required=True)
    r.add_argument("--train_file", help="required for --mode finetune")
    r.add_argument("--adapter_dir", help="required for --mode adapter")
    r.add_argument("--out_dir", required=True)
    r.add_argument("--max_seq_len", type=int, default=1024)
    r.add_argument("--limit", type=int, help="evaluate only the first N validation examples (smoke test)")
    r.add_argument("--train_limit", type=int, help="train on only the first N examples (smoke test)")

    c = sub.add_parser("compare", help="paired McNemar + bootstrap between two run dirs")
    c.add_argument("--task", type=int, choices=sorted(TASKS), required=True)
    c.add_argument("--a", required=True, help="run dir of the reference arm")
    c.add_argument("--b", required=True, help="run dir of the arm compared against A")
    c.add_argument("--out", help="output JSON (default: next to B)")
    c.add_argument("--n_boot", type=int, default=1000)
    c.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()
