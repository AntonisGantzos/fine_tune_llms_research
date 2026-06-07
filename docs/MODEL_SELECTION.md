# Model Selection — Open-Source LLMs for CUAD Fine-Tuning

This document recommends Hugging Face models for the two stages of this project:

1. **Validate the pipeline (proof of concept)** — small, fast-to-download, fast-to-train models whose only job is to prove the data → format → QLoRA → save → inference loop works end-to-end. Accuracy is irrelevant here.
2. **Actually perform the task** — capable models that fit on consumer hardware via QLoRA and can realistically learn the three CUAD/LEDGAR tasks (T1 risk-clause recognition, T2 entity extraction, T3 jurisdiction classification).

> **Context for the choices below**
> - Hardware assumption: a single consumer GPU (8–24 GB VRAM). All "perform" picks are sized for QLoRA (4-bit NF4) within that budget.
> - Task shape: legal contract text, often long clauses, requiring subtle distinctions and **valid-JSON output** (T2). This rewards stronger instruction-following and longer context.
> - License matters for a research project that may be published or commercialized — noted per model.

---

## TL;DR recommendations

| Stage | First choice | Why |
|-------|--------------|-----|
| **Validate pipeline** | `meta-llama/Llama-3.2-1B` | Same Llama tokenizer/family as the real target, ~2.5 GB download, trains in minutes. Zero surprises when you scale up. |
| **Validate (no gated access)** | `Qwen/Qwen2.5-1.5B` or `HuggingFaceTB/SmolLM2-1.7B` | Ungated, tiny, permissive license. Use if Llama access/download is the blocker. |
| **Perform the task** | `Qwen/Qwen2.5-7B-Instruct` | Strong instruction-following + reliable JSON, 32k context (fits long clauses), Apache-2.0. Best all-round for T1/T2/T3 on consumer hardware. |
| **Perform (legal-specialized)** | `Equall/Saul-7B-Instruct-v1` | Pretrained on a large legal corpus; strong domain prior for contract language. MIT license. |

---

## Stage 1 — Validate the pipeline (proof of concept)

**Goal:** confirm the whole notebook runs — tokenization, prompt template, `SFTTrainer`, adapter save/merge, and a sane generation — *without waiting hours*. Pick the smallest thing that shares the target model's plumbing.

| Model | Params | ~fp16 download | QLoRA VRAM* | License | Notes |
|-------|--------|----------------|-------------|---------|-------|
| **`meta-llama/Llama-3.2-1B`** ⭐ | 1.2B | ~2.5 GB | ~3–4 GB | Llama 3.2 (gated) | **Best POC pick** — same tokenizer/chat template/family as the 8B target, so the pipeline transfers 1:1. |
| `meta-llama/Llama-3.2-3B` | 3.2B | ~6 GB | ~5–6 GB | Llama 3.2 (gated) | Slightly more capable POC; download still ~5 h at 326 kB/s, ~minutes on a normal connection. |
| `Qwen/Qwen2.5-1.5B` | 1.5B | ~3 GB | ~4 GB | Apache-2.0 | **Ungated** — no HF approval needed. Great if Llama access is the bottleneck. |
| `HuggingFaceTB/SmolLM2-1.7B` | 1.7B | ~3.4 GB | ~4 GB | Apache-2.0 | Ungated, built for cheap experimentation. |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | ~2.2 GB | ~3 GB | Apache-2.0 | Smallest/fastest. Weakest quality, but for a smoke-test that doesn't matter. |

\* *Rough QLoRA footprint at `max_seq_length=2048`, batch size 1–2: 4-bit base weights (~0.5 GB/B params) + adapter optimizer state + activations.*

**Why match the family:** if you POC on a Llama-family model, the chat template, special tokens, `target_modules` (`q_proj`/`v_proj`), and tokenizer all behave identically when you swap in Llama-3-8B. POC'ing on a different family (e.g. Qwen) still validates the *logic* but you may have to re-tune the prompt template and target modules. Use a same-family small model if you can; fall back to Qwen/SmolLM only if Llama gating/download blocks you.

**Expectation for a POC run:** with a 1–1.5B model on a sampled CUAD set, a full train+eval cycle should complete in **minutes** on a consumer GPU — its purpose is to surface bugs, not to produce a usable model.

---

## Stage 2 — Actually perform the task

**Goal:** a model that genuinely learns the tasks. Selection criteria, in priority order:

1. **Instruction-following + structured output** — T2 requires valid JSON; weak models hallucinate or break format.
2. **Context length** — CUAD clauses + prompt can be long; 8k is the floor, 32k is comfortable.
3. **Fits QLoRA on consumer VRAM** — 7–14B is the sweet spot for 16–24 GB; up to ~32B possible on 24 GB at 4-bit.
4. **License** — prefer Apache-2.0/MIT for research you may publish or commercialize.

### General-purpose (recommended)

| Model | Params | Context | QLoRA VRAM* | License | Why it fits |
|-------|--------|---------|-------------|---------|-------------|
| **`Qwen/Qwen2.5-7B-Instruct`** ⭐ | 7B | 32k | ~10–12 GB | Apache-2.0 | **Top all-round pick.** Excellent instruction-following and reliable JSON, long context for full clauses, fully permissive. |
| `meta-llama/Meta-Llama-3.1-8B` (or `-Instruct`) | 8B | 128k | ~11–13 GB | Llama 3.1 (gated) | The project's current target, refreshed. Use `Instruct` if you want a chat-tuned base; use the base model to fine-tune from scratch as the notebook does. |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 32k | ~10–12 GB | Apache-2.0 | Lightweight, well-supported, permissive. Slightly behind Qwen2.5 on instruction benchmarks but very stable. |
| `google/gemma-2-9b-it` | 9B | 8k | ~12–14 GB | Gemma (permissive) | Strong quality; **8k context** is the main limitation for long clauses. |

### Legal-specialized (domain prior)

| Model | Params | Context | License | Why consider it |
|-------|--------|---------|---------|-----------------|
| **`Equall/Saul-7B-Instruct-v1`** ⭐ | 7B (Mistral-based) | 32k | MIT | **SaulLM** — continued-pretrained on a large legal corpus. Starts with a strong prior for contract/legal language, so it can need less data to reach good T1/T2 performance. Drop-in size with Mistral-7B. |
| `Equall/SaulLM-7B` (base) | 7B | 32k | MIT | Non-instruct base variant — fine-tune from scratch like the notebook's current Llama base. |

> SaulLM also has 54B/141B variants (Mixtral-based, MIT) that are state-of-the-art on LegalBench but **do not fit consumer hardware** even at 4-bit — listed for awareness only.

### If you have more VRAM (24 GB+)

| Model | Params | Context | License | Notes |
|-------|--------|---------|---------|-------|
| `Qwen/Qwen2.5-14B-Instruct` | 14B | 32k | Apache-2.0 | Noticeably stronger than 7B; QLoRA fits ~18–22 GB. Best quality/cost on a single 24 GB card. |
| `mistralai/Mistral-Nemo-Instruct-2407` | 12B | 128k | Apache-2.0 | Long context + strong multilingual; good middle ground. |

\* *QLoRA footprint at `max_seq_length=2048`. Longer sequences and larger batches increase activation memory — reduce batch size / use gradient checkpointing if you OOM.*

---

## How the choices map to the three tasks

| Task | Output type | What it stresses | Best fit |
|------|-------------|------------------|----------|
| **T1 – Risk clause recognition** | Binary Yes/No (32 categories) | Class imbalance, subtle legal distinctions | Qwen2.5-7B or **Saul-7B** (domain prior helps rare clauses) |
| **T2 – Structured entity extraction** | **Valid JSON** (9 categories) | Format reliability, long context | **Qwen2.5-7B-Instruct** (strongest JSON adherence) |
| **T3 – Jurisdiction identification** | Provision classification (LEDGAR) | Straightforward classification | Any 7–8B general model; Mistral-7B is sufficient |

---

## Practical notes & justifications

- **Download is your current bottleneck, not compute.** At the observed 326 kB/s, size in GB ≈ hours. A 1B POC model (~2.5 GB) downloads in well under the 8B's ~14 h, letting you validate the pipeline *today* while the larger model downloads in parallel or later. Pre-pull with `huggingface-cli download <repo>` so it's cached and resumable.
- **Instruct vs. base.** The notebook fine-tunes a base model from scratch with a custom prompt template. For the *perform* stage you can stay on base models, but an `-Instruct` variant often converges faster for JSON/format tasks because it already follows instructions. Pick one and keep the prompt template consistent.
- **License summary.** Qwen2.5 (Apache-2.0), Mistral (Apache-2.0), SaulLM (MIT) are the most permissive. Llama 3.x is gated and under the Llama Community License (free for most uses, with conditions). Gemma uses Google's permissive Gemma license.
- **Consider Unsloth** for the perform stage: drop-in QLoRA trainer reported at ~2× faster / ~60% less VRAM than vanilla HF `SFTTrainer`, which lets you fit a larger model (e.g. 14B) or longer sequences on the same card.
- **Benchmark to consult:** *ContractEval* (clause-level legal risk identification in commercial contracts) is a directly relevant yardstick for T1 if you want to compare your fine-tuned model against published baselines.

---

## Suggested concrete plan

1. **POC:** `meta-llama/Llama-3.2-1B` (or `Qwen/Qwen2.5-1.5B` if Llama is gated/slow) on `master_clauses_cleaned_sampled.csv` → confirm full loop runs.
2. **Perform:** `Qwen/Qwen2.5-7B-Instruct` as the primary model; benchmark against `Equall/Saul-7B-Instruct-v1` to test whether the legal domain prior beats a stronger general model on T1/T2.
3. **Stretch (24 GB):** `Qwen/Qwen2.5-14B-Instruct` if VRAM allows and 7B underperforms.

---

### Sources

- [Best Open-Source LLMs 2026 — Qwen, GLM, DeepSeek & Llama Compared](https://www.buildfastwithai.com/blogs/collection/open-source-llms)
- [Fine-Tune LLMs on Your Laptop with QLoRA in 2026](https://www.buildmvpfast.com/blog/fine-tune-llm-laptop-qlora-local-gpu-2026)
- [Best GPU for local LLM Inference and Training – 2026 (BIZON)](https://bizon-tech.com/blog/best-gpu-llm-training-inference)
- [SaulLM-54B & SaulLM-141B: Scaling Up Domain Adaptation for the Legal Domain (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea3f85a33f9ba072058e3df233cf6cca-Paper-Conference.pdf)
- [ContractEval: Benchmarking LLMs for Clause-Level Legal Risk Identification in Commercial Contracts (arXiv)](https://arxiv.org/html/2508.03080)
- [The Best Open Source LLM for Contract Processing & Review in 2026 (SiliconFlow)](https://www.siliconflow.com/articles/en/best-open-source-llm-for-contract-processing-review)
