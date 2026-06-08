# GPU Scaling Options — From PoC to Full Training

Guidance for moving the QLoRA fine-tuning pipeline (Llama-3-8B / Mistral-7B) off a
small local GPU to a larger one, ranked by **free first, then lowest cost**.

> Prices are USD/hr and fluctuate — treat them as ballpark figures, confirm on the
> provider's pricing page before committing.

---

## Sizing first: how big a GPU do you actually need?

The current config is small:

- 4-bit NF4 quantization (`BitsAndBytesConfig`)
- LoRA `r=16`, `lora_alpha=16`, targets `["q_proj", "v_proj"]` only
- `max_seq_length=2048`, optimizer `paged_adamw_32bit`

This trains in roughly **12–16 GB VRAM**. So "bigger GPU" mostly means **more
runtime, fewer disconnects, and faster epochs** — not 80 GB cards.

A single **16–24 GB GPU** is the sweet spot. You only need an A100/H100 if you:

- raise `max_seq_length` substantially (full contracts),
- add more LoRA target modules,
- switch to full fine-tuning instead of LoRA, or
- move to a 70B model.

---

## Free options (ranked for this workload)

| Platform | GPU | VRAM | Free quota | Notes | Source |
|---|---|---|---|---|---|
| **Kaggle Notebooks** ⭐ | 2× T4 *or* P100 | 30 GB (2×T4) / 16 GB | **~30 GPU-hrs/week** | Best free option. 9–12 hr sessions, real weekly budget. | https://www.kaggle.com/code |
| **Google Colab (free)** | T4 | 16 GB | Unpredictable, disconnects | Fine for PoC, painful for long runs (idle timeouts). | https://colab.research.google.com |
| **Lightning AI Studio** | L4 | 24 GB | ~22 free GPU-hrs/month | L4 newer/faster than T4; VS Code-like env. | https://lightning.ai |
| **SageMaker Studio Lab** | T4 | 16 GB | 4 hr/session, 8 hr/day | Free but waitlist/approval can be slow. | https://studiolab.sagemaker.aws |
| **Paperspace Gradient (free)** | M4000 / limited | 8 GB | Free tier (often queued) | Weaker card, tight VRAM — last resort. | https://www.paperspace.com/gradient |

### Recommendation (free): **Kaggle**

30 hr/week and 2×T4 (30 GB combined) is enough to train the full CUAD dataset across
all 41 categories without babysitting disconnects. Use `device_map="auto"` so
`accelerate` shards the 4-bit model across both T4s.

**Migration steps:**

1. Upload `master_clauses_cleaned.csv` (and generated JSONL) as a **Kaggle Dataset**
   → mounts read-only at `/kaggle/input/...`.
2. Set `DATA_DIR` to that path; write outputs to `/kaggle/working/` (only writable
   dir, persisted as run output).
3. Add your HF token via **Add-ons → Secrets** instead of interactive
   `huggingface_hub.login()` (needed for the gated Llama-3 download).
4. Settings → Accelerator → **GPU T4 ×2**, and turn **Internet on**.

---

## Lowest-cost paid options (if you outgrow free)

Cheapest first. GPU marketplaces (Vast/RunPod) undercut the big clouds.

| Platform | Cheap card | VRAM | ~$/hr | Why | Source |
|---|---|---|---|---|---|
| **Vast.ai** | RTX 3090 / 4090 | 24 GB | **$0.20–0.40** | Cheapest anywhere (spot marketplace). Bring a Docker image. Interruptible. | https://vast.ai |
| **RunPod** | RTX 4090 | 24 GB | **$0.34–0.44** | Best UX/price balance. One-click PyTorch + Jupyter. "Community Cloud" = cheapest. | https://www.runpod.io |
| **RunPod / Lambda** | A40 / A6000 | 48 GB | $0.40–0.80 | Headroom for longer context or 13B. | https://www.runpod.io |
| **TensorDock** | RTX 4090 / A6000 | 24–48 GB | $0.30–0.60 | Cheap marketplace alternative to Vast. | https://www.tensordock.com |
| **Lambda Labs** | A100 | 40 GB | ~$1.10 | Cleanest for serious runs; on-demand often sold out. | https://lambdalabs.com/service/gpu-cloud |
| **Modal** | A10G / A100 (serverless) | 24–80 GB | per-second + free credits | Script runs instead of babysitting a notebook. Free monthly credits. | https://modal.com |
| **Google Colab Pro / Pro+** | T4 / L4 / A100 | 16–40 GB | $10 / $50 per month | Easiest upgrade if you already live in Colab. | https://colab.research.google.com/signup |

### Recommendation (paid): **RunPod, RTX 4090, Community Cloud (~$0.35/hr)**

24 GB is 1.5–2× what the job needs (room for longer sequences or more LoRA targets),
and a full multi-epoch run over CUAD is single-digit dollars.

**Steps:** pick a "PyTorch 2.x" template → attach a **network volume** (so checkpoints
survive pod termination) → `git clone` the repo → run the notebook via built-in Jupyter.

**Absolute floor:** Vast.ai with a 3090 at ~$0.25/hr (~half the cost) — just checkpoint
frequently to a persistent volume since instances can be reclaimed.

---

## Bottom line

Since training fits in ~16 GB, **finish on Kaggle free first.** Only move to paid when
you specifically need longer `max_seq_length` (full contracts), a 70B model, full
fine-tuning, or you're iterating fast enough that the 30 hr/week cap is the bottleneck.
