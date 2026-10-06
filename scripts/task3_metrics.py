"""Task 3 (LEDGAR provision classification, 100 labels) prompt, answer parsing and scoring.

The parsing rule (first line, strip, case-insensitive exact label match, else INVALID) and the
metrics (sklearn F1 with labels=LABELS, so invalid predictions count as misses and never as
false positives) are extracted verbatim from the evaluation cells of
`llm_fine_tuning_LORA_task3.ipynb` and `llama_3.1_task_3_no_fine_tune.ipynb`, so every arm is
scored with the code that produced the published Llama numbers. CPU-only.

The label set comes from LEDGAR's labels.json and must be loaded with `load_labels(path)`
before parsing or scoring: macro-F1 averages over all 100 labels, and the validation set
covers only 99 of them, so the label set cannot be recovered from the data.

Per-example values flowing through the harness:
    gold = the JSONL `output` string (a label name; identical to `category`)
    pred = the canonical label name, or the INVALID sentinel
"""
import json
from collections import Counter

from sklearn.metrics import f1_score, precision_recall_fscore_support

TASK_NAME = "task3_provision_classification"
# The baseline notebook decoded 16 tokens; the fine-tune notebook decoded longest label + 2
# (Llama: 5 + 2). 16 covers every tokenizer (Saul/Mistral longest label: 8 tokens), and the
# parser reads only the first line, so the extra budget changes nothing once a label is emitted.
MAX_NEW_TOKENS = 16
INVALID = "__INVALID__"

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

LABELS = []            # id order, filled by load_labels()
LABEL_BY_LOWER = {}


def load_labels(path):
    """Load LEDGAR's labels.json ({"0": name, ...}) into LABELS / LABEL_BY_LOWER."""
    with open(path, encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    LABELS[:] = [id2label[i] for i in range(len(id2label))]
    LABEL_BY_LOWER.clear()
    LABEL_BY_LOWER.update({label.lower(): label for label in LABELS})


def _require_labels():
    assert LABELS, "task3_metrics.load_labels(labels.json) must be called first"


def instruction():
    """The notebooks' instruction (the JSONL rows already carry it; used by the tests)."""
    _require_labels()
    return ("Classify the following contract provision. "
            f"Answer with exactly one label from this list: [{', '.join(LABELS)}].")


def build_prompt(instruction_text, input_text):
    """Assemble the T3 prompt (same template as T1/T2). The gold label is the completion.
    Input trimming is done by the harness before this is called (the notebook's truncate_input)."""
    return PROMPT_TEMPLATE.format(instruction=instruction_text, input=input_text)


def parse_prediction(raw, category=None):
    """Map a raw completion to (pred, leniently_valid).

    pred: the notebooks' rule, unchanged — first line, stripped, must equal a label
        (case-insensitive); otherwise INVALID.
    leniently_valid: new diagnostic, not used by the notebooks — True when the first line
        STARTS with a label (e.g. "Governing Laws. This Agreement ..."), i.e. the right answer
        format followed by run-on text. `category` is unused (shared interface).
    """
    _require_labels()
    text = raw.strip().split("\n", 1)[0].strip().lower()
    canonical = LABEL_BY_LOWER.get(text)
    lenient = canonical is not None or any(text.startswith(label) for label in LABEL_BY_LOWER)
    return (canonical if canonical is not None else INVALID), lenient


def score(gold, pred, valid):
    """Aggregate metrics in the notebooks' eval_metrics.json schema, plus lenient validity.

    gold: label names; pred: values from parse_prediction; valid: its lenient flags.
    Returns (metrics_dict, report_text).
    """
    _require_labels()
    n = len(gold)
    valid_label_rate = sum(p != INVALID for p in pred) / n
    lenient_rate = sum(valid) / n
    accuracy = sum(g == p for g, p in zip(gold, pred)) / n
    macro_f1 = f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)
    micro_f1 = f1_score(gold, pred, labels=LABELS, average="micro", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(gold, pred, labels=LABELS,
                                                             zero_division=0)
    top_conf = Counter((g, p) for g, p in zip(gold, pred) if g != p).most_common(15)

    per_label_rows = sorted(
        ({"label": LABELS[i], "precision": prec[i], "recall": rec[i],
          "f1": f1[i], "support": int(support[i])} for i in range(len(LABELS))),
        key=lambda r: r["f1"],
    )
    hdr = f"{'Label':40s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'support':>8s}"
    tbl = [hdr, "-" * len(hdr)]
    for r in per_label_rows:
        tbl.append(f"{r['label']:40s} {r['precision']:9.2f} {r['recall']:7.2f} "
                   f"{r['f1']:6.2f} {r['support']:8d}")

    metrics = {
        "task": TASK_NAME,
        "n_validation_examples": n,
        "valid_label_rate": float(valid_label_rate),
        "lenient_valid_label_rate": float(lenient_rate),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "per_label": {
            LABELS[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                        "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(LABELS))
        },
        "top_confusions": [{"gold": g, "pred": p, "count": c} for (g, p), c in top_conf],
    }
    report_text = (
        "Task 3 — LEDGAR provision classification (validation)\n\n"
        f"Validation examples : {n}\n"
        f"Valid-label rate    : {valid_label_rate:.4f}\n"
        f"Accuracy            : {accuracy:.4f}\n"
        f"Macro-F1 (headline) : {macro_f1:.4f}\n"
        f"Micro-F1            : {micro_f1:.4f}\n"
        f"Lenient valid-label rate (first line starts with a label): {lenient_rate:.4f}\n"
        "\nTop 15 confused (gold -> pred) pairs:\n"
        + "".join(f"  {c:3d}  {g}  ->  {p}\n" for (g, p), c in top_conf)
        + "\nPer-label precision / recall / F1 (worst F1 first):\n"
        + "\n".join(tbl) + "\n"
    )
    return metrics, report_text


def is_correct(gold_label, pred):
    """Per-example correctness used by McNemar in `compare` (T3: exact label)."""
    return gold_label == pred


def headline_f1(gold, pred):
    """F1 metric bootstrapped in `compare`: macro-F1 over all 100 labels (the notebooks' headline)."""
    _require_labels()
    return f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)
