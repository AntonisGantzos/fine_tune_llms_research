"""Task 2 (CUAD structured entity extraction, single-key JSON) prompt, answer parsing and scoring.

`parse_prediction`'s strict rule, `norm`, `token_f1` and `value_f1` are extracted verbatim from
the evaluation cell of `llm_fine_tuning_LORA_task2.ipynb` (identical in
`llama_3.1_task_2_no_fine_tune.ipynb`) so every arm is scored with the code that produced the
published Llama numbers. The pandas groupby aggregation is re-done in plain Python with the
same result. CPU-only: no torch / transformers imports.

Per-example values flowing through the harness:
    gold  = the JSONL `output` string, e.g. '{"Governing Law": "Nevada"}'
    pred  = the parsed value (None | str | list[str]) when the completion is valid JSON,
            else the INVALID sentinel — kept distinct from a valid `null` prediction, which is
            a correct answer when the gold value is null.
"""
import json
from collections import Counter

TASK_NAME = "task2_entity_extraction"
MAX_NEW_TOKENS = 128
INVALID = "__INVALID__"

# Canonical category order of the notebooks' eval_report.txt / eval_metrics.json.
CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law", "Warranty Duration",
]

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def build_prompt(instruction, input_text):
    """Assemble the T2 prompt (same template as T1). The gold `output` JSON is the completion."""
    return PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)


def _single_key_value(obj, category):
    """(value, True) if obj is a dict whose only key is `category`, else (None, False)."""
    if not isinstance(obj, dict) or set(obj.keys()) != {category}:
        return None, False
    return obj[category], True


def parse_prediction(raw, category):
    """Map a raw completion to (pred, leniently_valid).

    pred: the notebooks' STRICT rule, unchanged so published numbers reproduce — json.loads of
        the WHOLE stripped completion must give {category: value}; otherwise INVALID. A base
        model that writes a correct object and then keeps generating is therefore invalid.
    leniently_valid: new diagnostic, not used by the notebooks — True when the completion
        STARTS with a valid {category: value} object, ignoring anything after it. The gap
        between the two rates is "right JSON, didn't stop", which matters for zero-shot models.
    """
    text = raw.strip()
    try:
        value, strict_ok = _single_key_value(json.loads(text), category)
    except json.JSONDecodeError:
        value, strict_ok = None, False
    try:
        lenient_ok = _single_key_value(json.JSONDecoder().raw_decode(text)[0], category)[1]
    except json.JSONDecodeError:
        lenient_ok = False
    return (value if strict_ok else INVALID), lenient_ok


def norm(v):
    """Light normalization for exact match: lowercase + strip; lists become
    order-insensitive (sorted tuples) — the order of parties is not meaningful."""
    if v is None:
        return None
    if isinstance(v, list):
        return tuple(sorted(str(x).strip().lower() for x in v))
    return str(v).strip().lower()


def token_f1(pred, gold):
    """SQuAD-style word-overlap F1 between two scalar values."""
    p, g = str(pred).lower().split(), str(gold).lower().split()
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def value_f1(pred, gold):
    """Token F1 extended to None and list values. For lists: greedily align each
    gold item to its best-matching predicted item; unmatched items on either
    side count as 0 (divide by max(len(pred), len(gold)))."""
    if gold is None and pred is None:
        return 1.0            # correctly said "not there"
    if gold is None or pred is None:
        return 0.0            # hallucinated a value, or missed a present one
    pred_list = pred if isinstance(pred, list) else [pred]
    gold_list = gold if isinstance(gold, list) else [gold]
    used, scores = set(), []
    for g in gold_list:
        best, best_j = 0.0, None
        for j, p in enumerate(pred_list):
            if j in used:
                continue
            s = token_f1(p, g)
            if s > best:
                best, best_j = s, j
        if best_j is not None:
            used.add(best_j)
        scores.append(best)
    return sum(scores) / max(len(pred_list), len(gold_list))


def _gold_category_value(gold_output):
    """'{"<category>": value}' -> (category, value)."""
    (category, value), = json.loads(gold_output).items()
    return category, value


def _example_scores(gold_output, pred):
    """The notebook's per-example row: json_valid, exact_match, f1 (0 when invalid)."""
    _, gold = _gold_category_value(gold_output)
    valid = not (isinstance(pred, str) and pred == INVALID)
    return {"json_valid": valid,
            "exact_match": bool(valid and norm(pred) == norm(gold)),
            "f1": value_f1(pred, gold) if valid else 0.0}


def _means(rows):
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in ("json_valid", "exact_match", "f1")}


def score(gold, pred, valid):
    """Aggregate metrics in the notebooks' eval_metrics.json schema, plus lenient validity.

    gold: JSONL `output` strings; pred: values from parse_prediction; valid: its lenient flags.
    Returns (metrics_dict, report_text). `overall`, `per_category` and `per_category_counts`
    match the notebooks' keys and values.
    """
    rows = [_example_scores(g, p) for g, p in zip(gold, pred)]
    cats = [_gold_category_value(g)[0] for g in gold]
    by_cat = {c: [r for r, rc in zip(rows, cats) if rc == c] for c in CATEGORIES}
    per_category = {c: _means(rs) for c, rs in by_cat.items() if rs}
    overall = _means(rows)
    lenient_rate = sum(valid) / len(valid)

    header = f"{'Category':45s} {'json_valid':>10s} {'exact_match':>12s} {'token_f1':>9s}"
    lines = [header]
    for c, m in per_category.items():
        lines.append(f"{c:45s} {m['json_valid']:10.2f} {m['exact_match']:12.2f} {m['f1']:9.2f}")
    lines.append("-" * len(header))
    lines.append(f"{'OVERALL':45s} {overall['json_valid']:10.2f} {overall['exact_match']:12.2f} "
                 f"{overall['f1']:9.2f}")

    metrics = {
        "task": TASK_NAME,
        "n_validation_examples": len(gold),
        "overall": overall,
        "lenient_json_valid_rate": lenient_rate,
        "per_category": per_category,
        "per_category_counts": dict(Counter(cats)),
    }
    report_text = (
        "Task 2 — JSON validity / exact match / token F1 on validation:\n\n"
        + "\n".join(lines)
        + f"\n\nLenient JSON-valid rate (completion starts with a valid object): {lenient_rate:.4f}\n"
    )
    return metrics, report_text


def is_correct(gold_output, pred):
    """Per-example correctness used by McNemar in `compare` (T2: exact match)."""
    return _example_scores(gold_output, pred)["exact_match"]


def headline_f1(gold, pred):
    """F1 metric bootstrapped in `compare`: mean token/value F1 (invalid counts 0)."""
    return sum(_example_scores(g, p)["f1"] for g, p in zip(gold, pred)) / len(gold)
