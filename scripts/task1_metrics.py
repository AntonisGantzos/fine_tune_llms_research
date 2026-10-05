"""Task 1 (CUAD risk-clause recognition, Yes/No) prompt, answer parsing and scoring.

Extracted verbatim from the `predict` / evaluation cells of
`llm_fine_tuning_LORA_task1_v2.ipynb` (identical in `llama_3.1_task_1_no_fine_tune.ipynb`)
so the legal-model extension scores every arm with exactly the code that produced the
published Llama numbers. CPU-only: no torch / transformers imports.
"""
from sklearn.metrics import classification_report, confusion_matrix

TASK_NAME = "task1_risk_clause_recognition"
LABELS = ["Yes", "No"]
MAX_NEW_TOKENS = 3

# Training and eval used this exact template; `### Response:\n` is where completion-only
# loss stops masking.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def build_prompt(instruction, input_text):
    """Assemble the T1 prompt. The completion (gold `output`) is appended after it."""
    return PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)


def parse_prediction(raw):
    """Map a raw 3-token completion to (label, strictly_valid).

    label: the notebooks' LENIENT rule, kept unchanged so published accuracies reproduce —
        "Yes" if the substring "yes" occurs anywhere (case-insensitive), else "No". So any
        garbage, including an empty string, counts as a "No" prediction.
    strictly_valid: new diagnostic, not used by the notebooks — True only when the first
        line of the completion is exactly "yes" or "no" (case-insensitive, surrounding
        whitespace and a trailing period ignored). Separates format adherence from content,
        which matters for zero-shot base models.
    """
    label = "Yes" if "yes" in raw.strip().lower() else "No"
    first_line = raw.strip().split("\n", 1)[0].strip().rstrip(".").lower()
    return label, first_line in ("yes", "no")


def score(gold, pred, valid):
    """Aggregate metrics in the notebooks' eval_metrics.json schema, plus strict validity.

    gold, pred: lists of "Yes"/"No"; valid: list of bools from parse_prediction.
    Returns (metrics_dict, report_text). The dict keys `n_validation_examples`, `accuracy`,
    `classification_report` and `confusion_matrix` match the notebooks byte for byte.
    """
    report_str = classification_report(gold, pred, labels=LABELS, zero_division=0)
    report_dict = classification_report(gold, pred, labels=LABELS, zero_division=0,
                                        output_dict=True)
    cm = confusion_matrix(gold, pred, labels=LABELS)
    strict_valid_rate = sum(valid) / len(valid)
    metrics = {
        "task": TASK_NAME,
        "n_validation_examples": len(gold),
        "accuracy": report_dict["accuracy"],
        "strict_valid_rate": strict_valid_rate,
        "classification_report": report_dict,
        "confusion_matrix": {
            "labels": LABELS,
            "rows_are_true_cols_are_pred": True,
            "matrix": cm.tolist(),
        },
    }
    report_text = (
        "Per-class precision / recall / F1 on validation:\n\n"
        f"{report_str}\n\n"
        "Confusion matrix (rows = true [Yes, No], cols = pred [Yes, No]):\n"
        f"{cm}\n\n"
        f"Strict-valid rate (first line exactly Yes/No): {strict_valid_rate:.4f}\n"
    )
    return metrics, report_text


def is_correct(gold_label, pred_label):
    """Per-example correctness used by McNemar in `compare` (T1: accuracy)."""
    return gold_label == pred_label


def headline_f1(gold, pred):
    """F1 metric bootstrapped in `compare`: macro-F1 over Yes/No."""
    return classification_report(gold, pred, labels=LABELS, zero_division=0,
                                 output_dict=True)["macro avg"]["f1-score"]
