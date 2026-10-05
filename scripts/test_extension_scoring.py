"""CPU-only tests for the extension's extracted scorers. No pytest needed:

    python scripts/test_extension_scoring.py      (pytest also discovers the test_* functions)

The reproduction tests rebuild per-example (gold, pred) pairs from the confusion matrix saved
by each original Kaggle run — the runs did not save per-example predictions — and check that
the extracted scorer returns the saved eval_metrics.json numbers exactly. That verifies the
aggregation; the parsing rules are covered by the hand-written cases.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import task1_metrics as t1  # noqa: E402

T1_RUNS = [
    ROOT / "kaggle_output_task1_fine_tuned" / "eval_metrics.json",
    ROOT / "kaggle_output_task1_baseline" / "no_finetune_baseline" / "eval_metrics.json",
]


def test_t1_prompt_matches_notebook():
    expected = ("### Instruction:\nIs it X?\n\n"
                "### Input:\nsome clause\n\n"
                "### Response:\n")
    assert t1.build_prompt("Is it X?", "some clause") == expected


def test_t1_parse_valid():
    assert t1.parse_prediction("Yes") == ("Yes", True)
    assert t1.parse_prediction(" No\n") == ("No", True)
    assert t1.parse_prediction("yes.") == ("Yes", True)
    assert t1.parse_prediction("NO") == ("No", True)


def test_t1_parse_lenient_matches_notebook():
    # Notebook rule: "yes" anywhere -> Yes, everything else -> No. Kept for reproducibility.
    assert t1.parse_prediction("Yes, it")[0] == "Yes"
    assert t1.parse_prediction("eyes")[0] == "Yes"        # substring quirk, preserved on purpose
    assert t1.parse_prediction("")[0] == "No"
    assert t1.parse_prediction("### Input")[0] == "No"
    assert t1.parse_prediction("Maybe")[0] == "No"


def test_t1_parse_strict_validity():
    assert t1.parse_prediction("Yes, it")[1] is False
    assert t1.parse_prediction("")[1] is False
    assert t1.parse_prediction("Maybe")[1] is False
    assert t1.parse_prediction("No\n###")[1] is True      # first line decides


def test_t1_score_small_case():
    metrics, report = t1.score(["Yes", "No", "No", "Yes"], ["Yes", "No", "Yes", "No"],
                               [True, True, False, True])
    assert metrics["accuracy"] == 0.5
    assert metrics["strict_valid_rate"] == 0.75
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]
    assert "Strict-valid rate" in report


def _pairs_from_confusion(matrix):
    """[[TP, FN], [FP, TN]] with rows=true [Yes, No], cols=pred [Yes, No] -> (gold, pred)."""
    gold, pred = [], []
    for i, true_label in enumerate(t1.LABELS):
        for j, pred_label in enumerate(t1.LABELS):
            gold += [true_label] * matrix[i][j]
            pred += [pred_label] * matrix[i][j]
    return gold, pred


def test_t1_reproduces_saved_kaggle_metrics():
    for path in T1_RUNS:
        if not path.exists():   # kaggle_output_* is git-ignored; skip on a fresh clone
            print(f"  skip (missing): {path}")
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        gold, pred = _pairs_from_confusion(saved["confusion_matrix"]["matrix"])
        metrics, _ = t1.score(gold, pred, [True] * len(gold))
        assert metrics["n_validation_examples"] == saved["n_validation_examples"]
        assert metrics["accuracy"] == saved["accuracy"], (metrics["accuracy"], saved["accuracy"])
        assert metrics["classification_report"] == saved["classification_report"]
        assert metrics["confusion_matrix"] == saved["confusion_matrix"]
        print(f"  reproduced {path.relative_to(ROOT)}: accuracy {metrics['accuracy']:.4f}")


def test_compare_cli_on_synthetic_runs():
    """End-to-end `compare`: B fixes 3 of A's errors and breaks 1."""
    gold = ["Yes", "No"] * 10
    pred_a = list(gold)
    pred_b = list(gold)
    for i in (0, 1, 2):
        pred_a[i] = "No" if gold[i] == "Yes" else "Yes"
    pred_b[5] = "No" if gold[5] == "Yes" else "Yes"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, pred in (("a", pred_a), ("b", pred_b)):
            (tmp / name).mkdir()
            with open(tmp / name / "predictions.jsonl", "w", encoding="utf-8") as f:
                for i, (g, p) in enumerate(zip(gold, pred)):
                    f.write(json.dumps({"idx": i, "gold": g, "pred": p, "correct": g == p}) + "\n")
        out = tmp / "cmp.json"
        subprocess.run([sys.executable, str(SCRIPTS / "legal_model_extension.py"), "compare",
                        "--task", "1", "--a", str(tmp / "a"), "--b", str(tmp / "b"),
                        "--out", str(out), "--n_boot", "200"],
                       check=True, capture_output=True)
        result = json.loads(out.read_text(encoding="utf-8"))
    assert result["mcnemar"]["only_a_correct"] == 1
    assert result["mcnemar"]["only_b_correct"] == 3
    assert result["accuracy_a"] == 17 / 20 and result["accuracy_b"] == 19 / 20
    assert abs(result["mcnemar"]["p_value"] - 0.625) < 1e-9   # exact binomial, 1 vs 3
    lo, hi = result["headline_f1_diff_b_minus_a_ci95"]
    assert lo <= hi


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\nAll {len(tests)} tests passed.")
