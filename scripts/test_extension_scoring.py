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
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import task1_metrics as t1  # noqa: E402
import task2_metrics as t2  # noqa: E402
import task3_metrics as t3  # noqa: E402

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


# ----------------------------------------------------------------------------- Task 2

T2_VAL = ROOT / "cuad" / "validation" / "cuad_task2_validation.jsonl"
T2_RUNS = [
    ROOT / "kaggle_output_task2_fine_tuned" / "eval_metrics.json",
    ROOT / "kaggle_output_task2_baseline" / "no_finetune_baseline_task2" / "eval_metrics.json",
]
GL = '{"Governing Law": "Nevada"}'


def test_t2_prompt_matches_notebook():
    assert t2.build_prompt("Is it X?", "some clause") == t1.build_prompt("Is it X?", "some clause")


def test_t2_parse_strict_matches_notebook():
    assert t2.parse_prediction(' {"Governing Law": "Nevada"}\n', "Governing Law") == ("Nevada", True)
    assert t2.parse_prediction('{"Parties": ["A", "B"]}', "Parties") == (["A", "B"], True)
    assert t2.parse_prediction('{"Governing Law": null}', "Governing Law") == (None, True)
    # Trailing text: notebook rule says invalid; the lenient diagnostic still sees the object.
    assert t2.parse_prediction('{"Governing Law": "Nevada"}\n\n### Input', "Governing Law") \
        == (t2.INVALID, True)
    assert t2.parse_prediction('{"Law": "Nevada"}', "Governing Law") == (t2.INVALID, False)
    assert t2.parse_prediction('{"Governing Law": "Nevada", "x": 1}', "Governing Law") \
        == (t2.INVALID, False)
    assert t2.parse_prediction('Nevada', "Governing Law") == (t2.INVALID, False)
    assert t2.parse_prediction('', "Governing Law") == (t2.INVALID, False)


def test_t2_value_scoring_rules():
    assert t2.norm(["B ", "a"]) == t2.norm(["A", "b"])               # list order ignored
    assert t2.value_f1(None, None) == 1.0
    assert t2.value_f1("Nevada", None) == 0.0
    assert t2.value_f1(["A Corp", "B Inc"], ["A Corp"]) == 0.5       # extra item penalised
    assert abs(t2.token_f1("State of Nevada", "Nevada") - 0.5) < 1e-12


def test_t2_invalid_is_not_a_null_prediction():
    null_gold = '{"Warranty Duration": null}'
    assert t2.is_correct(null_gold, None) is True
    assert t2.is_correct(null_gold, t2.INVALID) is False
    metrics, report = t2.score([GL, GL, null_gold], ["Nevada", t2.INVALID, None],
                               [True, True, False])
    assert metrics["overall"] == {"json_valid": 2 / 3, "exact_match": 2 / 3, "f1": 2 / 3}
    assert metrics["per_category"]["Governing Law"]["json_valid"] == 0.5
    assert metrics["lenient_json_valid_rate"] == 2 / 3
    assert t2.headline_f1([GL, GL, null_gold], ["Nevada", t2.INVALID, None]) == 2 / 3
    assert "Lenient JSON-valid rate" in report


def test_t2_matches_saved_kaggle_validation_set():
    """The runs saved no per-example predictions, so check what can be checked: the repo
    validation set is the one they scored (same n and per-category counts), and a perfect
    prediction scores 1.0 on every metric in the notebooks' schema."""
    rows = [json.loads(line) for line in T2_VAL.read_text(encoding="utf-8").splitlines() if line]
    gold = [r["output"] for r in rows]
    perfect = [json.loads(g)[r["category"]] for g, r in zip(gold, rows)]
    metrics, _ = t2.score(gold, perfect, [True] * len(gold))
    assert metrics["overall"] == {"json_valid": 1.0, "exact_match": 1.0, "f1": 1.0}
    assert list(metrics["per_category"]) == t2.CATEGORIES
    for path in T2_RUNS:
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert metrics["n_validation_examples"] == saved["n_validation_examples"]
        assert metrics["per_category_counts"] == saved["per_category_counts"]
        assert list(saved["per_category"]) == t2.CATEGORIES
        print(f"  matched {path.relative_to(ROOT)}: n={saved['n_validation_examples']}")


def test_compare_cli_task2():
    gold = [GL] * 4
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, pred in (("a", ["Nevada", t2.INVALID, "Texas", "Nevada"]),
                           ("b", ["Nevada", "Nevada", "Nevada", "Nevada"])):
            (tmp / name).mkdir()
            with open(tmp / name / "predictions.jsonl", "w", encoding="utf-8") as f:
                for i, (g, p) in enumerate(zip(gold, pred)):
                    f.write(json.dumps({"idx": i, "gold": g, "pred": p,
                                        "correct": t2.is_correct(g, p)}) + "\n")
        out = tmp / "cmp.json"
        subprocess.run([sys.executable, str(SCRIPTS / "legal_model_extension.py"), "compare",
                        "--task", "2", "--a", str(tmp / "a"), "--b", str(tmp / "b"),
                        "--out", str(out), "--n_boot", "100"],
                       check=True, capture_output=True)
        result = json.loads(out.read_text(encoding="utf-8"))
    assert result["mcnemar"]["only_b_correct"] == 2 and result["mcnemar"]["only_a_correct"] == 0
    assert result["headline_f1_a"] == 0.5 and result["headline_f1_b"] == 1.0


# ----------------------------------------------------------------------------- Task 3

T3_LABELS = ROOT / "data" / "LEDGAR" / "labels.json"
T3_VAL = ROOT / "ledgar" / "validation" / "ledgar_task3_validation.jsonl"
T3_RUNS = [
    ROOT / "kaggle_output_task3_fine_tuned" / "eval_metrics.json",
    ROOT / "kaggle_output_task3_baseline" / "eval_metrics.json",
]
t3.load_labels(T3_LABELS)


def test_t3_prompt_and_instruction_match_data():
    rows = [json.loads(line) for line in T3_VAL.read_text(encoding="utf-8").splitlines() if line]
    assert len(t3.LABELS) == 100
    assert {r["instruction"] for r in rows} == {t3.instruction()}
    assert all(r["output"] == r["category"] in t3.LABELS for r in rows)
    assert t3.build_prompt("I", "x") == t1.build_prompt("I", "x")


def test_t3_parse_matches_notebook():
    assert t3.parse_prediction("Governing Laws") == ("Governing Laws", True)
    assert t3.parse_prediction(" governing laws \nmore text") == ("Governing Laws", True)
    assert t3.parse_prediction("GOVERNING LAWS") == ("Governing Laws", True)
    # Run-on first line: invalid by the notebook rule, leniently valid.
    assert t3.parse_prediction("Governing Laws. This Agreement") == (t3.INVALID, True)
    assert t3.parse_prediction("Governing Law") == (t3.INVALID, False)
    assert t3.parse_prediction("") == (t3.INVALID, False)
    assert t3.parse_prediction("\nGoverning Laws") == ("Governing Laws", True)  # leading strip


def test_t3_invalid_is_a_miss_not_a_false_positive():
    gold = ["Taxes", "Taxes", "Notices"]
    metrics, report = t3.score(gold, ["Taxes", t3.INVALID, "Taxes"], [True, False, True])
    assert metrics["valid_label_rate"] == 2 / 3 and metrics["accuracy"] == 1 / 3
    assert metrics["per_label"]["Taxes"]["precision"] == 0.5     # INVALID adds no FP
    assert metrics["per_label"]["Taxes"]["recall"] == 0.5
    # Macro averages over all 100 labels: Taxes F1 0.5, everything else 0.
    assert abs(metrics["macro_f1"] - 0.5 / 100) < 1e-12
    assert t3.headline_f1(gold, ["Taxes", t3.INVALID, "Taxes"]) == metrics["macro_f1"]
    assert "Lenient valid-label rate" in report


def _t3_pairs_from_saved(saved):
    """Rebuild (gold, pred) pairs with the saved per-label TP / FP / FN and invalid count.
    FP of a label with TP = 0 is not recoverable (precision 0 either way); those labels get the
    remaining FPs, which leaves every saved metric unchanged (their precision and F1 stay 0)."""
    n = saved["n_validation_examples"]
    n_valid = round(saved["valid_label_rate"] * n)
    tp, fp, fn = {}, {}, {}
    for label, m in saved["per_label"].items():
        tp[label] = round(m["recall"] * m["support"])
        fn[label] = m["support"] - tp[label]
        fp[label] = round(tp[label] / m["precision"]) - tp[label] if tp[label] else 0
    spare = n_valid - sum(tp.values()) - sum(fp.values())
    zero_tp = [label for label in t3.LABELS if tp[label] == 0]
    assert spare >= 0 and (spare == 0 or zero_tp)
    for k in range(spare):
        fp[zero_tp[k % len(zero_tp)]] += 1
    gold = [l for l in t3.LABELS for _ in range(tp[l])]
    pred = list(gold)
    wrong_gold = [l for l in t3.LABELS for _ in range(fn[l])]
    wrong_pred = [l for l in t3.LABELS for _ in range(fp[l])] + [t3.INVALID] * (n - n_valid)
    assert len(wrong_gold) == len(wrong_pred)
    # Pair errors so no pair accidentally agrees: shift by the largest label multiplicity.
    shift = max(Counter(wrong_gold).values(), default=0)
    wrong_pred = sorted(wrong_pred, key=lambda p: (p == t3.INVALID, p))
    wrong_pred = wrong_pred[shift:] + wrong_pred[:shift]
    for i in range(len(wrong_gold)):            # repair any residual collisions by swapping
        if wrong_gold[i] == wrong_pred[i]:
            j = next(j for j in range(len(wrong_gold)) if wrong_pred[j] != wrong_gold[i]
                     and wrong_pred[i] != wrong_gold[j])
            wrong_pred[i], wrong_pred[j] = wrong_pred[j], wrong_pred[i]
    return gold + wrong_gold, pred + wrong_pred


def test_t3_reproduces_saved_kaggle_metrics():
    for path in T3_RUNS:
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        gold, pred = _t3_pairs_from_saved(saved)
        metrics, _ = t3.score(gold, pred, [p != t3.INVALID for p in pred])
        for key in ("n_validation_examples", "valid_label_rate", "accuracy", "macro_f1", "micro_f1"):
            assert metrics[key] == saved[key], (path, key, metrics[key], saved[key])
        # Per-label F1 can differ in the last bit: newer sklearn computes 2TP/(2TP+FP+FN)
        # where Kaggle's computed 2PR/(P+R).
        for label, m in saved["per_label"].items():
            ours = metrics["per_label"][label]
            assert ours["support"] == m["support"], (path, label)
            assert all(abs(ours[k] - m[k]) < 1e-12 for k in ("precision", "recall", "f1")), \
                (path, label, ours, m)
        print(f"  reproduced {path.relative_to(ROOT)}: macro-F1 {metrics['macro_f1']:.4f}")


def test_compare_cli_task3():
    gold = ["Taxes", "Notices", "Taxes", "Notices"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, pred in (("a", ["Taxes", t3.INVALID, "Notices", "Notices"]),
                           ("b", gold)):
            (tmp / name).mkdir()
            with open(tmp / name / "predictions.jsonl", "w", encoding="utf-8") as f:
                for i, (g, p) in enumerate(zip(gold, pred)):
                    f.write(json.dumps({"idx": i, "gold": g, "pred": p,
                                        "correct": t3.is_correct(g, p)}) + "\n")
        out = tmp / "cmp.json"
        subprocess.run([sys.executable, str(SCRIPTS / "legal_model_extension.py"), "compare",
                        "--task", "3", "--labels_file", str(T3_LABELS),
                        "--a", str(tmp / "a"), "--b", str(tmp / "b"),
                        "--out", str(out), "--n_boot", "50"],
                       check=True, capture_output=True)
        result = json.loads(out.read_text(encoding="utf-8"))
    assert result["mcnemar"]["only_b_correct"] == 2 and result["mcnemar"]["only_a_correct"] == 0
    assert abs(result["headline_f1_b"] - 2 / 100) < 1e-12   # 2 perfect labels of 100


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\nAll {len(tests)} tests passed.")
