"""Merge every legal-model-extension result into two flat CSVs for the analysis notebook.

Inputs (read-only):
  <runs_dir>/<model>_t<task>_<mode>[_adapterN]/eval_metrics.json (+ train_metrics.json)
      one per harness arm; `*_smoke` runs are skipped.
  <runs_dir>/compare_t<task>_<a>_vs_<b>.json
      paired tests written by `legal_model_extension.py compare`.
  The original Llama zero-shot notebooks' eval_metrics.json (kaggle_output_task*_baseline/), because
      the Llama baselines were never re-run through the harness. They are bf16 and have no
      per-example predictions, so they appear in the arms table but in no paired test.

Outputs (written to <out_dir>):
  extension_arms.csv         one row per (task, model, arm): validity gate, correctness metric,
                             headline F1, run times, source.
  extension_comparisons.csv  one row per paired test: discordant counts, McNemar p, headline-F1
                             difference B-A with its bootstrap 95% CI.

Metrics per task (same as `compare`): correctness = T1 accuracy / T2 exact match / T3 accuracy;
headline F1 = T1 macro-F1 / T2 mean token F1 / T3 macro-F1; validity = T1 strict Yes/No rate /
T2 strict JSON-valid / T3 valid-label rate.

Usage (repo root):
    python scripts/collect_extension_results.py [--runs_dir kaggle_output_extension/runs]
                                                [--out_dir kaggle_output_extension]
"""
import argparse
import csv
import json
import re
from pathlib import Path

LLAMA_NOTEBOOK_BASELINES = {
    1: "kaggle_output_task1_baseline/no_finetune_baseline/eval_metrics.json",
    2: "kaggle_output_task2_baseline/no_finetune_baseline_task2/eval_metrics.json",
    3: "kaggle_output_task3_baseline/eval_metrics.json",
}
RUN_NAME = re.compile(r"^(?P<model>[a-z]+)_t(?P<task>\d)_(?P<mode>baseline|finetune|adapter)(_adapter\d)?$")
COMPARE_NAME = re.compile(r"^compare_t(?P<task>\d)_(?P<a>.+)_vs_(?P<b>.+)\.json$")
ARM = {"baseline": "zero-shot", "finetune": "qlora", "adapter": "qlora"}


def task_metrics(task, m):
    """(validity, correctness, headline_f1, secondary) from one eval_metrics.json, for either schema."""
    if task == 1:
        report = m["classification_report"]
        return (m.get("strict_valid_rate"), m["accuracy"], report["macro avg"]["f1-score"],
                report["Yes"]["f1-score"])
    if task == 2:
        overall = m["overall"]
        return overall["json_valid"], overall["exact_match"], overall["f1"], m.get("lenient_json_valid_rate")
    return m["valid_label_rate"], m["accuracy"], m["macro_f1"], m["micro_f1"]


SECONDARY_NAME = {1: "yes_f1", 2: "lenient_json_valid", 3: "micro_f1"}


def arm_row(task, model, arm, run, source, metrics, train=None):
    validity, correct, f1, secondary = task_metrics(task, metrics)
    return {
        "task": task, "model": model, "arm": arm, "run": run, "source": source,
        "n": metrics["n_validation_examples"], "validity": validity, "correctness": correct,
        "headline_f1": f1, "secondary_name": SECONDARY_NAME[task], "secondary": secondary,
        "train_hours": round(train["train_runtime_seconds"] / 3600, 3) if train else None,
        "stopped_on_time_budget": train["stopped_on_time_budget"] if train else None,
        "eval_seconds": metrics.get("eval_runtime_seconds"),
    }


def collect_arms(runs_dir, repo_root):
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/eval_metrics.json")):
        match = RUN_NAME.match(metrics_path.parent.name)
        if not match:   # smoke runs and anything unexpected
            continue
        task, mode = int(match["task"]), match["mode"]
        train_path = metrics_path.parent / "train_metrics.json"
        train = json.loads(train_path.read_text(encoding="utf-8")) if train_path.exists() else None
        source = "harness fp16" + (" (Llama adapter re-scored)" if mode == "adapter" else "")
        rows.append(arm_row(task, match["model"], ARM[mode], metrics_path.parent.name, source,
                            json.loads(metrics_path.read_text(encoding="utf-8")), train))
    for task, rel in LLAMA_NOTEBOOK_BASELINES.items():
        path = repo_root / rel
        if path.exists():
            rows.append(arm_row(task, "llama", "zero-shot", rel, "original notebook bf16",
                                json.loads(path.read_text(encoding="utf-8"))))
    return sorted(rows, key=lambda r: (r["task"], r["arm"], r["model"]))


def collect_comparisons(runs_dir):
    rows = []
    for path in sorted(runs_dir.glob("compare_*.json")):
        match = COMPARE_NAME.match(path.name)
        if not match:
            continue
        c = json.loads(path.read_text(encoding="utf-8"))
        low, high = c["headline_f1_diff_b_minus_a_ci95"]
        rows.append({
            "task": c["task"], "a": Path(c["a"]).name, "b": Path(c["b"]).name, "n": c["n"],
            "correctness_a": c["accuracy_a"], "correctness_b": c["accuracy_b"],
            "only_a_correct": c["mcnemar"]["only_a_correct"], "only_b_correct": c["mcnemar"]["only_b_correct"],
            "mcnemar_p": c["mcnemar"]["p_value"],
            "headline_f1_a": c["headline_f1_a"], "headline_f1_b": c["headline_f1_b"],
            "f1_diff_b_minus_a": c["headline_f1_b"] - c["headline_f1_a"], "f1_diff_ci_low": low,
            "f1_diff_ci_high": high, "n_boot": c["n_boot"], "seed": c["seed"],
        })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", default="kaggle_output_extension/runs")
    parser.add_argument("--out_dir", default="kaggle_output_extension")
    parser.add_argument("--repo_root", default=".", help="where the original kaggle_output_task*_baseline/ live")
    args = parser.parse_args()
    runs_dir, out_dir = Path(args.runs_dir), Path(args.out_dir)
    write_csv(out_dir / "extension_arms.csv", collect_arms(runs_dir, Path(args.repo_root)))
    write_csv(out_dir / "extension_comparisons.csv", collect_comparisons(runs_dir))


if __name__ == "__main__":
    main()
