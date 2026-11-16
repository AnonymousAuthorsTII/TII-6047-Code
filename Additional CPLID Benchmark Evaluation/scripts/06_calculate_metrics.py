"""计算二分类测试指标。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS = ("Normal", "Major")


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    parser = argparse.ArgumentParser(description="计算绝缘子二分类指标")
    parser.add_argument(
        "--model", default="paper_gllm_insulator_60_language_e300:latest"
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "predictions",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(result_dir.glob("test_*.json"))
    ]
    if not records:
        raise FileNotFoundError(f"没有测试结果：{result_dir}")

    confusion = {
        true: {pred: 0 for pred in (*LABELS, "Unparsed")} for true in LABELS
    }
    for record in records:
        pred = record.get("answer") or "Unparsed"
        confusion[record["true_grade"]][pred] += 1

    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fn = sum(confusion[label].values()) - tp
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "support": sum(confusion[label].values()),
        }

    total = len(records)
    correct = sum(confusion[label][label] for label in LABELS)
    macro_precision = sum(per_class[x]["precision"] for x in LABELS) / 2
    macro_recall = sum(per_class[x]["recall"] for x in LABELS) / 2
    macro_f1 = sum(per_class[x]["f1"] for x in LABELS) / 2
    over_grading = safe_div(confusion["Normal"]["Major"], total)
    under_grading = safe_div(confusion["Major"]["Normal"], total)
    nmae = (
        safe_div(confusion["Normal"]["Major"], sum(confusion["Normal"].values()))
        + safe_div(confusion["Major"]["Normal"], sum(confusion["Major"].values()))
    ) / 2
    metrics = {
        "model": args.model,
        "total": total,
        "correct": correct,
        "unparsed": sum(confusion[label]["Unparsed"] for label in LABELS),
        "accuracy": safe_div(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "balanced_accuracy": macro_recall,
        "macro_f1": macro_f1,
        "highest_severity_recall": per_class["Major"]["recall"],
        "over_grading_rate": over_grading,
        "under_grading_rate": under_grading,
        "nmae": nmae,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
    output = args.output or (PROJECT_DIR / "results" / "metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("dataset", "total_samples", total, "count"),
        ("dataset", "correct_samples", correct, "count"),
        ("dataset", "unparsed_samples", metrics["unparsed"], "count"),
        ("overall", "accuracy", metrics["accuracy"] * 100, "percent"),
        ("overall", "macro_precision", macro_precision * 100, "percent"),
        ("overall", "macro_recall", macro_recall * 100, "percent"),
        ("overall", "macro_f1", macro_f1 * 100, "percent"),
        ("severity", "major_recall", per_class["Major"]["recall"] * 100, "percent"),
        ("severity", "over_grading_rate", over_grading * 100, "percent"),
        ("severity", "under_grading_rate", under_grading * 100, "percent"),
        ("severity", "nmae", nmae * 100, "percent"),
    ]
    for label in LABELS:
        rows.extend(
            [
                (label, "precision", per_class[label]["precision"] * 100, "percent"),
                (label, "recall", per_class[label]["recall"] * 100, "percent"),
                (label, "f1", per_class[label]["f1"] * 100, "percent"),
                (label, "support", per_class[label]["support"], "count"),
            ]
        )
    rows.extend(
        [
            ("confusion_matrix", "actual_Normal_pred_Normal", confusion["Normal"]["Normal"], "count"),
            ("confusion_matrix", "actual_Normal_pred_Major", confusion["Normal"]["Major"], "count"),
            ("confusion_matrix", "actual_Major_pred_Normal", confusion["Major"]["Normal"], "count"),
            ("confusion_matrix", "actual_Major_pred_Major", confusion["Major"]["Major"], "count"),
        ]
    )
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scope", "metric", "value", "unit"))
        for scope, metric, value, unit in rows:
            writer.writerow((scope, metric, f"{value:.6f}" if isinstance(value, float) else value, unit))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
