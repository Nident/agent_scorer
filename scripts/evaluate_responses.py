from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


LABELS = ["opposite", "absent", "present"]
INT_TO_LABEL = {-1: "opposite", 0: "absent", 1: "present"}
GOLD_TO_LABEL = {
    "contradiction": "opposite",
    "opposite": "opposite",
    "neutral": "absent",
    "absent": "absent",
    "entailment": "present",
    "present": "present",
}


def normalize_label(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Unknown label: {value!r}")
    if isinstance(value, (int, float)):
        label = int(value)
        if label in INT_TO_LABEL:
            return INT_TO_LABEL[label]

    normalized = str(value).strip().lower()
    normalized = normalized.replace("label:", "").strip()
    normalized = normalized.split()[0] if normalized else normalized
    aliases = {
        "present": "present",
        "entailment": "present",
        "entailed": "present",
        "true": "present",
        "yes": "present",
        "1": "present",
        "+1": "present",
        "absent": "absent",
        "neutral": "absent",
        "unknown": "absent",
        "not_present": "absent",
        "not-present": "absent",
        "no": "absent",
        "0": "absent",
        "opposite": "opposite",
        "contradiction": "opposite",
        "contradictory": "opposite",
        "false": "opposite",
        "-1": "opposite",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown label: {value!r}")
    return aliases[normalized]


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def final_label_from_point(point: dict[str, Any]) -> Any:
    final = parse_jsonish(point.get("final"))
    if isinstance(final, dict):
        if "label" in final:
            return final["label"]
        if "score" in final:
            return final["score"]
        if "raw_response" in final:
            nested = parse_jsonish(final["raw_response"])
            if isinstance(nested, dict):
                return nested.get("label", nested.get("score"))
    return None


def simple_points(resp: dict[str, Any]) -> list[dict[str, Any]]:
    final = parse_jsonish(resp.get("final"))
    if isinstance(final, dict) and "raw_response" in final:
        final = parse_jsonish(final["raw_response"])
    if isinstance(final, dict) and isinstance(final.get("points"), list):
        return final["points"]
    return []


def prediction_items(resp: dict[str, Any], model_type: str) -> list[Any]:
    if model_type == "simple":
        return [point.get("score") for point in simple_points(resp) if isinstance(point, dict)]
    return [final_label_from_point(point) for point in resp.get("results", []) if isinstance(point, dict)]


def metrics_from_pairs(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    total = len(y_true)
    confusion_matrix = {true: {pred: 0 for pred in LABELS} for true in LABELS}
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label][pred_label] += 1

    per_class = {}
    for label in LABELS:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[other][label] for other in LABELS if other != label)
        fn = sum(confusion_matrix[label][other] for other in LABELS if other != label)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": sum(confusion_matrix[label].values()),
        }

    accuracy = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label) / total if total else 0.0
    macro_precision = sum(value["precision"] for value in per_class.values()) / len(LABELS)
    macro_recall = sum(value["recall"] for value in per_class.values()) / len(LABELS)
    macro_f1 = sum(value["f1_score"] for value in per_class.values()) / len(LABELS)
    weighted_f1 = sum(value["f1_score"] * value["support"] for value in per_class.values()) / total if total else 0.0

    return {
        "total": total,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "weighted_f1_score": weighted_f1,
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
    }


def evaluate_file(resp_path: Path, model_type: str) -> dict[str, Any]:
    resp = json.loads(resp_path.read_text(encoding="utf-8"))
    gt_path = Path(resp["metadata"]["criterion_path"])
    gt_data = yaml.safe_load(gt_path.read_text(encoding="utf-8")) or {}
    gold = gt_data.get("gold", [])
    predictions = prediction_items(resp, model_type)

    y_true = []
    y_pred = []
    label_errors = []
    for idx, (gt, prediction) in enumerate(zip(gold, predictions)):
        try:
            y_true.append(normalize_label(gt["label"]))
            y_pred.append(normalize_label(prediction))
        except ValueError as exc:
            label_errors.append({"idx": idx, "error": str(exc), "prediction": prediction, "gold": gt})

    metrics = metrics_from_pairs(y_true, y_pred)
    metrics.update(
        {
            "gt_count": len(gold),
            "pred_count": len(predictions),
            "paired_count": min(len(gold), len(predictions)),
            "length_mismatch": len(gold) != len(predictions),
            "missing_predictions": max(len(gold) - len(predictions), 0),
            "extra_predictions": max(len(predictions) - len(gold), 0),
            "label_error_count": len(label_errors),
            "label_errors": label_errors,
        }
    )
    return {
        "response_file": resp_path.name,
        "criterion_file": gt_path.name,
        "metrics": metrics,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = []
    y_pred = []
    for item in results:
        matrix = item["metrics"]["confusion_matrix"]
        for true_label in LABELS:
            for pred_label in LABELS:
                count = matrix[true_label][pred_label]
                y_true.extend([true_label] * count)
                y_pred.extend([pred_label] * count)
    return metrics_from_pairs(y_true, y_pred)


def evaluate_dir(responses_dir: Path, model_type: str) -> dict[str, Any]:
    results = []
    errors = []
    for resp_path in sorted(responses_dir.glob("*.json")):
        try:
            results.append(evaluate_file(resp_path, model_type))
        except Exception as exc:
            errors.append({"response_file": resp_path.name, "error": str(exc)})

    problem_files = [
        item for item in results
        if item["metrics"]["length_mismatch"] or item["metrics"]["label_error_count"]
    ]
    return {
        "model_type": model_type,
        "responses_dir": str(responses_dir),
        "files": len(results),
        "errors": errors,
        "problem_summary": {
            "files_with_problems": len(problem_files),
            "files_with_length_mismatch": sum(item["metrics"]["length_mismatch"] for item in results),
            "files_with_label_errors": sum(item["metrics"]["label_error_count"] > 0 for item in results),
            "missing_predictions": sum(item["metrics"]["missing_predictions"] for item in results),
            "extra_predictions": sum(item["metrics"]["extra_predictions"] for item in results),
            "label_errors": sum(item["metrics"]["label_error_count"] for item in results),
        },
        "overall": aggregate(results) if results else None,
        "files_detail": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses_dir")
    parser.add_argument("--model-type", choices=["simple", "points"], required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = evaluate_dir(Path(args.responses_dir), args.model_type)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
