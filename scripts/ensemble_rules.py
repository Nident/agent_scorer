from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_responses import metrics_from_pairs, normalize_label, prediction_items


def load_variant(root: Path, variant: str) -> dict[tuple[str, int], tuple[str, str]]:
    model_type = "simple" if variant == "simple" else "points"
    rows = {}
    for resp_path in sorted((root / variant).glob("*.json")):
        resp = json.loads(resp_path.read_text(encoding="utf-8"))
        gt_path = Path(resp["metadata"]["criterion_path"])
        gt = yaml.safe_load(gt_path.read_text(encoding="utf-8")) or {}
        preds = prediction_items(resp, model_type)
        for index, (gold, prediction) in enumerate(zip(gt.get("gold", []), preds)):
            try:
                rows[(resp_path.stem, index)] = (
                    normalize_label(gold["label"]),
                    normalize_label(prediction),
                )
            except Exception:
                continue
    return rows


def present_boost(preds: dict[str, str]) -> str:
    if preds["points_three"] == "opposite":
        return "opposite"
    if preds["simple"] == "present":
        return "present"
    if preds["points_three"] == "present":
        return "present"
    if preds["points_verify"] == "opposite" and preds["simple"] != "absent":
        return "opposite"
    return preds["points_verify"]


def evaluate_rule(responses_root: Path) -> dict:
    variants = ["simple", "points_verify", "points_three"]
    records = {variant: load_variant(responses_root, variant) for variant in variants}
    common = sorted(set.intersection(*(set(records[variant]) for variant in variants)))

    y_true = []
    y_pred = []
    decisions = []
    for key in common:
        gold = records["simple"][key][0]
        preds = {variant: records[variant][key][1] for variant in variants}
        chosen = present_boost(preds)
        y_true.append(gold)
        y_pred.append(chosen)
        decisions.append({"key": f"{key[0]}:{key[1]}", "gold": gold, "predictions": preds, "chosen": chosen})

    return {
        "rule": "present_boost",
        "total": len(common),
        "metrics": metrics_from_pairs(y_true, y_pred),
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-root", default=str(ROOT / "data" / "responses" / "benchmark"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "responses" / "benchmark" / "ensemble_present_boost_metrics.json"),
    )
    args = parser.parse_args()

    report = evaluate_rule(Path(args.responses_root))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {key: value for key, value in report.items() if key != "decisions"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
