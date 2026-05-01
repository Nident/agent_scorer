from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model import Model
from scripts.evaluate_responses import metrics_from_pairs, normalize_label, prediction_items

DEFAULT_OUTPUT = ROOT / "data" / "responses" / "benchmark" / "adjudicated_ensemble_metrics.json"


PROMPT = """You are a careful natural language inference adjudicator.

Classify the Hypothesis against the Premise.

Labels:
- 1 = entailment / present: the Hypothesis follows from the Premise.
- 0 = neutral / absent: the Hypothesis might be true, but is not guaranteed by the Premise.
- -1 = contradiction / opposite: an important part of the Hypothesis conflicts with the Premise.

Use ordinary language understanding, paraphrases, dates, arithmetic, geography, and common knowledge when appropriate.
Do not over-penalize paraphrases. If one unsupported extra claim is added, use neutral unless it conflicts.

Premise:
{premise}

Hypothesis:
{hypothesis}

Simple model prediction:
{simple_prediction}

Point-chain prediction:
{points_prediction}

Return JSON only:
{{
  "label": 0,
  "rationale": "short reason",
  "confidence": 0.0
}}
"""


def parse_jsonish(value: str) -> dict[str, Any]:
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": value}
    return parsed if isinstance(parsed, dict) else {"raw_response": value}


def dialogue_text(resp: dict[str, Any]) -> str:
    blocks = resp.get("blocks")
    if isinstance(blocks, list) and blocks:
        return "\n\n".join(str(block.get("dialogue_block", "")) for block in blocks if isinstance(block, dict))
    results = resp.get("results")
    if isinstance(results, list) and results:
        first_blocks = results[0].get("blocks", []) if isinstance(results[0], dict) else []
        return "\n\n".join(str(block.get("dialogue_block", "")) for block in first_blocks if isinstance(block, dict))
    return ""


def simple_prediction_detail(resp: dict[str, Any], index: int) -> dict[str, Any]:
    final = resp.get("final", {})
    if isinstance(final, dict) and "raw_response" in final:
        try:
            final = json.loads(final["raw_response"])
        except json.JSONDecodeError:
            return {"raw_response": final["raw_response"]}
    points = final.get("points", []) if isinstance(final, dict) else []
    if index < len(points) and isinstance(points[index], dict):
        return points[index]
    return {}


def point_prediction_detail(resp: dict[str, Any], index: int) -> dict[str, Any]:
    results = resp.get("results", [])
    if index < len(results) and isinstance(results[index], dict):
        return results[index].get("final", {})
    return {}


def load_pairs(simple_dir: Path, points_dir: Path):
    pairs = []
    for simple_path in sorted(simple_dir.glob("*.json")):
        points_path = points_dir / simple_path.name
        if not points_path.exists():
            continue

        simple_resp = json.loads(simple_path.read_text(encoding="utf-8"))
        points_resp = json.loads(points_path.read_text(encoding="utf-8"))
        gt_path = Path(simple_resp["metadata"]["criterion_path"])
        gt = yaml.safe_load(gt_path.read_text(encoding="utf-8")) or {}
        simple_preds = prediction_items(simple_resp, "simple")
        points_preds = prediction_items(points_resp, "points")

        for index, gold in enumerate(gt.get("gold", [])):
            if index >= len(simple_preds) or index >= len(points_preds):
                continue
            try:
                simple_label = normalize_label(simple_preds[index])
                points_label = normalize_label(points_preds[index])
                gold_label = normalize_label(gold["label"])
            except Exception:
                continue
            pairs.append(
                {
                    "key": f"{simple_path.stem}:{index}",
                    "gold": gold_label,
                    "hypothesis": gold.get("hypothesis", ""),
                    "premise": dialogue_text(simple_resp),
                    "simple_label": simple_label,
                    "points_label": points_label,
                    "simple_detail": simple_prediction_detail(simple_resp, index),
                    "points_detail": point_prediction_detail(points_resp, index),
                }
            )
    return pairs


def run_adjudication(args) -> dict[str, Any]:
    load_dotenv(args.model_env, override=True)
    llm = Model.load_model(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=os.getenv("API_KEY", ""),
        temperature=0.0,
        max_tokens=args.max_tokens,
        top_p=1.0,
    )

    pairs = load_pairs(Path(args.simple_dir), Path(args.points_dir))
    y_true = []
    y_pred = []
    decisions = []

    for item in pairs:
        y_true.append(item["gold"])
        if item["simple_label"] == item["points_label"]:
            chosen = item["points_label"]
            source = "agreement"
            adjudication = {}
        else:
            prompt = PROMPT.format(
                premise=item["premise"],
                hypothesis=item["hypothesis"],
                simple_prediction=json.dumps(item["simple_detail"], ensure_ascii=False),
                points_prediction=json.dumps(item["points_detail"], ensure_ascii=False),
            )
            raw = llm.invoke(prompt)
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            adjudication = parse_jsonish(content)
            try:
                chosen = normalize_label(adjudication.get("label"))
            except Exception:
                chosen = item["points_label"]
            source = "adjudicator"

        y_pred.append(chosen)
        decisions.append({**item, "chosen": chosen, "source": source, "adjudication": adjudication})

    metrics = metrics_from_pairs(y_true, y_pred)
    return {
        "total": len(pairs),
        "disagreements": sum(1 for item in pairs if item["simple_label"] != item["points_label"]),
        "metrics": metrics,
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple-dir", default=str(ROOT / "data" / "responses" / "benchmark" / "simple"))
    parser.add_argument("--points-dir", default=str(ROOT / "data" / "responses" / "benchmark" / "points_three"))
    parser.add_argument("--model-env", default=str(ROOT / "config" / "model.env"))
    parser.add_argument("--model-name", default="deepseek-reasoner")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = run_adjudication(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {key: value for key, value in report.items() if key != "decisions"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
