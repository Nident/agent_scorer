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
from scripts.evaluate_responses import (
    metrics_from_pairs,
    normalize_label,
    parse_jsonish,
    prediction_items,
    simple_points,
)


DEFAULT_OUTPUT = ROOT / "data" / "responses" / "benchmark" / "candidate_adjudicated_metrics.json"

LABEL_TO_SCORE = {"opposite": -1, "absent": 0, "present": 1}

PROMPT = """You are an ANLI-style natural language inference adjudicator.

Classify the Hypothesis against the Premise.

Labels:
- present / 1: a careful ANLI reader would infer the Hypothesis from the Premise.
- absent / 0: the Hypothesis may be true, but the Premise does not support it enough.
- opposite / -1: an important part of the Hypothesis conflicts with the Premise.

Candidate model predictions are listed below. Re-read the Premise and Hypothesis yourself,
then choose the final label. Prefer one of the candidate labels when it is plausible.
Choose a non-candidate label only if all candidates miss a clear textual fact.

Important traps:
- ANLI is broader than formal logic. Use the label a careful dataset annotator would choose,
  not only what is mathematically guaranteed.
- Entailment allows paraphrases, coreference, loose wording, likely causal statements, counts,
  dates, simple arithmetic, geography, titles, roles, and ordinary world knowledge.
- If the Hypothesis says "may", "could", or expresses uncertainty, it can be present when the
  Premise states uncertainty, dispute, allegations, or possibility.
- "Known as" or "best known as" can support a natural claim about how people know that entity.
- "At least N" can be present when the Premise states an equal-or-larger value for the relevant
  object, site, altitude, size, count, or quantity.
- A work about a specific real event can support a broad causal hypothesis that the work depends
  on that event existing.
- Treat relative year claims using the implicit ANLI/Wikipedia corpus timeframe around 2019 when
  the Premise gives the base year.
- Contradiction requires an incompatible detail. An unsupported date, award, role, author, number,
  or event is absent, not opposite, unless the Premise gives a conflicting fact.
- "Nominated for an award" does not mean "won the award"; that is absent unless the Premise says
  someone else won or explicitly excludes winning.
- Turning a town/place into the wrong county/region, saying a predecessor is the last work, or
  saying an unreleased/upcoming work already came out can be opposite when the Premise conflicts.
- If a Hypothesis adds one unsupported extra claim to otherwise supported facts, use absent
  unless the extra claim conflicts with the Premise.
- Do not mark absent just because the exact words differ.

Premise:
{premise}

Hypothesis:
{hypothesis}

Candidate predictions:
{candidates_json}

Return JSON only:
{{
  "label": 0,
  "chosen_candidate": "variant name or none",
  "rationale": "one short reason",
  "confidence": 0.0
}}
"""


def variant_model_type(variant: str) -> str:
    return "simple" if variant.startswith("simple") else "points"


def parse_json_dict(value: Any) -> dict[str, Any]:
    parsed = parse_jsonish(value)
    return parsed if isinstance(parsed, dict) else {}


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
    points = simple_points(resp)
    if index < len(points) and isinstance(points[index], dict):
        point = dict(points[index])
        if "score" in point and "label" not in point:
            point["label"] = point["score"]
        return point
    return {}


def point_prediction_detail(resp: dict[str, Any], index: int) -> dict[str, Any]:
    results = resp.get("results", [])
    if index < len(results) and isinstance(results[index], dict):
        return parse_json_dict(results[index].get("final", {}))
    return {}


def load_variant(root: Path, variant: str) -> dict[tuple[str, int], dict[str, Any]]:
    model_type = variant_model_type(variant)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for resp_path in sorted((root / variant).glob("*.json")):
        resp = json.loads(resp_path.read_text(encoding="utf-8"))
        gt_path = Path(resp["metadata"]["criterion_path"])
        gt = yaml.safe_load(gt_path.read_text(encoding="utf-8")) or {}
        predictions = prediction_items(resp, model_type)

        for index, gold in enumerate(gt.get("gold", [])):
            if index >= len(predictions):
                continue
            try:
                label = normalize_label(predictions[index])
                gold_label = normalize_label(gold["label"])
            except Exception:
                continue
            detail = (
                simple_prediction_detail(resp, index)
                if model_type == "simple"
                else point_prediction_detail(resp, index)
            )
            rows[(resp_path.stem, index)] = {
                "file": resp_path.name,
                "gold": gold_label,
                "hypothesis": gold.get("hypothesis", ""),
                "premise": dialogue_text(resp),
                "label": label,
                "score": LABEL_TO_SCORE[label],
                "detail": detail,
            }
    return rows


def extract_json(value: str) -> dict[str, Any]:
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": value}
    return parsed if isinstance(parsed, dict) else {"raw_response": value}


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(args.model_env, override=True)
    llm = Model.load_model(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=os.getenv("API_KEY", ""),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1.0,
    )

    root = Path(args.responses_root)
    variants = args.variants
    records = {variant: load_variant(root, variant) for variant in variants}
    common = sorted(set.intersection(*(set(rows) for rows in records.values())))

    y_true: list[str] = []
    y_pred: list[str] = []
    decisions: list[dict[str, Any]] = []
    calls = 0

    for key in common:
        base = records[variants[0]][key]
        gold = base["gold"]
        candidates = {
            variant: {
                "label": records[variant][key]["label"],
                "score": records[variant][key]["score"],
                "detail": records[variant][key]["detail"],
            }
            for variant in variants
        }
        candidate_labels = {item["label"] for item in candidates.values()}

        if len(candidate_labels) == 1 and not args.adjudicate_agreements:
            chosen = next(iter(candidate_labels))
            source = "agreement"
            adjudication: dict[str, Any] = {}
        else:
            prompt = PROMPT.format(
                premise=base["premise"],
                hypothesis=base["hypothesis"],
                candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
            )
            raw = llm.invoke(prompt)
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            adjudication = extract_json(content)
            try:
                chosen = normalize_label(adjudication.get("label"))
            except Exception:
                chosen = records[args.fallback_variant][key]["label"]
            source = "adjudicator"
            calls += 1

        y_true.append(gold)
        y_pred.append(chosen)
        decisions.append(
            {
                "key": f"{key[0]}:{key[1]}",
                "gold": gold,
                "hypothesis": base["hypothesis"],
                "candidates": {variant: records[variant][key]["label"] for variant in variants},
                "chosen": chosen,
                "source": source,
                "adjudication": adjudication,
            }
        )

    return {
        "variants": variants,
        "fallback_variant": args.fallback_variant,
        "total": len(common),
        "adjudicator_calls": calls,
        "metrics": metrics_from_pairs(y_true, y_pred),
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-root", default=str(ROOT / "data" / "responses" / "benchmark"))
    parser.add_argument("--model-env", default=str(ROOT / "config" / "model.env"))
    parser.add_argument("--model-name", default="deepseek-reasoner")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["simple", "simple_nli_reasoner", "points_verify", "points_three"],
    )
    parser.add_argument("--fallback-variant", default="simple_nli_reasoner")
    parser.add_argument("--adjudicate-agreements", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = run(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {key: value for key, value in report.items() if key != "decisions"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
