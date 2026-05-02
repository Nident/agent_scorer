from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_responses import metrics_from_pairs


DEFAULT_INPUT = ROOT / "data" / "responses" / "benchmark" / "candidate_adjudicated_anli_all_50_metrics.json"
DEFAULT_OUTPUT = ROOT / "data" / "responses" / "benchmark" / "candidate_policy_hybrid_50_metrics.json"


def choose_with_policy(decision: dict[str, Any], present_vote_threshold: int) -> tuple[str, str]:
    candidates = decision.get("candidates", {})
    chosen = decision.get("chosen", "absent")

    present_votes = sum(label == "present" for label in candidates.values())
    if present_vote_threshold and present_votes >= present_vote_threshold and candidates.get("points_three") != "opposite":
        return "present", f"present_vote_guard_{present_vote_threshold}"

    return chosen, decision.get("source", "adjudicator")


def apply_policy(input_path: Path, present_vote_threshold: int) -> dict[str, Any]:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    y_true: list[str] = []
    y_pred: list[str] = []
    decisions: list[dict[str, Any]] = []

    for decision in report.get("decisions", []):
        chosen, source = choose_with_policy(decision, present_vote_threshold)
        y_true.append(decision["gold"])
        y_pred.append(chosen)
        decisions.append(
            {
                **decision,
                "policy_chosen": chosen,
                "policy_source": source,
            }
        )

    return {
        "source_report": str(input_path),
        "policy": {
            "present_vote_threshold": present_vote_threshold,
            "present_vote_guard": (
                "Choose present when at least this many candidate models choose present "
                "and points_three does not choose opposite."
            ),
        },
        "total": len(decisions),
        "metrics": metrics_from_pairs(y_true, y_pred),
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--present-vote-threshold", type=int, default=2)
    args = parser.parse_args()

    report = apply_policy(Path(args.input), args.present_vote_threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {key: value for key, value in report.items() if key != "decisions"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
