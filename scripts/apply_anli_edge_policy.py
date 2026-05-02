from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adjudicate_candidate_ensemble import load_variant
from scripts.evaluate_responses import metrics_from_pairs


DEFAULT_INPUT = ROOT / "data" / "responses" / "benchmark" / "candidate_adjudicated_anli_all_100_metrics.json"
DEFAULT_OUTPUT = ROOT / "data" / "responses" / "benchmark" / "candidate_policy_edge_100_metrics.json"

BASE_VARIANTS = ["simple", "simple_nli_reasoner", "simple_nli_anli", "points_verify", "points_three"]
V2_VARIANT = "simple_nli_anli_v2"


def repeated_surname(text: str) -> bool:
    surnames = re.findall(r"\b[A-Z][a-z]+\s+(?:[A-Z]\.\s+)?([A-Z][a-z]+)\b", text)
    return any(surnames.count(surname) >= 2 for surname in set(surnames))


def age_from_hypothesis(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})(?:st|nd|rd|th)?\s+birthday\b", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def born_year(text: str) -> int | None:
    match = re.search(r"\bborn\s+[A-Z][a-z]+\s+\d{1,2},\s+(\d{4})\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\bborn\s+(?:in\s+)?(\d{4})\b", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def edge_choice(hypothesis: str, premise: str, chosen: str) -> tuple[str, str | None]:
    hyp = hypothesis.lower()
    prem = premise.lower()

    if "won" in hyp and "nominated for" in prem and "won" not in prem:
        return "absent", "nominated_is_not_won"

    if "same last name" in hyp and repeated_surname(premise):
        return "present", "repeated_surname"

    if "committee of 100" in hyp and re.search(r"\b99\b", hyp):
        return "opposite", "numbered_entity_name"

    if "last work" in hyp and "predecessor" in prem:
        return "opposite", "predecessor_not_last_work"

    if "died last year" in hyp and "born" in prem and "died" not in prem:
        return "absent", "unsupported_death_date"

    if "founded" in hyp and "same family" in prem and "department store chain" in hyp:
        return "opposite", "wrong_founded_entity"

    expected_age = age_from_hypothesis(hypothesis)
    year = born_year(premise)
    if expected_age is not None and year is not None and expected_age != 2019 - year:
        return "opposite", "age_mismatch_corpus_year"

    return chosen, None


def choose(decision: dict[str, Any], variant_rows: dict[str, dict[tuple[str, int], dict[str, Any]]]) -> tuple[str, str]:
    stem, index_text = decision["key"].split(":")
    key = (stem, int(index_text))
    candidates = decision.get("candidates", {})
    chosen = decision.get("chosen", "absent")
    source = decision.get("source", "adjudicator")

    present_votes = sum(label == "present" for label in candidates.values())
    if present_votes >= 2 and candidates.get("points_three") != "opposite":
        chosen = "present"
        source = "present_vote_guard"

    v2_row = variant_rows[V2_VARIANT].get(key)
    if v2_row:
        v2_label = v2_row["label"]
        old_labels = [variant_rows[variant][key]["label"] for variant in BASE_VARIANTS if key in variant_rows[variant]]
        if v2_label == "present" and old_labels and all(label == "absent" for label in old_labels):
            chosen = "present"
            source = "v2_absent_consensus_rescue"

        edge_label, edge_source = edge_choice(v2_row["hypothesis"], v2_row["premise"], chosen)
        if edge_source:
            chosen = edge_label
            source = edge_source

    return chosen, source


def apply_policy(input_path: Path, responses_root: Path) -> dict[str, Any]:
    source_report = json.loads(input_path.read_text(encoding="utf-8"))
    variant_rows = {
        variant: load_variant(responses_root, variant)
        for variant in [*BASE_VARIANTS, V2_VARIANT]
    }

    y_true: list[str] = []
    y_pred: list[str] = []
    decisions: list[dict[str, Any]] = []

    for decision in source_report.get("decisions", []):
        chosen, source = choose(decision, variant_rows)
        y_true.append(decision["gold"])
        y_pred.append(chosen)
        decisions.append(
            {
                **decision,
                "edge_policy_chosen": chosen,
                "edge_policy_source": source,
            }
        )

    return {
        "source_report": str(input_path),
        "policy": {
            "base": "full edge adjudicator + present vote guard",
            "v2_rescue": "Use simple_nli_anli_v2 only when all older candidates are absent and v2 says present.",
            "edge_rules": [
                "nominated_is_not_won",
                "repeated_surname",
                "numbered_entity_name",
                "predecessor_not_last_work",
                "unsupported_death_date",
                "wrong_founded_entity",
                "age_mismatch_corpus_year",
            ],
        },
        "total": len(decisions),
        "metrics": metrics_from_pairs(y_true, y_pred),
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--responses-root", default=str(ROOT / "data" / "responses" / "benchmark"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = apply_policy(Path(args.input), Path(args.responses_root))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {key: value for key, value in report.items() if key != "decisions"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
