from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "evaluation_ANLI" / "data" / "anli_dev_r1.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation_ANLI" / "generated"


LABEL_ID_TO_NAME = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def safe_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return value[:96] or "item"


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_premise(row: dict[str, Any]) -> str:
    utterances = row.get("transcript", {}).get("utterances", [])
    return str(utterances[0].get("text", "")) if utterances else ""


def row_hypothesis(row: dict[str, Any]) -> str:
    utterances = row.get("transcript", {}).get("utterances", [])
    return str(utterances[1].get("text", "")) if len(utterances) > 1 else ""


def row_label(row: dict[str, Any]) -> str:
    anli = row.get("anli", {})
    if "label_name" in anli:
        return str(anli["label_name"])
    if "label" in anli:
        return LABEL_ID_TO_NAME[int(anli["label"])]
    if "gold_label" in row:
        return str(row["gold_label"])
    raise ValueError(f"Missing label in row {row.get('id')}")


def group_rows(rows: list[dict[str, Any]], max_groups: int | None) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        premise = row_premise(row)
        if premise not in grouped and max_groups is not None and len(grouped) >= max_groups:
            continue
        grouped.setdefault(premise, []).append(row)

    items = list(grouped.items())
    if max_groups is not None:
        items = items[:max_groups]
    return items


def write_benchmark(input_path: Path, output_dir: Path, limit: int | None, overwrite: bool) -> dict[str, Any]:
    rows = read_rows(input_path)
    grouped = group_rows(rows, limit)

    dialogues_dir = output_dir / "dialogues"
    criteria_dir = output_dir / "criteria"
    index_path = output_dir / "index.jsonl"

    dialogues_dir.mkdir(parents=True, exist_ok=True)
    criteria_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for premise_index, (premise, group) in enumerate(grouped, start=1):
        first_id = str(group[0].get("id", premise_index))
        record_id = f"premise_{premise_index:06d}_{safe_filename(first_id)}"
        created_at = datetime.now(timezone.utc).isoformat()

        dialogue_path = dialogues_dir / f"{record_id}.json"
        criterion_path = criteria_dir / f"{record_id}.yaml"

        dialogue = {
            "schema_version": "transcript_json_v1",
            "created_at": created_at,
            "transcript": {
                "source_format": "speaker_turns",
                "source_file": str(dialogue_path.relative_to(ROOT)),
                "source_dataset": "local_anli_jsonl",
                "language": "en",
                "turn_count": 1,
                "speakers": [{"speaker_label": "B", "role": "evidence", "utterance_count": 1}],
                "utterances": [
                    {
                        "timestamp": "00:00",
                        "speaker_label": "B",
                        "role": "evidence",
                        "text": premise,
                        "turn_index": 1,
                    }
                ],
            },
        }

        gold_items = [
            {
                "uid": str(row.get("id", "")),
                "label": row_label(row),
                "hypothesis": row_hypothesis(row),
                "reason": str(row.get("anli", {}).get("reason", "")),
            }
            for row in group
        ]
        criterion = {
            "name": f"ANLI hypotheses for {record_id}",
            "premise_id": record_id,
            "points": [item["hypothesis"] for item in gold_items],
            "gold": gold_items,
        }

        dialogue_path.write_text(json.dumps(dialogue, ensure_ascii=False, indent=2), encoding="utf-8")
        criterion_path.write_text(
            yaml.safe_dump(criterion, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        created.append(
            {
                "premise_id": record_id,
                "dialogue_path": str(dialogue_path.relative_to(ROOT)),
                "criterion_path": str(criterion_path.relative_to(ROOT)),
                "hypothesis_count": len(gold_items),
            }
        )

    with index_path.open("w", encoding="utf-8") as file:
        for item in created:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "groups": len(created),
        "hypotheses": sum(item["hypothesis_count"] for item in created),
        "dialogues_dir": str(dialogues_dir),
        "criteria_dir": str(criteria_dir),
        "index_path": str(index_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = write_benchmark(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
