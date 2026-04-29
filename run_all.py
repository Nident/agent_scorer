from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "config" / ".env"
CRITERIA_DIR = ROOT / "evaluation_ANLI" / "generated" / "criteria"
DIALOGUES_DIR = ROOT / "evaluation_ANLI" / "generated" / "dialogues"
RESPONSES_DIR = ROOT / "data" / "responses" / "anli_all"


def update_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    seen = set()
    new_lines = []

    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={value}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def iter_pairs():
    criteria = sorted(CRITERIA_DIR.glob("*.yaml"))
    dialogues = sorted(DIALOGUES_DIR.glob("*.json"))

    if len(criteria) != len(dialogues):
        raise RuntimeError(f"criteria={len(criteria)} dialogues={len(dialogues)}")

    for criterion_path, dialogue_path in zip(criteria, dialogues):
        if criterion_path.stem != dialogue_path.stem:
            raise RuntimeError(f"Pair mismatch: {criterion_path.name} != {dialogue_path.name}")
        yield criterion_path, dialogue_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    pairs = list(iter_pairs())[args.start :]
    if args.limit is not None:
        pairs = pairs[: args.limit]

    for index, (criterion_path, dialogue_path) in enumerate(pairs, start=args.start + 1):
        output_path = RESPONSES_DIR / f"{criterion_path.stem}.json"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}] skip existing {output_path.name}")
            continue

        updates = {
            "DIALOGUE_INPUT_PATH": str(dialogue_path),
            "CRETERIONS_PATH": str(criterion_path),
            "LLM_RESPONSE_PATH": str(output_path),
        }
        print(f"[{index}] run {criterion_path.stem}")

        if args.dry_run:
            print(updates)
            continue

        update_env(ENV_PATH, updates)
        subprocess.run([sys.executable, "main.py"], cwd=ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
