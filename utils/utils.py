from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_steps(prompt_path: str | Path, model_key: str) -> list[dict]:
    path = resolve_project_path(prompt_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    steps = data.get(model_key)
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Expected non-empty '{model_key}' list in {path}")
    return steps


def load_dialogue(dialogue_path: str | Path):
    path = resolve_project_path(dialogue_path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    return raw


def resolve_dialogue_path(
    value: str,
    default_rel_path: str = "data/dialogues/transcript-1_transcript.json",
) -> Path:
    default_path = ROOT_DIR / default_rel_path
    if not value:
        return default_path

    path = resolve_project_path(value)
    if path.exists():
        return path

    return default_path


def resolve_criterion_path(
    value: str,
    default_rel_path: str = "data/creterions/PrepareForNegotiations.yaml",
) -> Path:
    if not value:
        return ROOT_DIR / default_rel_path

    path = resolve_project_path(value)
    if path.is_dir():
        yaml_files = sorted(path.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No criterion YAML files found in {path}")
        return yaml_files[0]
    return path


def load_text_if_exists(path_value: str | Path) -> str:
    if not path_value:
        return ""
    path = resolve_project_path(path_value)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def dialogue_to_text(dialogue: Any, block_size: int = 6):
    if block_size <= 0:
        block_size = 1

    lines: list[str] = []
    if isinstance(dialogue, str):
        lines = [line.strip() for line in dialogue.splitlines() if line.strip()]
    elif isinstance(dialogue, dict):
        utterances = dialogue.get("transcript", {}).get("utterances", [])
        if isinstance(utterances, list):
            for utterance in utterances:
                if not isinstance(utterance, dict):
                    continue
                speaker = str(utterance.get("speaker_label", "")).strip()
                text = str(utterance.get("text", "")).strip()
                if speaker and text:
                    lines.append(f"{speaker}: {text}")
    else:
        lines = [json.dumps(dialogue, ensure_ascii=False)]

    if not lines:
        return

    for idx in range(0, len(lines), block_size):
        yield "\n".join(lines[idx : idx + block_size])


def load_points(criterion_path: str | Path, context: dict) -> list[str]:
    points = context.get("points")
    if isinstance(points, list) and points:
        return [str(point).strip() for point in points if str(point).strip()]

    if not criterion_path:
        return []

    path = resolve_project_path(criterion_path)
    if not path.exists() or not path.is_file():
        return []

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_points = data.get("points")

    if isinstance(raw_points, list):
        return [str(point).strip() for point in raw_points if str(point).strip()]

    if isinstance(raw_points, str):
        parsed = []
        for line in raw_points.splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                parsed.append(stripped.lstrip("-").strip())
        return [point for point in parsed if point]

    return []


def get_summary_blocks(context: dict) -> list[dict]:
    summary_blocks = context.get("dialogue_summaries")
    if isinstance(summary_blocks, list):
        return [block for block in summary_blocks if isinstance(block, dict)]

    summary = context.get("summary")
    if isinstance(summary, dict):
        blocks = summary.get("blocks")
        if isinstance(blocks, list):
            return [block for block in blocks if isinstance(block, dict)]

    return []


def summary_for_block(context: dict, block_index: int) -> str:
    for block in get_summary_blocks(context):
        if int(block.get("block_index", -1)) == block_index:
            return str(block.get("summary", "") or "")
    return ""


def summary_history_before(context: dict, block_index: int) -> str:
    history = []
    for block in get_summary_blocks(context):
        current_index = int(block.get("block_index", -1))
        if 0 < current_index < block_index and block.get("summary"):
            history.append(f"Блок {current_index}: {block['summary']}")
    return "\n".join(history)


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default
