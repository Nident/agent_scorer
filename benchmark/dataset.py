from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Criterion:
    path: Path
    name: str
    points: tuple[str, ...]


@dataclass(frozen=True)
class ExpectedAnswer:
    matched_point_indices: tuple[int, ...]
    score_raw: int
    max_score: int
    score_0_10: float


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    title: str
    role: str
    evaluated_speaker: str
    criterion: Criterion
    dialogue: str
    expected: ExpectedAnswer


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_points(points_block: str) -> tuple[str, ...]:
    points: list[str] = []
    for line in str(points_block).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if stripped:
            points.append(stripped)
    return tuple(points)


def load_criterion(path: str | Path) -> Criterion:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Criterion file was not found: {resolved}")

    raw = resolved.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Criterion YAML must be an object: {resolved}")

    name = str(data.get("name") or resolved.stem)
    points = parse_points(str(data.get("points") or ""))
    if not points:
        raise ValueError(f"Criterion has no points: {resolved}")

    return Criterion(path=resolved, name=name, points=points)


def _read_expected(raw_expected: dict[str, Any], criterion: Criterion) -> ExpectedAnswer:
    indices = tuple(int(item) for item in raw_expected.get("matched_point_indices", ()))
    if len(set(indices)) != len(indices):
        raise ValueError("Expected matched_point_indices must not contain duplicates.")

    max_score = int(raw_expected.get("max_score", len(criterion.points)))
    score_raw = int(raw_expected.get("score_raw", raw_expected.get("score", len(indices))))
    score_0_10 = float(
        raw_expected.get(
            "score_0_10",
            round((score_raw / max_score) * 10, 2) if max_score else 0.0,
        )
    )

    if max_score != len(criterion.points):
        raise ValueError(
            f"Expected max_score={max_score}, but criterion '{criterion.name}' "
            f"has {len(criterion.points)} points."
        )
    if score_raw != len(indices):
        raise ValueError(
            f"Expected score_raw={score_raw}, but matched_point_indices has "
            f"{len(indices)} items."
        )
    if not 0 <= score_raw <= max_score:
        raise ValueError("Expected score_raw must be between 0 and max_score.")

    for index in indices:
        if index < 1 or index > max_score:
            raise ValueError(
                f"Expected point index {index} is outside 1..{max_score} "
                f"for criterion '{criterion.name}'."
            )

    return ExpectedAnswer(
        matched_point_indices=indices,
        score_raw=score_raw,
        max_score=max_score,
        score_0_10=score_0_10,
    )


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Benchmark dataset was not found: {resolved}")

    raw = resolved.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    raw_cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(raw_cases, list):
        raise ValueError("Benchmark dataset must contain a 'cases' list.")

    criterion_cache: dict[Path, Criterion] = {}
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()

    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Each benchmark case must be a YAML object.")

        case_id = str(raw_case.get("id") or "").strip()
        if not case_id:
            raise ValueError("Each benchmark case must contain an id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate benchmark case id: {case_id}")
        seen_ids.add(case_id)

        criterion_path = resolve_path(raw_case.get("criterion_path", ""))
        criterion = criterion_cache.get(criterion_path)
        if criterion is None:
            criterion = load_criterion(criterion_path)
            criterion_cache[criterion_path] = criterion

        expected_raw = raw_case.get("expected")
        if not isinstance(expected_raw, dict):
            raise ValueError(f"Case '{case_id}' must contain an expected object.")

        dialogue = str(raw_case.get("dialogue") or "").strip()
        if not dialogue:
            raise ValueError(f"Case '{case_id}' must contain dialogue text.")

        cases.append(
            BenchmarkCase(
                id=case_id,
                title=str(raw_case.get("title") or case_id),
                role=str(raw_case.get("role") or ""),
                evaluated_speaker=str(raw_case.get("evaluated_speaker") or ""),
                criterion=criterion,
                dialogue=dialogue,
                expected=_read_expected(expected_raw, criterion),
            )
        )

    return cases

