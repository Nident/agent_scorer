from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmark.dataset import BenchmarkCase, load_benchmark_cases, resolve_path
from benchmark.metrics import PredictionRecord, compute_report
from model.model import (
    Model,
    read_optional_float,
    read_optional_int,
    read_optional_timeout,
    read_thinking_mode,
)


DEFAULT_CASES_PATH = "data/benchmark_cases.yaml"
DEFAULT_SAMPLE_PREDICTIONS_PATH = "data/benchmark_sample_predictions.jsonl"
DEFAULT_OUTPUT_DIR = "data/benchmark_runs/latest"


def build_benchmark_prompt(case: BenchmarkCase) -> str:
    points = "\n".join(
        f"{index}. {point}" for index, point in enumerate(case.criterion.points, start=1)
    )
    max_score = len(case.criterion.points)
    return f"""You are a strict negotiation-readiness evaluator.

Evaluate only the visible dialogue. Do not infer facts, tone, emotions, or preparation that are not supported by text.
The evaluated side can be either buyer/customer or seller; apply the same criterion fairly.

Scoring:
- Check every numbered point independently.
- Add 1 raw point only when the evaluated speaker has clear textual evidence for that point.
- Do not subtract points for negative evidence.
- max_score is {max_score}.
- score_0_10 = round(score_raw / max_score * 10, 2).
- Use 1-based point indices from the numbered criterion below.
- Evidence quotes must be exact substrings from the dialogue.
- Keep rationales compact: no more than 20 words each.

Evaluated speaker: {case.evaluated_speaker}
Criterion: {case.criterion.name}

Criterion points:
{points}

Dialogue:
{case.dialogue}

Return JSON only with this structure:
{{
  "evaluated_speaker": "{case.evaluated_speaker}",
  "criterion": "{case.criterion.name}",
  "score_raw": 0,
  "max_score": {max_score},
  "score_0_10": 0.0,
  "matched_point_indices": [],
  "matched_points": [
    {{
      "point_index": 1,
      "quote": "exact quote from the dialogue",
      "rationale": "short evidence-based rationale",
      "confidence": 0.0
    }}
  ]
}}"""


def load_predictions(path: str | Path) -> list[PredictionRecord]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Predictions file was not found: {resolved}")

    if resolved.suffix.lower() == ".jsonl":
        return _load_jsonl_predictions(resolved)
    return _load_json_predictions(resolved)


def run_benchmark(
    cases_path: str | Path,
    output_dir: str | Path,
    predictions_path: str | Path | None = None,
    run_llm: bool = False,
    model_env_path: str | Path = "config/model.env",
    limit: int | None = None,
    case_ids: list[str] | None = None,
) -> Path:
    cases = load_benchmark_cases(cases_path)
    if case_ids:
        selected = set(case_ids)
        cases = [case for case in cases if case.id in selected]
        missing = selected - {case.id for case in cases}
        if missing:
            raise ValueError(f"Unknown benchmark case id(s): {', '.join(sorted(missing))}")
    if limit is not None:
        cases = cases[:limit]
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if run_llm:
        predictions = _run_llm_predictions(cases, output, model_env_path)
    else:
        if predictions_path is None:
            predictions_path = DEFAULT_SAMPLE_PREDICTIONS_PATH
        predictions = load_predictions(predictions_path)

    report = compute_report(cases, predictions)
    report_json_path = output / "report.json"
    report_md_path = output / "report.md"

    report_json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_md_path.write_text(report.to_markdown(), encoding="utf-8")
    return report_md_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run negotiation-readiness benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command")
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate predictions or run an LLM on benchmark cases.",
    )
    benchmark_parser.add_argument("--cases", default=DEFAULT_CASES_PATH)
    benchmark_parser.add_argument("--predictions", default=None)
    benchmark_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    benchmark_parser.add_argument("--run-llm", action="store_true")
    benchmark_parser.add_argument("--model-env", default="config/model.env")
    benchmark_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases. Useful for API smoke checks.",
    )
    benchmark_parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Run only a specific case id. Can be passed multiple times.",
    )

    args = parser.parse_args(argv)
    if args.command in (None, "benchmark"):
        report_path = run_benchmark(
            cases_path=getattr(args, "cases", DEFAULT_CASES_PATH),
            predictions_path=getattr(args, "predictions", None),
            output_dir=getattr(args, "output_dir", DEFAULT_OUTPUT_DIR),
            run_llm=getattr(args, "run_llm", False),
            model_env_path=getattr(args, "model_env", "config/model.env"),
            limit=getattr(args, "limit", None),
            case_ids=getattr(args, "case_id", None),
        )
        print(f"Saved benchmark report to {report_path}")
        return

    parser.print_help()


def _load_jsonl_predictions(path: Path) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            records.append(
                PredictionRecord(
                    case_id=f"line_{line_number}",
                    prediction=None,
                    valid_json=False,
                    error=str(exc),
                )
            )
            continue
        records.append(_prediction_record_from_object(raw, f"line_{line_number}"))
    return records


def _load_json_predictions(path: Path) -> list[PredictionRecord]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            PredictionRecord(
                case_id=path.stem,
                prediction=None,
                valid_json=False,
                error=str(exc),
            )
        ]

    if isinstance(raw, dict) and isinstance(raw.get("predictions"), list):
        return [
            _prediction_record_from_object(item, f"item_{index}")
            for index, item in enumerate(raw["predictions"], start=1)
        ]
    if isinstance(raw, dict):
        records: list[PredictionRecord] = []
        for case_id, prediction in raw.items():
            if isinstance(prediction, dict):
                records.append(
                    PredictionRecord(case_id=str(case_id), prediction=prediction, valid_json=True)
                )
        return records
    if isinstance(raw, list):
        return [
            _prediction_record_from_object(item, f"item_{index}")
            for index, item in enumerate(raw, start=1)
        ]

    return [
        PredictionRecord(
            case_id=path.stem,
            prediction=None,
            valid_json=False,
            error="Predictions JSON must be an object or list.",
        )
    ]


def _prediction_record_from_object(raw: Any, fallback_id: str) -> PredictionRecord:
    if not isinstance(raw, dict):
        return PredictionRecord(
            case_id=fallback_id,
            prediction=None,
            valid_json=False,
            error="Prediction row is not a JSON object.",
        )

    case_id = str(raw.get("case_id") or raw.get("id") or fallback_id)
    declared_valid_json = bool(raw.get("valid_json", True))
    error = str(raw.get("error") or "")

    if "prediction" in raw:
        prediction = raw.get("prediction")
        if not declared_valid_json or not isinstance(prediction, dict):
            return PredictionRecord(
                case_id=case_id,
                prediction=None,
                valid_json=False,
                error=error or "Prediction payload is not a JSON object.",
            )
    else:
        prediction = {
            key: value
            for key, value in raw.items()
            if key not in {"case_id", "id", "valid_json", "error"}
        }

    if not isinstance(prediction, dict):
        return PredictionRecord(
            case_id=case_id,
            prediction=None,
            valid_json=False,
            error="Prediction payload is not a JSON object.",
        )
    return PredictionRecord(case_id=case_id, prediction=prediction, valid_json=True)


def _run_llm_predictions(
    cases: list[BenchmarkCase],
    output_dir: Path,
    model_env_path: str | Path,
) -> list[PredictionRecord]:
    load_dotenv(resolve_path(model_env_path))
    api_key = os.getenv("API_KEY", "")
    model = Model(
        model_name=os.getenv("MODEL_NAME", "deepseek-chat"),
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com"),
        temperature=read_optional_float("TEMPERATURE"),
        max_tokens=read_optional_int("MAX_TOKENS"),
        top_p=read_optional_float("TOP_P"),
        top_k=read_optional_int("TOP_K"),
        thinking_mode=read_thinking_mode(),
        max_retries=read_optional_int("RESPONSE_MAX_RETRIES") or 3,
        request_timeout=read_optional_timeout("REQUEST_TIMEOUT"),
    )

    predictions_path = output_dir / "predictions.jsonl"
    records: list[PredictionRecord] = []
    with predictions_path.open("w", encoding="utf-8") as file:
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{total}] Running DeepSeek benchmark case: {case.id}", flush=True)
            try:
                prediction = model.query_json(api_key, build_benchmark_prompt(case))
                record = PredictionRecord(case_id=case.id, prediction=prediction)
            except Exception as exc:  # noqa: BLE001 - benchmark should keep going per case.
                record = PredictionRecord(
                    case_id=case.id,
                    prediction=None,
                    valid_json=False,
                    error=str(exc),
                )
            records.append(record)
            file.write(
                json.dumps(
                    {
                        "case_id": record.case_id,
                        "prediction": record.prediction,
                        "valid_json": record.valid_json,
                        "error": record.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()
    return records


if __name__ == "__main__":
    main()
