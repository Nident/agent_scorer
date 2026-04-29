from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from models import DialogueSummaryModule, PointQueryModel, SimpleQueryModel
from utils import (
    ROOT_DIR,
    load_dialogue,
    load_points,
    load_steps,
    load_text_if_exists,
    resolve_criterion_path,
    resolve_dialogue_path,
    resolve_project_path,
    to_bool,
)


MODELS = {"simple": SimpleQueryModel, "points": PointQueryModel}
FRONTEND_DIR = ROOT_DIR / "frontend"


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(default="points")
    prompt_path: str | None = None
    criterion_path: str | None = None
    dialogue_path: str | None = None
    dialogue: Any | None = None
    evaluated_speaker: str = Field(default="B")
    dialogue_block_size: int = Field(default=6, ge=1, le=40)
    skip_predict: bool = True
    summary_skip_predict: bool | None = None
    api_key: str | None = None
    save_response: bool = False


def _resolve_prompt_path(model_type: str, prompt_path: str | None = None) -> Path:
    if prompt_path:
        candidate = resolve_project_path(prompt_path)
        if candidate.exists():
            return candidate

    env_prompt = os.getenv("PROMPT_PATH", "").strip()
    if env_prompt:
        candidate = resolve_project_path(env_prompt)
        if candidate.exists():
            return candidate

    return resolve_project_path(f"data/prompts/{model_type}_model.yaml")


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_criterion_meta(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    points = load_points(path, {})
    return {
        "name": raw.get("name") or path.stem,
        "path": str(path),
        "file": path.name,
        "points": points,
        "text": load_text_if_exists(path),
    }


def _label_to_number(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "+1", "present", "yes", "true", "entailment"}:
            return 1
        if normalized in {"-1", "opposite", "contradiction", "false"}:
            return -1
        if normalized in {"0", "absent", "neutral", "no", ""}:
            return 0
    return None


def _parse_possible_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    return _safe_json_loads(cleaned)


def _extract_competencies(model_type: str, result: dict[str, Any], criterion_points: list[str]) -> list[dict[str, Any]]:
    competencies: list[dict[str, Any]] = []

    if model_type == "points":
        for item in result.get("results", []):
            final = _parse_possible_json(item.get("final"))
            if not isinstance(final, dict):
                final = {}
            label = _label_to_number(final.get("label", final.get("score")))
            competencies.append(
                {
                    "point": item.get("point", ""),
                    "label": label,
                    "evidence": final.get("evidence", final.get("quote", "")),
                    "rationale": final.get("rationale", final.get("reasoning", "")),
                    "confidence": final.get("confidence"),
                }
            )
        return competencies

    final = _parse_possible_json(result.get("final", {}))
    if isinstance(final, dict) and isinstance(final.get("raw_response"), str):
        final = _parse_possible_json(final["raw_response"])

    if isinstance(final, dict) and isinstance(final.get("points"), list):
        for item in final["points"]:
            if not isinstance(item, dict):
                continue
            competencies.append(
                {
                    "point": item.get("point", ""),
                    "label": _label_to_number(item.get("score", item.get("label"))),
                    "evidence": item.get("quote", item.get("evidence", "")),
                    "rationale": item.get("rationale", item.get("reasoning", "")),
                    "confidence": item.get("confidence"),
                }
            )

    if not competencies:
        competencies = [{"point": point, "label": None, "evidence": "", "rationale": "", "confidence": None} for point in criterion_points]

    return competencies


def _build_score(model_type: str, result: dict[str, Any], criterion_points: list[str]) -> dict[str, Any]:
    competencies = _extract_competencies(model_type, result, criterion_points)
    scored = [item for item in competencies if item.get("label") is not None]
    if not scored:
        return {
            "score_10": 0.0,
            "score_100": 0,
            "evaluated": 0,
            "total": len(competencies),
            "competencies": competencies,
        }

    positive = sum(1 for item in scored if item["label"] == 1)
    score_10 = round((positive / len(scored)) * 10, 1)
    return {
        "score_10": score_10,
        "score_100": int(round(score_10 * 100)),
        "evaluated": len(scored),
        "total": len(competencies),
        "competencies": competencies,
    }


def _dialogue_metadata(dialogue: Any, path: Path | None) -> dict[str, Any]:
    if isinstance(dialogue, dict):
        transcript = dialogue.get("transcript", {})
        utterances = transcript.get("utterances", [])
        speakers = transcript.get("speakers", [])
        return {
            "file": path.name if path else "",
            "language": transcript.get("language", ""),
            "turn_count": len(utterances) if isinstance(utterances, list) else transcript.get("turn_count", 0),
            "speakers": speakers if isinstance(speakers, list) else [],
        }
    if isinstance(dialogue, str):
        return {
            "file": path.name if path else "",
            "language": "",
            "turn_count": len([line for line in dialogue.splitlines() if line.strip()]),
            "speakers": [],
        }
    return {"file": path.name if path else "", "language": "", "turn_count": 0, "speakers": []}


def run_analysis(request: AnalysisRequest) -> dict[str, Any]:
    model_type = request.model_type.strip().lower()
    if model_type not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {request.model_type}")

    prompt_path = _resolve_prompt_path(model_type, request.prompt_path)
    criterion_path = resolve_criterion_path(request.criterion_path or os.getenv("CRETERIONS_PATH", ""))
    dialogue_path = None
    if request.dialogue is None:
        dialogue_path = resolve_dialogue_path(request.dialogue_path or os.getenv("DIALOGUE_INPUT_PATH", ""))
        dialogue = load_dialogue(dialogue_path)
    else:
        dialogue = request.dialogue
        if isinstance(dialogue, str):
            dialogue = _safe_json_loads(dialogue)

    api_key = request.api_key or os.getenv("API_KEY", "")
    skip_predict = bool(request.skip_predict)
    summary_skip_predict = request.summary_skip_predict
    if summary_skip_predict is None:
        summary_skip_predict = skip_predict

    steps = load_steps(prompt_path, f"{model_type}_model")
    criterion_meta = _load_criterion_meta(criterion_path)
    base_context = {
        "api_key": api_key,
        "dialogue": dialogue,
        "criterion_path": str(criterion_path),
        "evaluated_speaker": request.evaluated_speaker,
        "dialogue_block_size": request.dialogue_block_size,
        "skip_predict": skip_predict,
        "summary_skip_predict": summary_skip_predict,
    }

    summary_module = DialogueSummaryModule()
    summary = summary_module.run(context=base_context)

    context = {
        **base_context,
        "summary": summary,
        "dialogue_summary": summary.get("combined_summary", ""),
        "dialogue_summaries": summary.get("blocks", []),
    }

    model = MODELS[model_type]()
    result = model.run(steps=steps, context=context)
    score = _build_score(model_type, result, criterion_meta["points"])

    saved_path = ""
    if request.save_response:
        output_path = resolve_project_path(os.getenv("LLM_RESPONSE_PATH", "data/responses/web"))
        saved_path = str(
            model.save_response(
                result,
                output_path,
                f"{model_type}_model",
                metadata={
                    "criterion_path": str(criterion_path),
                    "criterion_file": criterion_path.name,
                    "prompt_path": str(prompt_path),
                    "prompt_file": prompt_path.name,
                    "dialogue_path": str(dialogue_path) if dialogue_path else "",
                    "dialogue_file": dialogue_path.name if dialogue_path else "inline",
                    "summary_mode": summary.get("mode", ""),
                    "source": "web_api",
                },
            )
        )

    return {
        "session": {
            "model_type": model_type,
            "prompt_file": prompt_path.name,
            "criterion": criterion_meta,
            "dialogue": _dialogue_metadata(dialogue, dialogue_path),
            "evaluated_speaker": request.evaluated_speaker,
            "dialogue_block_size": request.dialogue_block_size,
            "skip_predict": skip_predict,
            "saved_path": saved_path,
        },
        "summary": summary,
        "score": score,
        "analysis": result,
    }


def create_app() -> FastAPI:
    load_dotenv(resolve_project_path("config/.env"))
    app = FastAPI(title="Agent Scorer", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/options")
    def options() -> dict[str, Any]:
        criteria_dir = ROOT_DIR / "data" / "creterions"
        dialogues_dir = ROOT_DIR / "data" / "dialogues"

        criteria = []
        for path in sorted(criteria_dir.glob("*.yaml")):
            try:
                meta = _load_criterion_meta(path)
            except Exception:
                meta = {"name": path.stem, "path": str(path), "file": path.name, "points": [], "text": ""}
            criteria.append(meta)

        dialogues = []
        for path in sorted(dialogues_dir.glob("*.json")):
            try:
                dialogue = load_dialogue(path)
                meta = _dialogue_metadata(dialogue, path)
            except Exception:
                meta = {"file": path.name, "language": "", "turn_count": 0, "speakers": []}
            dialogues.append({"path": str(path), **meta})

        return {
            "models": sorted(MODELS),
            "criteria": criteria,
            "dialogues": dialogues,
            "defaults": {
                "model_type": os.getenv("MODEL_TYPE", "points"),
                "dialogue_block_size": int(os.getenv("DIALOGUE_BLOCK_SIZE", "6")),
                "evaluated_speaker": os.getenv("EVALUATED_SPEAKER", "B"),
                "skip_predict": to_bool(os.getenv("SKIP_PREDICT", "1"), default=True),
            },
        }

    @app.post("/api/analyze")
    def analyze(request: AnalysisRequest) -> dict[str, Any]:
        return run_analysis(request)

    @app.get("/")
    def index() -> FileResponse:
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend is not built")
        return FileResponse(index_path)

    @app.get("/{path:path}")
    def spa_fallback(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend is not built")
        return FileResponse(index_path)

    return app


app = create_app()
