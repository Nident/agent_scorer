import os
from pathlib import Path

from dotenv import load_dotenv

from models import DialogueSummaryModule, PointQueryModel, SimpleQueryModel
from utils import (
    load_dialogue,
    load_steps,
    resolve_criterion_path,
    resolve_dialogue_path,
    resolve_project_path,
    to_bool,
)


MODELS = {"simple": SimpleQueryModel, "points": PointQueryModel}


def _resolve_prompt_path(model_type: str) -> Path:
    env_prompt = os.getenv("PROMPT_PATH", "").strip()
    default_prompt = f"data/prompts/{model_type}_model.yaml"
    if not env_prompt:
        return resolve_project_path(default_prompt)

    candidate = resolve_project_path(env_prompt)
    if candidate.exists():
        return candidate

    return resolve_project_path(default_prompt)


if __name__ == "__main__":
    load_dotenv(resolve_project_path("config/.env"))

    model_type = os.getenv("MODEL_TYPE", "simple").strip().lower()
    if model_type not in MODELS:
        raise ValueError(f"Unsupported MODEL_TYPE: {model_type}. Available: {', '.join(MODELS)}")

    prompt_path = _resolve_prompt_path(model_type)
    dialogue_path = resolve_dialogue_path(os.getenv("DIALOGUE_INPUT_PATH", ""))
    criterion_path = resolve_criterion_path(os.getenv("CRETERIONS_PATH", ""))
    output_path = resolve_project_path(os.getenv("LLM_RESPONSE_PATH", "data/response.json"))
    api_key = os.getenv("API_KEY", "")
    evaluated_speaker = os.getenv("EVALUATED_SPEAKER", "B")
    dialogue_block_size = int(os.getenv("DIALOGUE_BLOCK_SIZE", "6"))
    skip_predict = to_bool(os.getenv("SKIP_PREDICT", "1"), default=True)
    summary_skip_predict = to_bool(os.getenv("SUMMARY_SKIP_PREDICT", str(int(skip_predict))), default=skip_predict)

    steps = load_steps(prompt_path, f"{model_type}_model")
    dialogue = load_dialogue(dialogue_path)

    summary_module = DialogueSummaryModule()
    summary = summary_module.run(
        context={
            "api_key": api_key,
            "dialogue": dialogue,
            "evaluated_speaker": evaluated_speaker,
            "dialogue_block_size": dialogue_block_size,
            "skip_predict": skip_predict,
            "summary_skip_predict": summary_skip_predict,
        }
    )

    context = {
        "api_key": api_key,
        "dialogue": dialogue,
        "criterion_path": str(criterion_path),
        "evaluated_speaker": evaluated_speaker,
        "dialogue_block_size": dialogue_block_size,
        "skip_predict": skip_predict,
        "summary": summary,
        "dialogue_summary": summary.get("combined_summary", ""),
        "dialogue_summaries": summary.get("blocks", []),
    }

    model = MODELS[model_type]()
    result = model.run(steps=steps, context=context)

    saved_path = model.save_response(
        result,
        output_path,
        f"{model_type}_model",
        metadata={
            "criterion_path": str(criterion_path),
            "criterion_file": criterion_path.name,
            "prompt_path": str(prompt_path),
            "prompt_file": prompt_path.name,
            "dialogue_path": str(dialogue_path),
            "dialogue_file": dialogue_path.name,
            "summary_mode": summary.get("mode", ""),
        },
    )
    print(f"Saved output: {saved_path}")
