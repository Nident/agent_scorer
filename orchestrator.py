from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, overload

from dotenv import load_dotenv

from model import (
    PROJECT_ROOT,
    configure_file_logger,
    read_optional_float,
    read_optional_int,
    read_optional_timeout,
    read_thinking_mode,
)
from model.point_agents_model import DEFAULT_PROMPTS_PATH, PointAgentsModel
from model.simple_query_model import DEFAULT_SIMPLE_MODEL_OUTPUT_PATH, SimpleQueryModel

logger = configure_file_logger(__name__, "orchestrator.log")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "orchestrator"
DEFAULT_TRANSCRIPT_PATH = "data/test_dialogue.json"
DEFAULT_MODEL_TYPE = "simple"
DEFAULT_BATCH_SIZE = 2


def load_runtime_env() -> None:
    load_dotenv(PROJECT_ROOT / "config" / ".env")
    load_dotenv(PROJECT_ROOT / "config" / "model.env")
    load_dotenv(PROJECT_ROOT / "config" / "point_model.env")
    load_dotenv(PROJECT_ROOT / "config" / "orchestrator.env")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_transcript_payload(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Transcript file must be a JSON object: {resolved}")
    transcript = data.get("transcript")
    if not isinstance(transcript, dict):
        raise ValueError(f"Transcript JSON must contain 'transcript': {resolved}")
    utterances = transcript.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError(f"Transcript JSON must contain non-empty 'utterances': {resolved}")
    return data


def format_utterances(utterances: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for utterance in utterances:
        speaker = str(utterance.get("speaker_label") or "").strip()
        text = str(utterance.get("text") or "").strip()
        if not speaker or not text:
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def chunk_utterances(utterances: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("ORCHESTRATOR_BATCH_SIZE must be > 0")
    return [utterances[index : index + batch_size] for index in range(0, len(utterances), batch_size)]


def build_summary_prompt(previous_memory: str, batch_dialogue: str, batch_index: int, total_batches: int) -> str:
    return f"""You are a dialogue memory builder.
            Your task is to update a compact working memory for a negotiation dialogue.
            Keep only facts that matter for later evaluation:
            - goals and constraints
            - offers, counteroffers, concessions
            - priorities and objections
            - agreements, open questions, and unresolved tensions

            Rules:
            - keep it short and factual
            - do not invent anything
            - plain text only
            - maximum 8 bullet points

            Batch {batch_index} of {total_batches}

            Previous memory:
            {previous_memory or 'None'}

            Current batch:
            {batch_dialogue}

            Return the updated memory only.
            """


def build_final_dialogue(full_dialogue: str, memory: str) -> str:
    if not memory.strip():
        return full_dialogue
    return (
        "Conversation memory from previous batches:\n"
        f"{memory.strip()}\n\n"
        "Full dialogue:\n"
        f"{full_dialogue}"
    )


@overload
def build_model(model_type: Literal["simple"]) -> SimpleQueryModel: ...


@overload
def build_model(model_type: Literal["point"]) -> PointAgentsModel: ...


def build_model(model_type: Literal["simple", "point"]) -> SimpleQueryModel | PointAgentsModel:
    common_config = {
        "model_name": os.getenv("MODEL_NAME", "deepseek-chat"),
        "base_url": os.getenv("BASE_URL", "https://api.deepseek.com"),
        "temperature": read_optional_float("TEMPERATURE"),
        "max_tokens": read_optional_int("MAX_TOKENS"),
        "top_p": read_optional_float("TOP_P"),
        "top_k": read_optional_int("TOP_K"),
        "thinking_mode": read_thinking_mode(),
        "max_retries": read_optional_int("RESPONSE_MAX_RETRIES") or 3,
        "request_timeout": read_optional_timeout("REQUEST_TIMEOUT"),
    }

    if model_type == "simple":
        return SimpleQueryModel(**common_config)
    if model_type == "point":
        prompt_path = os.getenv("POINT_AGENTS_PROMPTS_PATH", str(DEFAULT_PROMPTS_PATH))
        return PointAgentsModel(prompt_path=prompt_path, **common_config)
    raise ValueError("ORCHESTRATOR_MODEL_TYPE must be 'simple' or 'point'.")


def summarize_dialogue_batches(
    model: SimpleQueryModel | PointAgentsModel,
    api_key: str,
    utterances: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    batch_size = read_optional_int("ORCHESTRATOR_BATCH_SIZE") or DEFAULT_BATCH_SIZE
    summary_model_name = os.getenv("SUMMARY_MODEL_NAME", "").strip() or model.model_name
    summary_temperature = read_optional_float("SUMMARY_TEMPERATURE")
    summary_max_tokens = read_optional_int("SUMMARY_MAX_TOKENS")
    summary_top_p = read_optional_float("SUMMARY_TOP_P")

    batches_dir = ensure_dir(output_dir / "batches")
    memory = ""
    batches = chunk_utterances(utterances, batch_size)
    total_batches = len(batches)
    logger.info("Transcript split into %s batches with batch_size=%s", total_batches, batch_size)

    for index, batch in enumerate(batches, start=1):
        batch_dialogue = format_utterances(batch)
        batch_prompt = build_summary_prompt(memory, batch_dialogue, index, total_batches)
        (batches_dir / f"{index:02d}_dialogue.md").write_text(batch_dialogue, encoding="utf-8")
        (batches_dir / f"{index:02d}_summary_prompt.md").write_text(batch_prompt, encoding="utf-8")
        logger.info("Summarizing batch %s/%s", index, total_batches)
        memory = model.request_text(
            api_key=api_key,
            prompt=batch_prompt,
            model_name=summary_model_name,
            temperature=summary_temperature,
            max_tokens=summary_max_tokens,
            top_p=summary_top_p,
        )
        (batches_dir / f"{index:02d}_memory.md").write_text(memory, encoding="utf-8")

    return memory


def run() -> Path:
    load_runtime_env()

    api_key = os.getenv("API_KEY", "")
    transcript_path = os.getenv("ORCHESTRATOR_INPUT_PATH", DEFAULT_TRANSCRIPT_PATH)
    criterion_path = os.getenv("CRETERIONS_PATH", "")
    template_path = os.getenv("TEMPLATE_PATH", "")
    evaluated_speaker = os.getenv("EVALUATED_SPEAKER", "B").strip() or "B"
    model_type = os.getenv("ORCHESTRATOR_MODEL_TYPE", DEFAULT_MODEL_TYPE).strip().lower()
    output_dir = ensure_dir(os.getenv("ORCHESTRATOR_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))

    if not criterion_path:
        raise ValueError("CRETERIONS_PATH is empty.")

    transcript_payload = load_transcript_payload(transcript_path)
    utterances = transcript_payload["transcript"]["utterances"]
    full_dialogue = format_utterances(utterances)
    (output_dir / "full_dialogue.md").write_text(full_dialogue, encoding="utf-8")

    if model_type == "simple":
        model = build_model("simple")
        memory = summarize_dialogue_batches(model, api_key, utterances, output_dir)
        final_dialogue = build_final_dialogue(full_dialogue, memory)
        (output_dir / "final_dialogue_with_memory.md").write_text(final_dialogue, encoding="utf-8")
        logger.info("Running final evaluation with model_type=%s", model_type)
        if not template_path:
            raise ValueError("TEMPLATE_PATH is empty for simple orchestrator mode.")
        output_path = os.getenv("LLM_RESPONSE_PATH", DEFAULT_SIMPLE_MODEL_OUTPUT_PATH)
        return model.simple_query(
            api_key=api_key,
            dialogue_block=final_dialogue,
            criterion_path=criterion_path,
            template_path=template_path,
            evaluated_speaker=evaluated_speaker,
            output_path=output_path,
        )

    model = build_model("point")
    memory = summarize_dialogue_batches(model, api_key, utterances, output_dir)
    final_dialogue = build_final_dialogue(full_dialogue, memory)
    (output_dir / "final_dialogue_with_memory.md").write_text(final_dialogue, encoding="utf-8")
    logger.info("Running final evaluation with model_type=%s", model_type)
    output_path = os.getenv("POINT_AGENTS_OUTPUT_PATH", "data/point_agents/point_agents_response.json")
    return model.evaluate_criterion(
        api_key=api_key,
        dialogue=final_dialogue,
        criterion_path=criterion_path,
        evaluated_speaker=evaluated_speaker,
        output_path=output_path,
    )


def main() -> None:
    try:
        saved_path = run()
        print(f"Saved orchestrator result to {saved_path}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Orchestrator run failed")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
