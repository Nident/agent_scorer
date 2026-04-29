import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from .model import Model, ROOT_DIR
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.model import Model, ROOT_DIR

try:
    from utils import (
        dialogue_to_text,
        load_dialogue,
        load_steps,
        load_text_if_exists,
        to_bool,
        resolve_criterion_path,
        resolve_dialogue_path,
        resolve_project_path,
    )
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils import (
        dialogue_to_text,
        load_dialogue,
        load_steps,
        load_text_if_exists,
        to_bool,
        resolve_criterion_path,
        resolve_dialogue_path,
        resolve_project_path,
    )


class SimpleQueryModel(Model):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "")
        kwargs.setdefault("base_url", "")
        kwargs.setdefault("api_key", "")
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", 1000)
        kwargs.setdefault("top_p", 1.0)
        super().__init__(**kwargs)
        self.llm: Any | None = None

    def predict(self, input_data):
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Call _apply_step_config(...) before predict().")
        prompt = input_data["prompt"]
        answer = self.llm.invoke(prompt)
        return answer.content if isinstance(answer.content, str) else str(answer.content)

    def _apply_step_config(self, step: dict, context: dict, initialize_llm: bool) -> None:
        self.model_name = step.get("model_name", self.model_name)
        self.base_url = step.get("base_url", self.base_url)
        self.temperature = step.get("temperature", self.temperature)
        self.max_tokens = step.get("max_tokens", self.max_tokens)
        self.top_p = step.get("top_p", self.top_p)
        self.api_key = context.get("api_key", self.api_key)

        if initialize_llm:
            self.llm = self.load_model(
                model_name=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
        else:
            self.llm = None

    def run(self, *, steps: list[dict], context: dict) -> dict:
        if not steps:
            raise ValueError("steps is empty for SimpleQueryModel")

        skip_predict = to_bool(context.get("skip_predict", True), default=True)
        step = steps[0]
        self._apply_step_config(step, context, initialize_llm=not skip_predict)

        block_size = int(context.get("dialogue_block_size", 2))
        dialogue_blocks = list(dialogue_to_text(context.get("dialogue", ""), block_size=block_size))
        if not dialogue_blocks:
            dialogue_blocks = [""]

        criterion_text = load_text_if_exists(context.get("criterion_path", ""))
        evaluated_speaker = context.get("evaluated_speaker", "B")
        response_type = (step.get("response_type") or "text").lower()

        block_results = []
        dialogue_history = ""

        for idx, dialogue_block in enumerate(dialogue_blocks, start=1):
            prompt = step["prompt"].format_map(
                {
                    "evaluated_speaker": evaluated_speaker,
                    "criterion": criterion_text,
                    "dialogue": dialogue_block,
                    "dialogue_block": dialogue_block,
                    "dialogue_history": dialogue_history,
                    "dialoghistory": dialogue_history,
                }
            )

            if skip_predict:
                parsed_response = {
                    "skipped_predict": True,
                    "message": "Predict is disabled for debug run",
                    "response_type": response_type,
                }
            else:
                raw_response = self.predict({"prompt": prompt})
                if response_type == "json":
                    parsed_response = json.loads(raw_response)
                else:
                    parsed_response = {"raw_response": raw_response}

            block_results.append(
                {
                    "block_index": idx,
                    "dialogue_history": dialogue_history,
                    "dialogue_block": dialogue_block,
                    "prompt": prompt,
                    "result": parsed_response,
                }
            )

            dialogue_history = dialogue_block

        final_result = block_results[-1]["result"] if block_results else {}
        return {
            "evaluated_speaker": evaluated_speaker,
            "skip_predict": skip_predict,
            "blocks": block_results,
            "final": final_result,
        }


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / "config" / ".env")

    prompt_path = resolve_project_path(os.getenv("SIMPLE_PROMPT_PATH", "data/prompts/simple_model.yaml"))
    dialogue_path = resolve_dialogue_path(os.getenv("DIALOGUE_INPUT_PATH", ""))
    criterion_path = resolve_criterion_path(os.getenv("CRETERIONS_PATH", ""))
    output_path = resolve_project_path(
        os.getenv("SIMPLE_DEBUG_OUTPUT_PATH", "data/debug/simple_model_response.json")
    )

    steps = load_steps(prompt_path, "simple_model")
    dialogue = load_dialogue(dialogue_path)

    model = SimpleQueryModel()
    context = {
        "api_key": os.getenv("API_KEY", ""),
        "dialogue": dialogue,
        "criterion_path": str(criterion_path),
        "evaluated_speaker": os.getenv("EVALUATED_SPEAKER", "B"),
        "dialogue_block_size": int(os.getenv("DIALOGUE_BLOCK_SIZE", "2")),
        "skip_predict": to_bool(os.getenv("SKIP_PREDICT", "1"), default=True),
    }
    result = model.run(steps=steps, context=context)

    saved_path = model.save_response(
        result,
        output_path,
        "simple_model",
        metadata={
            "criterion_path": str(criterion_path),
            "criterion_file": criterion_path.name,
            "prompt_path": str(prompt_path),
            "prompt_file": prompt_path.name,
            "dialogue_path": str(dialogue_path),
            "dialogue_file": dialogue_path.name,
        },
    )
    print(f"Saved debug output: {saved_path}")
