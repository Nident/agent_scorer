from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from textwrap import dedent
import traceback

from dotenv import load_dotenv
import yaml

try:
    from model.model import (
        configure_file_logger,
        PROJECT_ROOT,
        Model,
        read_optional_float,
        read_optional_int,
        read_optional_timeout,
        read_thinking_mode,
    )
except ModuleNotFoundError:
    from model import (
        configure_file_logger,
        PROJECT_ROOT,
        Model,
        read_optional_float,
        read_optional_int,
        read_optional_timeout,
        read_thinking_mode,
    )

logger = configure_file_logger(__name__, "simple_query_model.log")
SIMPLE_MODEL_DIR = PROJECT_ROOT / "data" / "simple_model"
SIMPLE_MODEL_PROMPTS_DIR = SIMPLE_MODEL_DIR / "prompts"
SIMPLE_MODEL_RESPONSES_DIR = SIMPLE_MODEL_DIR / "responses"
DEFAULT_SIMPLE_MODEL_OUTPUT_PATH = "data/simple_model/simple_model_response.json"
DEFAULT_TEST_DIALOGUE_PATH = "data/test_dialogue.yaml"


class SimpleQueryModel(Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.simple_model_dir = SIMPLE_MODEL_DIR
        self.prompts_dir = SIMPLE_MODEL_PROMPTS_DIR
        self.responses_dir = SIMPLE_MODEL_RESPONSES_DIR
        self.simple_model_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def load_prompt_template(self, template_path: str | Path) -> str:
        raw = self.load_text(template_path)
        data = yaml.safe_load(raw) or {}
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt template YAML must contain a non-empty 'prompt' field.")
        return dedent(prompt).strip()

    def load_criterion(self, criterion_path: str | Path) -> str:
        return self.load_text(criterion_path)

    def load_test_dialogue(self, dialogue_path: str | Path = DEFAULT_TEST_DIALOGUE_PATH) -> str:
        raw = self.load_text(dialogue_path)
        data = yaml.safe_load(raw) or {}
        dialogue = data.get("dialogue")
        if not isinstance(dialogue, str) or not dialogue.strip():
            raise ValueError("Test dialogue YAML must contain a non-empty 'dialogue' field.")
        return dialogue.strip()

    def build_prompt(
        self,
        dialogue_block: str,
        criterion_path: str | Path,
        template_path: str | Path,
        evaluated_speaker: str = "B",
    ) -> str:
        return self.load_prompt_template(template_path).format(
            evaluated_speaker=evaluated_speaker,
            criterion=self.load_criterion(criterion_path),
            dialogue_block=dialogue_block,
        )

    def build_prompt_from_file(
        self,
        dialogue_path: str | Path,
        criterion_path: str | Path,
        template_path: str | Path,
        evaluated_speaker: str = "B",
    ) -> str:
        dialogue_block = self.load_text(dialogue_path)
        return self.build_prompt(
            dialogue_block=dialogue_block,
            criterion_path=criterion_path,
            template_path=template_path,
            evaluated_speaker=evaluated_speaker,
        )

    def _save_prompt_copy(self, prompt: str) -> Path:
        prompt_copy_path = self.prompts_dir / "assembled_prompt.md"
        prompt_copy_path.write_text(prompt, encoding="utf-8")
        logger.info("Saved simple model prompt to %s", prompt_copy_path)
        return prompt_copy_path

    def _save_response_copy(self, response: dict) -> Path:
        response_copy_path = self.responses_dir / "simple_model_response.json"
        response_copy_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved simple model response copy to %s", response_copy_path)
        return response_copy_path

    def simple_query(
        self,
        api_key: str,
        dialogue_block: str,
        criterion_path: str | Path,
        template_path: str | Path,
        evaluated_speaker: str = "B",
        output_path: str | Path = DEFAULT_SIMPLE_MODEL_OUTPUT_PATH,
    ) -> Path:
        logger.info("Starting simple query run")
        logger.info("Criterion path: %s", criterion_path)
        logger.info("Template path: %s", template_path)
        logger.info("Output path: %s", output_path)
        logger.info("Model: %s", self.model_name)
        prompt = self.build_prompt(
            dialogue_block=dialogue_block,
            criterion_path=criterion_path,
            template_path=template_path,
            evaluated_speaker=evaluated_speaker,
        )
        self._save_prompt_copy(prompt)

        extra_body = None
        if self.thinking_mode == "enabled" and self.model_name != "deepseek-reasoner":
            logger.info("Thinking mode enabled for simple query")
            extra_body = {"thinking": {"type": "enabled"}}

        response = self.request_json(
            api_key=api_key,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            extra_body=extra_body,
        )
        self._save_response_copy(response)
        saved_path = self.save_json(response, output_path)
        logger.info("Saved simple query response to %s", saved_path)
        return saved_path

    def query_json(self, api_key: str, prompt: str) -> dict:
        logger.info("Running simple query_json")
        extra_body = None
        if self.thinking_mode == "enabled" and self.model_name != "deepseek-reasoner":
            extra_body = {"thinking": {"type": "enabled"}}

        return self.request_json(
            api_key=api_key,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            extra_body=extra_body,
        )

    def run(
        self,
        api_key: str,
        dialogue_block: str,
        criterion_path: str | Path,
        template_path: str | Path,
        evaluated_speaker: str = "B",
        output_path: str | Path = DEFAULT_SIMPLE_MODEL_OUTPUT_PATH,
    ) -> Path:
        return self.simple_query(
            api_key=api_key,
            dialogue_block=dialogue_block,
            criterion_path=criterion_path,
            template_path=template_path,
            evaluated_speaker=evaluated_speaker,
            output_path=output_path,
        )


if __name__ == "__main__":
    try:
        load_dotenv(PROJECT_ROOT / "config" / ".env")
        load_dotenv(PROJECT_ROOT / "config" / "model.env")

        response_path = os.getenv("LLM_RESPONSE_PATH", DEFAULT_SIMPLE_MODEL_OUTPUT_PATH)
        dialogue_input_path = os.getenv("DIALOGUE_INPUT_PATH", "").strip()
        test_dialogue_path = os.getenv("TEST_DIALOGUE_PATH", DEFAULT_TEST_DIALOGUE_PATH).strip()
        criterion_path = os.getenv("CRETERIONS_PATH", "").strip()
        template_path = os.getenv("TEMPLATE_PATH", "").strip()
        evaluated_speaker = os.getenv("EVALUATED_SPEAKER", "B").strip() or "B"
        api_key = os.getenv("API_KEY", "")
        model_name = os.getenv("MODEL_NAME", "deepseek-chat")
        base_url = os.getenv("BASE_URL", "https://api.deepseek.com")
        temperature = read_optional_float("TEMPERATURE")
        max_tokens = read_optional_int("MAX_TOKENS")
        top_p = read_optional_float("TOP_P")
        top_k = read_optional_int("TOP_K")
        thinking_mode = read_thinking_mode()
        max_retries = read_optional_int("RESPONSE_MAX_RETRIES") or 3
        request_timeout = read_optional_timeout("REQUEST_TIMEOUT")

        model = SimpleQueryModel(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            thinking_mode=thinking_mode,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )

        if not criterion_path:
            raise ValueError("CRETERIONS_PATH is empty.")
        if not template_path:
            raise ValueError("TEMPLATE_PATH is empty.")

        if dialogue_input_path:
            dialogue_block = model.load_text(dialogue_input_path)
        else:
            dialogue_block = model.load_test_dialogue(test_dialogue_path)

        saved_path = model.simple_query(
            api_key=api_key,
            dialogue_block=dialogue_block,
            criterion_path=criterion_path,
            template_path=template_path,
            evaluated_speaker=evaluated_speaker,
            output_path=response_path,
        )
        logger.info("Saved LLM response to %s", saved_path)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
