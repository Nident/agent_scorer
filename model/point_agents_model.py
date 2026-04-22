from __future__ import annotations

import json
import os
from pathlib import Path
import traceback
from typing import Any

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

logger = configure_file_logger(__name__, "point_agents_model.log")
POINT_AGENTS_DIR = PROJECT_ROOT / "data" / "point_agents"
POINT_AGENTS_PROMPTS_DIR = POINT_AGENTS_DIR / "prompts"
POINT_AGENTS_RESPONSES_DIR = POINT_AGENTS_DIR / "responses"
DEFAULT_PROMPTS_PATH = POINT_AGENTS_DIR / "point_agents_prompts.yaml"
DEFAULT_OUTPUT_PATH = "data/point_agents/point_agents_response.json"
DEFAULT_TEST_DIALOGUE_PATH = "data/test_dialogue.yaml"


class PointAgentsModel(Model):
    def __init__(
        self,
        prompt_path: str | Path = DEFAULT_PROMPTS_PATH,
        step_temperature: float | None = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.step_temperature = step_temperature
        self.point_agents_dir = POINT_AGENTS_DIR
        self.prompts_dir = POINT_AGENTS_PROMPTS_DIR
        self.responses_dir = POINT_AGENTS_RESPONSES_DIR
        self.point_agents_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_steps = self._load_prompt_steps(prompt_path)
        logger.info("Initialized PointAgentsModel with %s prompt steps", len(self.prompt_steps))


    def load_points_from_criterion(self, criterion_path: str | Path) -> list[str]:
        raw = self.load_text(criterion_path)
        data = yaml.safe_load(raw) or {}
        raw_points = data.get("points")

        if isinstance(raw_points, list):
            points = [str(point).strip() for point in raw_points if str(point).strip()]
        elif isinstance(raw_points, str):
            points = [
                line.removeprefix("-").strip()
                for line in raw_points.splitlines()
                if line.strip().startswith("-")
            ]
        else:
            raise ValueError("Criterion YAML must contain 'points' as a string or list.")

        if not points:
            raise ValueError(
                f"No points were found in criterion file: {self._resolve_path(criterion_path)}"
            )

        logger.info(
            "Loaded %s points from criterion file %s",
            len(points),
            self._resolve_path(criterion_path),
        )
        return points

    def load_test_dialogue(self, dialogue_path: str | Path = DEFAULT_TEST_DIALOGUE_PATH) -> str:
        raw = self.load_text(dialogue_path)
        data = yaml.safe_load(raw) or {}
        dialogue = data.get("dialogue")
        if not isinstance(dialogue, str) or not dialogue.strip():
            raise ValueError("Test dialogue YAML must contain a non-empty 'dialogue' field.")
        return dialogue.strip()

    def _load_prompt_steps(self, prompt_path: str | Path) -> list[dict[str, Any]]:
        raw = self.load_text(prompt_path)
        data = yaml.safe_load(raw) or {}
        prompts = data.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("Prompt YAML must contain a non-empty 'prompts' list.")

        steps: list[dict[str, Any]] = []
        for item in prompts:
            if not isinstance(item, dict):
                continue

            step_id = item.get("id")
            prompt_text = item.get("prompt")
            response_type = str(item.get("response_type", "text")).strip().lower()
            model_name = item.get("model_name")
            temperature = item.get("temperature")
            max_tokens = item.get("max_tokens")
            top_p = item.get("top_p")
            top_k = item.get("top_k")
            thinking_mode = item.get("thinking_mode")
            base_url = item.get("base_url")

            if not isinstance(step_id, str) or not step_id.strip():
                raise ValueError("Each prompt step must have a non-empty string 'id'.")
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                raise ValueError(f"Prompt step '{step_id}' must have a non-empty 'prompt'.")
            if response_type not in {"text", "json"}:
                raise ValueError(
                    f"Prompt step '{step_id}' has invalid response_type '{response_type}'. "
                    "Use 'text' or 'json'."
                )

            steps.append(
                {
                    "id": step_id.strip(),
                    "prompt": prompt_text,
                    "response_type": response_type,
                    "model_name": model_name.strip() if isinstance(model_name, str) else "",
                    "temperature": float(temperature) if temperature is not None else None,
                    "max_tokens": int(max_tokens) if max_tokens is not None else None,
                    "top_p": float(top_p) if top_p is not None else None,
                    "top_k": int(top_k) if top_k is not None else None,
                    "thinking_mode": (
                        str(thinking_mode).strip().lower() if thinking_mode is not None else ""
                    ),
                    "base_url": base_url.strip() if isinstance(base_url, str) else "",
                }
            )

        if not steps:
            raise ValueError("No valid prompt steps were loaded from prompt YAML.")
        return steps

    @staticmethod
    def _serialize_prompt_value(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        if value is None:
            return ""
        return str(value)

    class _PromptContext(dict):
        def __missing__(self, key: str) -> str:
            return ""

    def _render_prompt(self, prompt_template: str, context: dict[str, Any]) -> str:
        serialized_context = self._PromptContext({
            key: self._serialize_prompt_value(value) for key, value in context.items()
        })
        return prompt_template.format_map(serialized_context)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
        return "_".join(filter(None, slug.split("_")))[:80] or "point"

    def _save_rendered_prompt(
        self,
        step_id: str,
        point_index: int,
        point: str,
        prompt: str,
    ) -> Path:
        point_slug = self._slugify(point)
        prompt_path = self.prompts_dir / f"{point_index:02d}_{step_id}_{point_slug}.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        logger.info("Saved rendered prompt to %s", prompt_path)
        return prompt_path

    def _save_model_response(
        self,
        step_id: str,
        point_index: int,
        point: str,
        response: str | dict[str, Any],
    ) -> Path:
        point_slug = self._slugify(point)
        if isinstance(response, dict):
            filename = f"{point_index:02d}_{step_id}_{point_slug}_response.json"
            response_text = json.dumps(response, ensure_ascii=False, indent=2)
        else:
            filename = f"{point_index:02d}_{step_id}_{point_slug}_response.md"
            response_text = response

        response_path = self.responses_dir / filename
        response_path.write_text(response_text, encoding="utf-8")
        logger.info("Saved model response to %s", response_path)
        return response_path

    def _resolve_step_config(self, step: dict[str, Any]) -> dict[str, Any]:
        model_name = step["model_name"] or self.model_name
        base_url = step["base_url"] or self.base_url
        temperature = step["temperature"] 
        max_tokens = step["max_tokens"] 
        top_p = step["top_p"]
        top_k = step["top_k"]
        thinking_mode = step["thinking_mode"] or self.thinking_mode

        if thinking_mode not in {"", "none", "enabled"}:
            raise ValueError(
                f"Invalid thinking_mode for step '{step['id']}': {thinking_mode}. "
                "Use 'none' or 'enabled'."
            )

        return {
            "model_name": model_name,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "thinking_mode": thinking_mode or "none",
        }

    def query_model(
        self,
        api_key: str,
        step: dict[str, Any],
        context: dict[str, Any],
        point: str,
        point_index: int,
    ) -> str | dict[str, Any]:
        step_id = step["id"]
        response_type = step["response_type"]
        step_config = self._resolve_step_config(step)

        prompt = self._render_prompt(step["prompt"], context)
        self._save_rendered_prompt(step_id, point_index, point, prompt)

        logger.info(
            "Sending request to %s for point %s: model=%s response_type=%s temperature=%s max_tokens=%s top_p=%s top_k=%s thinking_mode=%s",
            step_id,
            point_index,
            step_config["model_name"],
            response_type,
            step_config["temperature"],
            step_config["max_tokens"],
            step_config["top_p"],
            step_config["top_k"],
            step_config["thinking_mode"],
        )

        extra_body = None
        if (
            step_config["thinking_mode"] == "enabled"
            and step_config["model_name"] != "deepseek-reasoner"
        ):
            extra_body = {"thinking": {"type": "enabled"}}

        if step_config["top_k"] is not None:
            logger.info(
                "Step %s requested top_k=%s, but this parameter is not sent to DeepSeek OpenAI-compatible API",
                step_id,
                step_config["top_k"],
            )

        if response_type == "json":
            response = self.request_json(
                api_key=api_key,
                prompt=prompt,
                model_name=step_config["model_name"],
                base_url=step_config["base_url"],
                temperature=step_config["temperature"],
                max_tokens=step_config["max_tokens"],
                top_p=step_config["top_p"],
                extra_body=extra_body,
            )
        else:
            response = self.request_text(
                api_key=api_key,
                prompt=prompt,
                model_name=step_config["model_name"],
                base_url=step_config["base_url"],
                temperature=step_config["temperature"],
                max_tokens=step_config["max_tokens"],
                top_p=step_config["top_p"],
                extra_body=extra_body,
            )

        self._save_model_response(step_id, point_index, point, response)
        logger.info("Received response from %s for point %s", step_id, point_index)
        return response

    def evaluate_point(
        self,
        api_key: str,
        dialogue: str,
        point: str,
        evaluated_speaker: str,
        point_index: int = 1,
    ) -> dict[str, Any]:
        logger.info("Evaluating point %s: %s", point_index, point)

        context: dict[str, Any] = {
            "dialogue": dialogue,
            "point": point,
            "evaluated_speaker": evaluated_speaker,
            "point_index": point_index,
            "previous_results": {},
            "last_result": "",
        }
        step_results: dict[str, str | dict[str, Any]] = {}
        final_response: str | dict[str, Any] | None = None

        for step in self.prompt_steps:
            response = self.query_model(
                api_key=api_key,
                step=step,
                context=context,
                point=point,
                point_index=point_index,
            )
            step_id = step["id"]
            step_results[step_id] = response
            context["previous_results"] = step_results
            context["last_result"] = response
            context[f"{step_id}_result"] = response
            final_response = response

        if isinstance(final_response, dict):
            result = dict(final_response)
            result.setdefault("point", point)
            return result

        return {
            "point": point,
            "label": "unknown",
            "evidence": "",
            "rationale": str(final_response or ""),
            "confidence": 0.0,
        }

    def evaluate_points(
        self,
        api_key: str,
        dialogue: str,
        points: list[str],
        evaluated_speaker: str,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
    ) -> Path:
        logger.info("Starting point-agents run for %s points", len(points))
        point_results = [
            self.evaluate_point(
                api_key=api_key,
                dialogue=dialogue,
                point=point,
                evaluated_speaker=evaluated_speaker,
                point_index=index,
            )
            for index, point in enumerate(points, start=1)
        ]

        result = {
            "evaluated_speaker": evaluated_speaker,
            "points": point_results,
            "present_count": sum(1 for item in point_results if item.get("label") == "present"),
            "absent_count": sum(1 for item in point_results if item.get("label") == "absent"),
            "opposite_count": sum(1 for item in point_results if item.get("label") == "opposite"),
        }
        saved_path = self.save_json(result, output_path)
        logger.info("Saved point-agents response to %s", saved_path)
        return saved_path
    
    def evaluate_criterion(
        self,
        api_key: str,
        dialogue: str,
        criterion_path: str | Path,
        evaluated_speaker: str,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
    ) -> Path:
        points = self.load_points_from_criterion(criterion_path)

        return self.evaluate_points(
            api_key=api_key,
            dialogue=dialogue,
            points=points,
            evaluated_speaker=evaluated_speaker,
            output_path=output_path,
        )

    def run(
        self,
        api_key: str,
        dialogue: str,
        criterion_path: str | Path,
        evaluated_speaker: str,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
    ) -> Path:
        return self.evaluate_criterion(
            api_key=api_key,
            dialogue=dialogue,
            criterion_path=criterion_path,
            evaluated_speaker=evaluated_speaker,
            output_path=output_path,
        )


if __name__ == "__main__":
    try:
        load_dotenv(PROJECT_ROOT / "config" / ".env")
        load_dotenv(PROJECT_ROOT / "config" / "model.env")
        load_dotenv(PROJECT_ROOT / "config" / "point_model.env")

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
        output_path = os.getenv("POINT_AGENTS_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)
        criterion_path = os.getenv("POINT_AGENTS_CRITERION_PATH") or os.getenv("CRETERIONS_PATH", "")
        prompt_path = os.getenv("POINT_AGENTS_PROMPTS_PATH", str(DEFAULT_PROMPTS_PATH))
        test_dialogue_path = os.getenv("TEST_DIALOGUE_PATH", DEFAULT_TEST_DIALOGUE_PATH)
        evaluated_speaker = os.getenv("EVALUATED_SPEAKER", "B")

        if not criterion_path:
            raise ValueError(
                "Criterion path is empty. Set POINT_AGENTS_CRITERION_PATH or CRETERIONS_PATH."
            )

        model = PointAgentsModel(
            prompt_path=prompt_path,
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
        saved_path = model.evaluate_criterion(
            api_key=api_key,
            dialogue=model.load_test_dialogue(test_dialogue_path),
            criterion_path=criterion_path,
            evaluated_speaker=evaluated_speaker,
            output_path=output_path,
        )
        logger.info("Saved point-agents response to %s", saved_path)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
