from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
import logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"


class Model(ABC):
    def __init__(
        self,
        model_name: str = "",
        base_url: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        thinking_mode: str = "none",
        max_retries: int = 3,
        request_timeout: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.thinking_mode = thinking_mode
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def load_text(self, path: str | Path) -> str:
        resolved = self._resolve_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"File was not found: {resolved}")
        return resolved.read_text(encoding="utf-8")

    def load_prompt(self, prompt_path: str | Path) -> str:
        return self.load_text(prompt_path)

    def load_dialogue_block(self, path: str | Path) -> str:
        resolved = self._resolve_path(path)
        if resolved.suffix.lower() != ".json":
            return self.load_text(resolved)

        raw = self.load_text(resolved)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw

        if not isinstance(data, dict):
            return raw

        transcript = data.get("transcript")
        if not isinstance(transcript, dict):
            return raw

        utterances = transcript.get("utterances")
        if not isinstance(utterances, list):
            return raw

        lines: list[str] = []
        for utterance in utterances:
            if not isinstance(utterance, dict):
                continue
            speaker = str(utterance.get("speaker_label") or "").strip()
            text = str(utterance.get("text") or "").strip()
            if not speaker or not text:
                continue
            lines.append(f"{speaker}: {text}")

        if not lines:
            raise ValueError(f"No dialogue utterances were found in transcript JSON: {resolved}")

        return "\n".join(lines)

    def compose_prompt(self, *sections: str) -> str:
        return "\n\n".join(section.strip() for section in sections if section and section.strip())

    def validate_json(self, raw_response: str) -> dict:
        response = raw_response.strip()

        if response.startswith("```"):
            lines = response.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response is not valid JSON.") from exc

        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object.")

        return data

    def save_json(self, data: dict, output_path: str | Path) -> Path:
        path = self._resolve_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def build_retry_prompt(self, prompt: str, raw_response: str, error: str) -> str:
        return self.compose_prompt(
            prompt,
            (
                "Your previous response was not valid JSON.\n"
                "Return the same answer again, but fix it so it is one valid JSON object only.\n"
                "Do not add markdown, comments, explanations, or code fences.\n\n"
                f"JSON validation error:\n{error}\n\n"
                f"Previous invalid response:\n{raw_response}"
            ),
        )

    def _prepare_runtime_env(self) -> None:
        os.environ.setdefault("USE_TORCH", "0")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    def _build_llm(
        self,
        api_key: str,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ):
        if not api_key:
            raise ValueError("API_KEY is empty. Add API_KEY=... to config/model.env")

        self._prepare_runtime_env()

        try:
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install dependencies first: python -m pip install -r requirements.txt"
            ) from exc

        resolved_model_name = model_name or self.model_name
        llm = ChatOpenAI(
            model=resolved_model_name,
            api_key=SecretStr(api_key),
            base_url=base_url or self.base_url,
            timeout=self.request_timeout,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
        )

        bind_kwargs = {}
        if response_format is not None:
            bind_kwargs["response_format"] = response_format
        if extra_body is not None:
            bind_kwargs["extra_body"] = extra_body
        if bind_kwargs:
            llm = llm.bind(**bind_kwargs)
        return llm

    def request_text(
        self,
        api_key: str,
        prompt: str,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        logging.getLogger(__name__).info(
            "Sending LLM request: model=%s temperature=%s max_tokens=%s top_p=%s thinking_mode=%s",
            model_name or self.model_name,
            temperature,
            max_tokens,
            top_p,
            self.thinking_mode,
        )
        llm = self._build_llm(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_body=extra_body,
            response_format=response_format,
        )
        answer = llm.invoke(prompt)
        return answer.content if isinstance(answer.content, str) else str(answer.content)

    def request_json(
        self,
        api_key: str,
        prompt: str,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
    ) -> dict:
        current_prompt = prompt
        raw_answer = ""
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            logging.getLogger(__name__).info(
                "Requesting JSON from LLM: attempt=%s/%s model=%s",
                attempt,
                self.max_retries,
                model_name or self.model_name,
            )
            raw_answer = self.request_text(
                api_key=api_key,
                prompt=current_prompt,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_body=extra_body,
                response_format={"type": "json_object"},
            )
            try:
                return self.validate_json(raw_answer)
            except ValueError as exc:
                last_error = str(exc)
                logging.getLogger(__name__).warning(
                    "Invalid JSON from LLM on attempt %s/%s: %s",
                    attempt,
                    self.max_retries,
                    last_error,
                )
                if attempt == self.max_retries:
                    break
                current_prompt = self.build_retry_prompt(prompt, raw_answer, last_error)

        raise ValueError(
            f"LLM response is not valid JSON after {self.max_retries} attempts. "
            f"Last error: {last_error}. Last response: {raw_answer}"
        )

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


def read_optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return float(value)


def read_optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return int(value)


def read_thinking_mode() -> str:
    thinking_mode = os.getenv("THINKING_MODE", "none").strip().lower().replace(" ", "_")
    allowed_modes = {"none", "enabled"}
    if thinking_mode not in allowed_modes:
        raise ValueError("THINKING_MODE must be 'none' or 'enabled' for DeepSeek API.")
    return thinking_mode


def read_optional_timeout(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return float(value)


def configure_file_logger(logger_name: str, log_filename: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = LOGS_DIR / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_log_path = str(log_path.resolve())
    has_file_handler = False
    has_stream_handler = False

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved_log_path:
            has_file_handler = True
        elif isinstance(handler, logging.StreamHandler):
            has_stream_handler = True

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not has_file_handler:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
