from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _prepare_runtime_env() -> None:
    os.environ.setdefault("USE_TORCH", "0")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


_prepare_runtime_env()


class Model(ABC):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    @staticmethod
    def load_model(model_name, base_url, api_key, temperature, max_tokens, top_p):
        _prepare_runtime_env()

        try:
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Dependencies are not installed. Run: python3 -m pip install -r requirements.txt"
            ) from exc

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=SecretStr(api_key) if api_key else None,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
        )

    @staticmethod
    def save_response(
        data: dict,
        output_path: str | Path,
        model_type: str,
        metadata: dict | None = None,
    ) -> Path:
        path = Path(output_path)
        if not path.is_absolute():
            path = ROOT_DIR / path

        if path.suffix.lower() != ".json":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path / f"{model_type}_response_{timestamp}.json"

        payload = dict(data)
        payload.setdefault("metadata", {})
        payload["metadata"].update(
            {
                "model_type": model_type,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if metadata:
            payload["metadata"].update(metadata)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @abstractmethod
    def predict(self, input_data):
        raise NotImplementedError
