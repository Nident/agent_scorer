from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from .model import Model
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.model import Model

try:
    from utils import dialogue_to_text, to_bool
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils import dialogue_to_text, to_bool


class DialogueSummaryModule(Model):
    """Builds per-block dialogue summaries for downstream scoring and UI output."""

    DEFAULT_PROMPT = """You are a negotiation dialogue summarizer.

Summarize the current dialogue block for a negotiation readiness assessment.
Use the previous summary only as context. Do not invent facts.

Return one valid JSON object only:
{{
  "summary": "2-3 concise sentences about the negotiation progress in this block",
  "events": [
    {{"speaker": "speaker label", "text": "important action, question, concession, objection, or agreement"}}
  ],
  "buyer_readiness": "short note or empty string",
  "seller_readiness": "short note or empty string"
}}

Previous summary:
{previous_summary}

Current dialogue block:
{dialogue_block}
"""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "")
        kwargs.setdefault("base_url", "")
        kwargs.setdefault("api_key", "")
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", 700)
        kwargs.setdefault("top_p", 1.0)
        super().__init__(**kwargs)
        self.llm: Any | None = None

    def predict(self, input_data):
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Call _apply_context_config(...) before predict().")
        answer = self.llm.invoke(input_data["prompt"])
        return answer.content if isinstance(answer.content, str) else str(answer.content)

    def _apply_context_config(self, context: dict, initialize_llm: bool) -> None:
        self.model_name = (
            context.get("summary_model_name")
            or os.getenv("SUMMARY_MODEL_NAME")
            or context.get("model_name")
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-chat"
        )
        self.base_url = (
            context.get("summary_base_url")
            or os.getenv("SUMMARY_BASE_URL")
            or context.get("base_url")
            or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        )
        self.temperature = float(context.get("summary_temperature", os.getenv("SUMMARY_TEMPERATURE", 0.0)))
        self.max_tokens = int(context.get("summary_max_tokens", os.getenv("SUMMARY_MAX_TOKENS", 700)))
        self.top_p = float(context.get("summary_top_p", os.getenv("SUMMARY_TOP_P", 1.0)))
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

    @staticmethod
    def _trim(value: str, limit: int = 220) -> str:
        normalized = " ".join(str(value).split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "..."

    @classmethod
    def _extract_events(cls, dialogue_block: str, limit: int = 5) -> list[dict[str, str]]:
        events: list[dict[str, str]] = []
        for line in dialogue_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if ":" in stripped:
                speaker, text = stripped.split(":", 1)
                events.append(
                    {
                        "speaker": cls._trim(speaker, 40),
                        "text": cls._trim(text, 180),
                    }
                )
            else:
                events.append({"speaker": "", "text": cls._trim(stripped, 180)})

            if len(events) >= limit:
                break
        return events

    @classmethod
    def _fallback_summary(cls, dialogue_block: str, block_index: int) -> dict[str, Any]:
        events = cls._extract_events(dialogue_block)
        if not events:
            return {
                "summary": f"Блок {block_index}: реплики не найдены.",
                "events": [],
                "buyer_readiness": "",
                "seller_readiness": "",
            }

        joined_events = " ".join(
            cls._trim(f"{event['speaker']}: {event['text']}".strip(": "), 150)
            for event in events[:3]
            if event.get("text")
        )
        return {
            "summary": cls._trim(f"Блок {block_index}: {joined_events}", 420),
            "events": events,
            "buyer_readiness": "",
            "seller_readiness": "",
        }

    @staticmethod
    def _parse_json(raw_response: str) -> dict[str, Any] | None:
        cleaned = raw_response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _combined_summary(blocks: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"Блок {block['block_index']}: {block.get('summary', '')}".strip()
            for block in blocks
            if block.get("summary")
        )

    def run(self, *, context: dict, block_size: int | None = None) -> dict[str, Any]:
        if block_size is None:
            block_size = int(context.get("dialogue_block_size", 6))
        if block_size <= 0:
            block_size = 1

        dialogue_blocks = list(dialogue_to_text(context.get("dialogue", ""), block_size=block_size))
        if not dialogue_blocks:
            dialogue_blocks = [""]

        summary_skip_predict = to_bool(
            context.get("summary_skip_predict", context.get("skip_predict", True)),
            default=True,
        )
        can_call_llm = not summary_skip_predict and bool(context.get("api_key"))
        self._apply_context_config(context, initialize_llm=can_call_llm)

        summary_blocks: list[dict[str, Any]] = []
        previous_summary = ""

        for block_index, dialogue_block in enumerate(dialogue_blocks, start=1):
            prompt = self.DEFAULT_PROMPT.format(
                previous_summary=previous_summary,
                dialogue_block=dialogue_block,
            )
            raw_response = ""

            if can_call_llm:
                raw_response = self.predict({"prompt": prompt})
                parsed = self._parse_json(raw_response) or {
                    "summary": self._trim(raw_response, 420),
                    "events": self._extract_events(dialogue_block),
                    "buyer_readiness": "",
                    "seller_readiness": "",
                }
            else:
                parsed = self._fallback_summary(dialogue_block, block_index)

            summary = self._trim(str(parsed.get("summary", "")).strip(), 620)
            events = parsed.get("events")
            if not isinstance(events, list):
                events = self._extract_events(dialogue_block)

            summary_blocks.append(
                {
                    "block_index": block_index,
                    "dialogue_block": dialogue_block,
                    "summary": summary,
                    "events": events,
                    "buyer_readiness": str(parsed.get("buyer_readiness", "") or ""),
                    "seller_readiness": str(parsed.get("seller_readiness", "") or ""),
                    "prompt": prompt if can_call_llm else "",
                    "raw_response": raw_response,
                    "skipped_predict": not can_call_llm,
                }
            )
            previous_summary = self._combined_summary(summary_blocks)

        return {
            "block_size": block_size,
            "mode": "llm" if can_call_llm else "extractive",
            "skip_predict": not can_call_llm,
            "blocks": summary_blocks,
            "combined_summary": self._combined_summary(summary_blocks),
        }
