import json
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Model:
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
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.thinking_mode = thinking_mode
        self.max_retries = max_retries

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def load_prompt(self, prompt_path: str | Path) -> str:
        path = self._resolve_path(prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file was not found: {path}")

        return path.read_text(encoding="utf-8")

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
        return (
            f"{prompt}\n\n"
            "Your previous response was not valid JSON.\n"
            "Return the same answer again, but fix it so it is one valid JSON object only.\n"
            "Do not add markdown, comments, explanations, or code fences.\n\n"
            f"JSON validation error:\n{error}\n\n"
            f"Previous invalid response:\n{raw_response}"
        )

    def simple_LLM_query(
        self,
        api_key: str,
        prompt_path: str | Path,
        output_path: str | Path = "data/llm_response.json",
    ) -> Path:
        if not api_key:
            raise ValueError("API_KEY is empty. Add API_KEY=... to config/model.env")

        os.environ.setdefault("USE_TORCH", "0")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        try:
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install dependencies first: python3 -m pip install -r ../requirements.txt"
            ) from exc

        prompt = self.load_prompt(prompt_path)

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=SecretStr(api_key),
            base_url=self.base_url,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        current_prompt = prompt
        raw_answer = ""
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            if self.thinking_mode == "enabled" and self.model_name != "deepseek-reasoner":
                answer = llm.invoke(
                    current_prompt,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "enabled"}},
                )
            else:
                answer = llm.invoke(
                    current_prompt,
                    response_format={"type": "json_object"},
                )

            raw_answer = answer.content if isinstance(answer.content, str) else str(answer.content)

            try:
                json_answer = self.validate_json(raw_answer)
                return self.save_json(json_answer, output_path)
            except ValueError as exc:
                last_error = str(exc)
                if attempt == self.max_retries:
                    break
                current_prompt = self.build_retry_prompt(prompt, raw_answer, last_error)

        raise ValueError(
            f"LLM response is not valid JSON after {self.max_retries} attempts. "
            f"Last error: {last_error}. Last response: {raw_answer}"
        )


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


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / "config" / "model.env")

    prompt_path = os.getenv("PROMPT_INPUT_PATH", "data/assembled_prompt.md")
    response_path = os.getenv("LLM_RESPONSE_PATH", "data/llm_response.json")
    api_key = os.getenv("API_KEY", "")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com")
    temperature = read_optional_float("TEMPERATURE")
    max_tokens = read_optional_int("MAX_TOKENS")
    top_p = read_optional_float("TOP_P")
    top_k = read_optional_int("TOP_K")
    thinking_mode = read_thinking_mode()
    max_retries = read_optional_int("RESPONSE_MAX_RETRIES") or 3

    saved_path = Model(
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        thinking_mode=thinking_mode,
        max_retries=max_retries,
    ).simple_LLM_query(
        api_key=api_key,
        prompt_path=prompt_path,
        output_path=response_path,
    )
    print(f"Saved LLM response to {saved_path}")
