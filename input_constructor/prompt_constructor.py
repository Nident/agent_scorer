from pathlib import Path
from textwrap import dedent

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PromptConstructor:
    def __init__(
        self,
        dialogue_block: str,
        criterion_path: str = "",
        prompt_template_path: str = "",
        evaluated_speaker: str = "B",
    ) -> None:
        self.dialogue_block = dialogue_block
        self.criterion_path = self._resolve_path(criterion_path)
        self.evaluated_speaker = evaluated_speaker
        self.prompt_template_path = self._resolve_path(prompt_template_path)

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def load_criterion(self) -> str:
        if not self.criterion_path.is_file():
            raise FileNotFoundError(f"Criterion file was not found: {self.criterion_path}")

        return self.criterion_path.read_text(encoding="utf-8")

    def load_template(self) -> str:
        raw = self.prompt_template_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        try:
            return dedent(data["prompt"]).strip()
        except (TypeError, KeyError) as exc:
            raise ValueError("Prompt template YAML must contain a 'prompt' field.") from exc

    def build_prompt(self) -> str:
        return self.load_template().format(
            evaluated_speaker=self.evaluated_speaker,
            criterion=self.load_criterion(),
            dialogue_block=self.dialogue_block,
        )

    def save_prompt(self, output_path: str = "data/assembled_prompt.md") -> Path:
        path = self._resolve_path(output_path)
        if path.suffix == "":
            path = path / "assembled_prompt.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build_prompt(), encoding="utf-8")
        return path


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    ENV = PROJECT_ROOT / "config" / ".env"
    load_dotenv(ENV)



    demo = """
    A: What matters most for you in this agreement?
    B: If you commit to a twelve-month term, we can keep the current price.
    A: We need a shorter term.
    B: I understand. Then we can discuss six months, but the price will change.
    """

    criterion_path = os.getenv("CRETERIONS_PATH", "")
    prompt_template_path = os.getenv("TEMPLATE_PATH", "")
    prompt_save_path = os.getenv("PROMPT_SAVE_PATH", "data/assembled_prompt.md")

    pconst = PromptConstructor(
        dialogue_block=demo,
        criterion_path=criterion_path,
        prompt_template_path=prompt_template_path,
        evaluated_speaker="B",
    )

    saved_path = pconst.save_prompt(prompt_save_path)
    print(f"Saved prompt to {saved_path}")
