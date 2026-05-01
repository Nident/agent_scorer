import os
import argparse
import json
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
from scripts.evaluate_responses import evaluate_dir
from scripts.prepare_anli_benchmark import DEFAULT_INPUT, DEFAULT_OUTPUT, write_benchmark


MODELS = {"simple": SimpleQueryModel, "points": PointQueryModel}
BENCHMARK_VARIANTS = {
    "simple": {
        "model_type": "simple",
        "prompt_path": "data/prompts/simple_model.yaml",
    },
    "points_single": {
        "model_type": "points",
        "prompt_path": "data/prompts/points_single_model.yaml",
    },
    "points_verify": {
        "model_type": "points",
        "prompt_path": "data/prompts/points_verify_model.yaml",
    },
    "points_three": {
        "model_type": "points",
        "prompt_path": "data/prompts/points_model.yaml",
    },
    "points_three_consensus": {
        "model_type": "points",
        "prompt_path": "data/prompts/points_model.yaml",
        "use_consensus": True,
    },
}


def _resolve_prompt_path(model_type: str) -> Path:
    env_prompt = os.getenv("PROMPT_PATH", "").strip()
    default_prompt = f"data/prompts/{model_type}_model.yaml"
    if not env_prompt:
        return resolve_project_path(default_prompt)

    candidate = resolve_project_path(env_prompt)
    if candidate.exists():
        return candidate

    return resolve_project_path(default_prompt)


def run_analysis_from_env() -> Path:
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
    return saved_path


def _iter_benchmark_pairs(generated_dir: Path, case_id: str = ""):
    index_path = generated_dir / "index.jsonl"
    if index_path.exists():
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            criterion_path = resolve_project_path(row["criterion_path"])
            dialogue_path = resolve_project_path(row["dialogue_path"])
            if case_id and case_id not in criterion_path.stem:
                continue
            yield criterion_path, dialogue_path
        return

    criteria = sorted((generated_dir / "criteria").glob("*.yaml"))
    dialogues = sorted((generated_dir / "dialogues").glob("*.json"))
    dialogue_by_stem = {path.stem: path for path in dialogues}
    for criterion_path in criteria:
        if case_id and case_id not in criterion_path.stem:
            continue
        dialogue_path = dialogue_by_stem.get(criterion_path.stem)
        if dialogue_path is None:
            raise RuntimeError(f"Missing dialogue for {criterion_path.name}")
        yield criterion_path, dialogue_path


def _run_one_case(
    *,
    variant_name: str,
    variant: dict,
    criterion_path: Path,
    dialogue_path: Path,
    output_path: Path,
    api_key: str,
    run_llm: bool,
    dialogue_block_size: int,
    evaluated_speaker: str,
) -> Path:
    model_type = variant["model_type"]
    prompt_path = resolve_project_path(variant["prompt_path"])
    steps = load_steps(prompt_path, f"{model_type}_model")
    dialogue = load_dialogue(dialogue_path)
    skip_predict = not run_llm

    summary_module = DialogueSummaryModule()
    summary = summary_module.run(
        context={
            "api_key": api_key,
            "dialogue": dialogue,
            "evaluated_speaker": evaluated_speaker,
            "dialogue_block_size": dialogue_block_size,
            "skip_predict": True,
            "summary_skip_predict": True,
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
        "points_use_consensus": bool(variant.get("use_consensus", False)),
    }

    model = MODELS[model_type]()
    result = model.run(steps=steps, context=context)
    return model.save_response(
        result,
        output_path,
        f"{variant_name}_model",
        metadata={
            "criterion_path": str(criterion_path),
            "criterion_file": criterion_path.name,
            "prompt_path": str(prompt_path),
            "prompt_file": prompt_path.name,
            "dialogue_path": str(dialogue_path),
            "dialogue_file": dialogue_path.name,
            "benchmark_variant": variant_name,
            "summary_mode": summary.get("mode", ""),
        },
    )


def run_benchmark(args) -> int:
    model_env = resolve_project_path(args.model_env)
    if model_env.exists():
        load_dotenv(model_env, override=True)
    else:
        print(f"Model env not found, using current environment: {model_env}")

    generated_dir = resolve_project_path(args.generated_dir)
    stats = write_benchmark(
        input_path=resolve_project_path(args.dataset),
        output_dir=generated_dir,
        limit=args.limit,
        overwrite=args.prepare_overwrite,
    )
    print("Prepared benchmark:", json.dumps(stats, ensure_ascii=False))

    if args.prepare_only:
        return 0

    api_key = os.getenv("API_KEY", "")
    if args.run_llm and not api_key:
        raise RuntimeError(
            "API_KEY is empty. Put it into --model-env or environment, or run without --run-llm for a dry run."
        )

    if args.variant == "all":
        variant_names = list(BENCHMARK_VARIANTS)
    else:
        variant_names = [args.variant]

    pairs = list(_iter_benchmark_pairs(generated_dir, args.case_id))
    if args.case_limit is not None:
        pairs = pairs[: args.case_limit]

    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports = {}

    for variant_name in variant_names:
        variant = BENCHMARK_VARIANTS[variant_name]
        responses_dir = output_root / variant_name
        responses_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{variant_name}] cases={len(pairs)} run_llm={args.run_llm}")
        for index, (criterion_path, dialogue_path) in enumerate(pairs, start=1):
            output_path = responses_dir / f"{criterion_path.stem}.json"
            if output_path.exists() and not args.overwrite:
                print(f"[{variant_name}] {index}/{len(pairs)} skip {output_path.name}")
                continue

            print(f"[{variant_name}] {index}/{len(pairs)} run {criterion_path.stem}")
            _run_one_case(
                variant_name=variant_name,
                variant=variant,
                criterion_path=criterion_path,
                dialogue_path=dialogue_path,
                output_path=output_path,
                api_key=api_key,
                run_llm=args.run_llm,
                dialogue_block_size=args.dialogue_block_size,
                evaluated_speaker=args.evaluated_speaker,
            )

        report = evaluate_dir(responses_dir, variant["model_type"])
        report_path = output_root / f"{variant_name}_metrics.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports[variant_name] = {
            "files": report["files"],
            "errors": len(report["errors"]),
            "problem_summary": report["problem_summary"],
            "overall": report["overall"],
            "metrics_path": str(report_path),
        }

    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--run-llm", action="store_true")
    benchmark.add_argument("--model-env", default="config/model.env")
    benchmark.add_argument("--dataset", default=str(DEFAULT_INPUT))
    benchmark.add_argument("--generated-dir", default=str(DEFAULT_OUTPUT))
    benchmark.add_argument("--output-root", default="data/responses/benchmark")
    benchmark.add_argument("--limit", type=int, default=100, help="Number of premise groups to prepare")
    benchmark.add_argument("--case-limit", type=int, default=None, help="Number of prepared cases to run")
    benchmark.add_argument("--case-id", default="")
    benchmark.add_argument("--variant", choices=[*BENCHMARK_VARIANTS.keys(), "all"], default="all")
    benchmark.add_argument("--dialogue-block-size", type=int, default=6)
    benchmark.add_argument("--evaluated-speaker", default="B")
    benchmark.add_argument("--overwrite", action="store_true")
    benchmark.add_argument("--prepare-overwrite", action="store_true")
    benchmark.add_argument("--prepare-only", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()

    if parsed_args.command == "benchmark":
        raise SystemExit(run_benchmark(parsed_args))

    load_dotenv(resolve_project_path("config/.env"))
    run_analysis_from_env()
