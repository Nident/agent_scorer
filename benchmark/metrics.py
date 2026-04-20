from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import Any, Iterable

from benchmark.dataset import BenchmarkCase


@dataclass(frozen=True)
class PredictionRecord:
    case_id: str
    prediction: dict[str, Any] | None
    valid_json: bool = True
    error: str = ""


@dataclass(frozen=True)
class MetricResult:
    key: str
    title: str
    value: float | None
    better: str
    explanation: str


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    title: str
    role: str
    criterion: str
    valid_json: bool
    schema_valid: bool
    error: str
    expected_score_0_10: float
    predicted_score_0_10: float | None
    absolute_error_0_10: float | None
    expected_score_raw: int
    predicted_score_raw: int | None
    expected_point_indices: tuple[int, ...]
    predicted_point_indices: tuple[int, ...]
    true_positive_points: int
    false_positive_points: int
    false_negative_points: int
    point_f1: float | None
    quote_total: int
    quote_supported: int


@dataclass(frozen=True)
class BenchmarkReport:
    generated_at: str
    case_count: int
    metrics: tuple[MetricResult, ...]
    cases: tuple[CaseEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "case_count": self.case_count,
            "metrics": [asdict(metric) for metric in self.metrics],
            "cases": [asdict(case) for case in self.cases],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Benchmark report",
            "",
            f"Generated at: {self.generated_at}",
            f"Cases: {self.case_count}",
            "",
            "## Metrics",
            "",
            "| Metric | Value | How to read it |",
            "| --- | ---: | --- |",
        ]
        for metric in self.metrics:
            lines.append(
                "| "
                f"{metric.title} | "
                f"{_format_value(metric.value)} | "
                f"{metric.explanation} "
                f"Направление: {metric.better}. |"
            )

        lines.extend(
            [
                "",
                "## Cases",
                "",
                "| Case | Expected | Predicted | Error | Expected points | Predicted points | Quote support |",
                "| --- | ---: | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for case in self.cases:
            lines.append(
                "| "
                f"{case.case_id} | "
                f"{case.expected_score_0_10:.2f} | "
                f"{_format_value(case.predicted_score_0_10)} | "
                f"{_format_value(case.absolute_error_0_10)} | "
                f"{_format_indices(case.expected_point_indices)} | "
                f"{_format_indices(case.predicted_point_indices)} | "
                f"{case.quote_supported}/{case.quote_total} |"
            )
        return "\n".join(lines) + "\n"


METRIC_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "json_validity_rate": (
        "JSON validity",
        "Доля ответов, которые удалось прочитать как один JSON-объект. "
        "Метрика ловит технические срывы формата: markdown вместо JSON, битый JSON, пустой ответ.",
        "выше лучше",
    ),
    "schema_validity_rate": (
        "Schema validity",
        "Доля ответов, где есть необходимые поля для оценки: балл и выбранные пункты критерия. "
        "Если JSON валиден, но модель не вернула структуру, дальнейшие метрики становятся менее надежными.",
        "выше лучше",
    ),
    "score_mae_0_10": (
        "MAE score 0-10",
        "Средняя абсолютная ошибка балла на шкале 0-10. Например, 0.8 значит, что в среднем модель ошибается "
        "меньше чем на один пункт шкалы.",
        "ниже лучше",
    ),
    "score_rmse_0_10": (
        "RMSE score 0-10",
        "Корень из средней квадратичной ошибки на шкале 0-10. Сильнее штрафует редкие крупные промахи, "
        "поэтому полезен рядом с MAE.",
        "ниже лучше",
    ),
    "score_within_1_accuracy": (
        "Within 1 point",
        "Доля кейсов, где прогнозный балл отличается от эталона не больше чем на 1 пункт из 10. "
        "Практичная метрика для переговорной оценки, где соседние уровни часто близки по смыслу.",
        "выше лучше",
    ),
    "raw_score_exact_accuracy": (
        "Exact raw score",
        "Доля кейсов, где модель угадала точное число найденных пунктов критерия. "
        "Это строгая метрика: ошибка на один подпункт уже считается промахом.",
        "выше лучше",
    ),
    "point_micro_precision": (
        "Point precision",
        "Среди всех пунктов, которые модель отметила как выполненные, доля действительно выполненных. "
        "Низкое значение означает, что модель приписывает участнику лишнюю подготовку.",
        "выше лучше",
    ),
    "point_micro_recall": (
        "Point recall",
        "Среди всех реально выполненных пунктов критерия доля найденных моделью. "
        "Низкое значение означает, что модель пропускает признаки подготовки.",
        "выше лучше",
    ),
    "point_micro_f1": (
        "Point F1",
        "Гармоническое среднее precision и recall по пунктам критерия. "
        "Показывает общий баланс между лишними срабатываниями и пропусками.",
        "выше лучше",
    ),
    "point_macro_f1": (
        "Case macro F1",
        "Средний F1 по кейсам. В отличие от micro F1, каждый диалог имеет одинаковый вес, "
        "даже если в одном кейсе найдено больше пунктов, чем в другом.",
        "выше лучше",
    ),
    "evidence_quote_support_rate": (
        "Quote support",
        "Доля цитат-доказательств, которые дословно встречаются в диалоге. "
        "Метрика не проверяет смысл цитаты, но ловит галлюцинации и выдуманные доказательства.",
        "выше лучше",
    ),
    "spearman_score_correlation": (
        "Spearman correlation",
        "Ранговая корреляция между эталонными и прогнозными баллами. "
        "Показывает, правильно ли модель упорядочивает участников от слабой подготовки к сильной.",
        "выше лучше",
    ),
}


def compute_report(
    cases: Iterable[BenchmarkCase],
    predictions: Iterable[PredictionRecord],
) -> BenchmarkReport:
    case_list = list(cases)
    records_by_id = {record.case_id: record for record in predictions}
    case_evaluations = tuple(
        _evaluate_case(case, records_by_id.get(case.id)) for case in case_list
    )
    metrics = tuple(_compute_metrics(case_evaluations))

    return BenchmarkReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        case_count=len(case_evaluations),
        metrics=metrics,
        cases=case_evaluations,
    )


def _evaluate_case(
    case: BenchmarkCase,
    record: PredictionRecord | None,
) -> CaseEvaluation:
    if record is None:
        return _missing_case(case)

    prediction = record.prediction if isinstance(record.prediction, dict) else None
    valid_json = record.valid_json and prediction is not None
    if not valid_json:
        return _missing_case(case, record.error or "Prediction is not a JSON object.")

    predicted_score_raw = _extract_int(prediction, ("score_raw", "score"))
    max_score = _extract_int(prediction, ("max_score",)) or case.expected.max_score
    predicted_score_0_10 = _extract_float(
        prediction,
        ("score_0_10", "normalized_score", "score_normalized"),
    )
    if predicted_score_0_10 is None and predicted_score_raw is not None and max_score:
        predicted_score_0_10 = round((predicted_score_raw / max_score) * 10, 4)

    predicted_indices, has_point_signal = _extract_point_indices(prediction, case)
    expected_indices = set(case.expected.matched_point_indices)
    predicted_set = set(predicted_indices)

    true_positive = len(expected_indices & predicted_set)
    false_positive = len(predicted_set - expected_indices)
    false_negative = len(expected_indices - predicted_set)
    point_f1 = _f1(true_positive, false_positive, false_negative)

    quote_total, quote_supported = _quote_support(prediction, case.dialogue)
    schema_valid = predicted_score_0_10 is not None and has_point_signal

    return CaseEvaluation(
        case_id=case.id,
        title=case.title,
        role=case.role,
        criterion=case.criterion.name,
        valid_json=True,
        schema_valid=schema_valid,
        error="" if schema_valid else "Missing score or matched point fields.",
        expected_score_0_10=case.expected.score_0_10,
        predicted_score_0_10=predicted_score_0_10,
        absolute_error_0_10=(
            abs(predicted_score_0_10 - case.expected.score_0_10)
            if predicted_score_0_10 is not None
            else None
        ),
        expected_score_raw=case.expected.score_raw,
        predicted_score_raw=predicted_score_raw,
        expected_point_indices=case.expected.matched_point_indices,
        predicted_point_indices=tuple(sorted(predicted_set)),
        true_positive_points=true_positive,
        false_positive_points=false_positive,
        false_negative_points=false_negative,
        point_f1=point_f1 if has_point_signal else None,
        quote_total=quote_total,
        quote_supported=quote_supported,
    )


def _missing_case(case: BenchmarkCase, error: str = "Missing prediction.") -> CaseEvaluation:
    return CaseEvaluation(
        case_id=case.id,
        title=case.title,
        role=case.role,
        criterion=case.criterion.name,
        valid_json=False,
        schema_valid=False,
        error=error,
        expected_score_0_10=case.expected.score_0_10,
        predicted_score_0_10=None,
        absolute_error_0_10=None,
        expected_score_raw=case.expected.score_raw,
        predicted_score_raw=None,
        expected_point_indices=case.expected.matched_point_indices,
        predicted_point_indices=(),
        true_positive_points=0,
        false_positive_points=0,
        false_negative_points=len(case.expected.matched_point_indices),
        point_f1=None,
        quote_total=0,
        quote_supported=0,
    )


def _compute_metrics(cases: tuple[CaseEvaluation, ...]) -> list[MetricResult]:
    total = len(cases)
    score_cases = [case for case in cases if case.predicted_score_0_10 is not None]
    point_cases = [case for case in cases if case.point_f1 is not None]
    quote_total = sum(case.quote_total for case in cases)
    quote_supported = sum(case.quote_supported for case in cases)

    errors = [case.absolute_error_0_10 for case in score_cases]
    raw_exact = [
        case.predicted_score_raw == case.expected_score_raw
        for case in cases
        if case.predicted_score_raw is not None
    ]
    within_1 = [
        case.absolute_error_0_10 <= 1.0
        for case in score_cases
        if case.absolute_error_0_10 is not None
    ]

    tp = sum(case.true_positive_points for case in point_cases)
    fp = sum(case.false_positive_points for case in point_cases)
    fn = sum(case.false_negative_points for case in point_cases)

    values: dict[str, float | None] = {
        "json_validity_rate": _safe_rate(sum(case.valid_json for case in cases), total),
        "schema_validity_rate": _safe_rate(sum(case.schema_valid for case in cases), total),
        "score_mae_0_10": _mean(errors),
        "score_rmse_0_10": sqrt(_mean([error * error for error in errors])) if errors else None,
        "score_within_1_accuracy": _safe_rate(sum(within_1), len(within_1)),
        "raw_score_exact_accuracy": _safe_rate(sum(raw_exact), len(raw_exact)),
        "point_micro_precision": _safe_rate(tp, tp + fp),
        "point_micro_recall": _safe_rate(tp, tp + fn),
        "point_micro_f1": _f1(tp, fp, fn),
        "point_macro_f1": _mean([case.point_f1 for case in point_cases if case.point_f1 is not None]),
        "evidence_quote_support_rate": _safe_rate(quote_supported, quote_total),
        "spearman_score_correlation": _spearman(
            [case.expected_score_0_10 for case in score_cases],
            [case.predicted_score_0_10 for case in score_cases if case.predicted_score_0_10 is not None],
        ),
    }

    return [
        MetricResult(
            key=key,
            title=METRIC_DEFINITIONS[key][0],
            value=values[key],
            explanation=METRIC_DEFINITIONS[key][1],
            better=METRIC_DEFINITIONS[key][2],
        )
        for key in METRIC_DEFINITIONS
    ]


def _extract_int(data: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return None


def _extract_float(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_point_indices(
    prediction: dict[str, Any],
    case: BenchmarkCase,
) -> tuple[tuple[int, ...], bool]:
    explicit_indices = prediction.get("matched_point_indices")
    if isinstance(explicit_indices, list):
        indices = _clean_indices(explicit_indices, len(case.criterion.points))
        return tuple(indices), True

    matched_points = prediction.get("matched_points")
    if not isinstance(matched_points, list):
        return (), False

    indices: list[int] = []
    for item in matched_points:
        if not isinstance(item, dict):
            continue
        index = item.get("point_index")
        if index is not None:
            indices.extend(_clean_indices([index], len(case.criterion.points)))
            continue

        point_text = item.get("point")
        if point_text:
            index_by_text = _find_point_index(str(point_text), case.criterion.points)
            if index_by_text is not None:
                indices.append(index_by_text)

    return tuple(sorted(set(indices))), True


def _clean_indices(values: list[Any], max_index: int) -> list[int]:
    indices: list[int] = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= max_index:
            indices.append(index)
    return sorted(set(indices))


def _find_point_index(point_text: str, points: tuple[str, ...]) -> int | None:
    normalized = _normalize(point_text)
    for index, point in enumerate(points, start=1):
        if _normalize(point) == normalized:
            return index
    return None


def _quote_support(prediction: dict[str, Any], dialogue: str) -> tuple[int, int]:
    matched_points = prediction.get("matched_points")
    if not isinstance(matched_points, list):
        return 0, 0

    dialogue_normalized = _normalize(dialogue)
    total = 0
    supported = 0
    for item in matched_points:
        if not isinstance(item, dict):
            continue
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            continue
        total += 1
        if _normalize(quote) in dialogue_normalized:
            supported += 1
    return total, supported


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _f1(tp: int, fp: int, fn: int) -> float | None:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _spearman(expected: list[float], predicted: list[float]) -> float | None:
    if len(expected) != len(predicted) or len(expected) < 2:
        return None
    expected_rank = _rank(expected)
    predicted_rank = _rank(predicted)
    return _pearson(expected_rank, predicted_rank)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[position][1]:
            end += 1
        average_rank = (position + end + 2) / 2
        for item_index in range(position, end + 1):
            ranks[indexed[item_index][0]] = average_rank
        position = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = _mean(left)
    right_mean = _mean(right)
    if left_mean is None or right_mean is None:
        return None

    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(left_value * right_value for left_value, right_value in zip(left_centered, right_centered))
    left_norm = sqrt(sum(value * value for value in left_centered))
    right_norm = sqrt(sum(value * value for value in right_centered))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def _format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_indices(indices: tuple[int, ...]) -> str:
    if not indices:
        return "-"
    return ", ".join(str(index) for index in indices)

