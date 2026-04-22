import unittest

from benchmark.dataset import load_benchmark_cases
from benchmark.metrics import PredictionRecord, compute_report
from benchmark.runner import DEFAULT_CASES_PATH, load_predictions


class BenchmarkMetricsTest(unittest.TestCase):
    def test_perfect_predictions_get_perfect_quality_metrics(self) -> None:
        cases = load_benchmark_cases(DEFAULT_CASES_PATH)
        predictions = [
            PredictionRecord(
                case_id=case.id,
                prediction={
                    "score_raw": case.expected.score_raw,
                    "max_score": case.expected.max_score,
                    "score_0_10": case.expected.score_0_10,
                    "matched_point_indices": list(case.expected.matched_point_indices),
                    "matched_points": [],
                },
            )
            for case in cases
        ]

        report = compute_report(cases, predictions)
        metrics = {metric.key: metric.value for metric in report.metrics}

        self.assertEqual(metrics["json_validity_rate"], 1.0)
        self.assertEqual(metrics["schema_validity_rate"], 1.0)
        self.assertEqual(metrics["score_mae_0_10"], 0.0)
        self.assertEqual(metrics["raw_score_exact_accuracy"], 1.0)
        self.assertEqual(metrics["point_micro_f1"], 1.0)
        self.assertEqual(metrics["point_macro_f1"], 1.0)

    def test_sample_predictions_are_loaded_and_evaluated(self) -> None:
        cases = load_benchmark_cases(DEFAULT_CASES_PATH)
        predictions = load_predictions("data/benchmark_sample_predictions.jsonl")

        report = compute_report(cases, predictions)
        metrics = {metric.key: metric.value for metric in report.metrics}

        self.assertEqual(report.case_count, 4)
        self.assertEqual(metrics["json_validity_rate"], 1.0)
        self.assertGreater(metrics["point_micro_f1"], 0.8)
        self.assertLess(metrics["score_mae_0_10"], 1.2)


if __name__ == "__main__":
    unittest.main()
