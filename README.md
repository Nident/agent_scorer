# agent_scorer

https://docs.google.com/document/d/1dWWC2eoFjQ_3LfLg79yPwMYdmS7NIRZTA_iHsw7sVSc/edit?tab=t.0

## Benchmark

Бэнчмарк оценивает, насколько LLM-критик правильно выставляет готовность к переговорам на шкале 0-10 и насколько корректно находит конкретные пункты критерия.

Офлайн-проверка на примерных предсказаниях:

```bash
python main.py benchmark
```

Запуск LLM по кейсам:

```bash
python main.py benchmark --run-llm --model-env config/model.env
```

Быстрая проверка API на первом кейсе:

```bash
python main.py benchmark --run-llm --model-env config/model.env --limit 1
```

Свой файл предсказаний:

```bash
python main.py benchmark --predictions data/benchmark_sample_predictions.jsonl
```

Отчет сохраняется в `data/benchmark_runs/latest/report.md` и `data/benchmark_runs/latest/report.json`.

Ключевые метрики:

- `JSON validity` - доля ответов, которые можно распарсить как JSON.
- `Schema validity` - доля ответов с нужными полями для расчета качества.
- `MAE score 0-10` и `RMSE score 0-10` - ошибка балла готовности на шкале 0-10.
- `Within 1 point` - доля кейсов, где ошибка не больше 1 пункта.
- `Exact raw score` - точное совпадение числа выполненных подпунктов.
- `Point precision / recall / F1` - качество определения конкретных признаков подготовки.
- `Quote support` - доля цитат-доказательств, которые дословно есть в диалоге.
- `Spearman correlation` - сохраняет ли модель правильный порядок кейсов от слабой подготовки к сильной.
