# agent_scorer

Проект для оценки LLM-судей на ANLI-style данных. Входные данные представлены как пары:

- `evaluation_ANLI/generated/dialogues/*.json` - диалог, собранный из premise/hypothesis.
- `evaluation_ANLI/generated/criteria/*.yaml` - критерий с `points` и разметкой `gold`.

Основной запуск идет через `main.py`. Тип модели выбирается переменной `MODEL_TYPE`.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Конфигурация

Проект читает настройки из `config/.env`.

Минимальный набор:

```env
MODEL_TYPE=simple
DIALOGUE_BLOCK_SIZE=2
SKIP_PREDICT=0

DIALOGUE_INPUT_PATH=/abs/path/to/dialogue.json
CRETERIONS_PATH=/abs/path/to/criterion.yaml
LLM_RESPONSE_PATH=data/responses/output.json

API_KEY=...
EVALUATED_SPEAKER=B
```

Основные переменные:

- `MODEL_TYPE` - `simple` или `points`.
- `PROMPT_PATH` - кастомный YAML prompt-файл. Если пусто, используется `data/prompts/{MODEL_TYPE}_model.yaml`.
- `DIALOGUE_INPUT_PATH` - путь к одному dialogue JSON.
- `CRETERIONS_PATH` - путь к одному criterion YAML.
- `LLM_RESPONSE_PATH` - путь к JSON-файлу или директории для сохранения ответа.
- `DIALOGUE_BLOCK_SIZE` - размер блока диалога.
- `SKIP_PREDICT` - `1` для debug без вызова LLM, `0` для реального запроса.
- `API_KEY` - ключ API провайдера.
- `EVALUATED_SPEAKER` - оцениваемый speaker, обычно `B`.

## Модели

### `simple`

Использует один prompt из `data/prompts/simple_model.yaml`.

Ожидаемый финальный ответ модели:

```json
{
  "points": [
    {
      "point": "...",
      "score": 1,
      "quote": "...",
      "rationale": "...",
      "confidence": 1.0
    }
  ]
}
```

Классы:

- `1` - entailment / present
- `0` - neutral / absent
- `-1` - contradiction / opposite

### `points`

Использует цепочку prompt-шагов из `data/prompts/points_model.yaml`.
Каждый criterion point оценивается отдельно, а итог берется из последнего шага.

Ожидаемый финальный label:

```text
present | absent | opposite
```

## Одиночный запуск

Настроить `config/.env`, затем:

```bash
python3 main.py
```

Если `LLM_RESPONSE_PATH` указывает на файл, результат перезапишется в этот файл.
Если путь указывает на директорию, будет создан файл с timestamp.

## Batch-запуск по ANLI

Оркестратор:

```bash
python3 evaluation_ANLI/run_all.py
```

Он:

1. берет пары из `evaluation_ANLI/generated/criteria` и `evaluation_ANLI/generated/dialogues`;
2. проверяет совпадение имени файла без расширения;
3. перезаписывает в `config/.env` переменные `DIALOGUE_INPUT_PATH`, `CRETERIONS_PATH`, `LLM_RESPONSE_PATH`;
4. запускает `main.py`;
5. сохраняет ответы в `data/responses/anli_all/<stem>.json`.

Полезные режимы:

```bash
python3 evaluation_ANLI/run_all.py --dry-run --limit 2
python3 evaluation_ANLI/run_all.py --limit 10
python3 evaluation_ANLI/run_all.py --start 100 --limit 50
python3 evaluation_ANLI/run_all.py --overwrite
```

`--overwrite` перезапускает уже существующие ответы. Без него готовые JSON-файлы пропускаются.

## Оценка качества

Ноутбуки:

- `evaluation_ANLI/evaluate_simple_response.ipynb` - оценка ответов `simple`.
- `evaluation_ANLI/evaluate_point_response.ipynb` - оценка ответов `points`.

Обе оценки считают:

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1_score`
- `weighted_f1_score`
- `confusion_matrix`
- per-class `TP/TN/FP/FN`

Также отдельно фиксируются проблемы:

- разные длины `gold` и предсказаний;
- отсутствующие predictions;
- лишние predictions;
- неизвестные labels.

Для point-модели batch-оценка по всей папке:

```python
batch_results, overall_stats, problem_stats, batch_df, problem_df, batch_errors = collect_batch_results(
    "/Users/nident/Desktop/JOB/startup/agent_scorer/data/responses/point_model"
)
```

Для simple-модели аналогичная функция читает папку с ответами simple-модели.

## Структура

```text
config/.env                         # локальная конфигурация
main.py                             # основной entrypoint
models/simple_model.py              # one-shot модель по всем points
models/points_model.py              # multi-step модель по каждому point
data/prompts/simple_model.yaml      # prompt для simple
data/prompts/points_model.yaml      # prompt-chain для points
data/responses/                     # сохраненные ответы моделей
evaluation_ANLI/generated/          # сгенерированные ANLI criteria/dialogues
evaluation_ANLI/run_all.py          # batch-оркестратор
evaluation_ANLI/*.ipynb             # ноутбуки оценки
utils/                              # загрузка данных, путей, points
```

## Важно

Не коммитьте реальные ключи API и локальные `.env`-файлы.
