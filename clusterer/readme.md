# Clusterization Pipeline

Интерактивный пайплайн для кластеризации эмбеддингов, восстановления после неудачного запуска и интерпретации сохранённых результатов кластеризации.

Скрипт работает с эмбеддингами из Parquet-файла, параметрами из JSON-конфига и метаданными из DuckDB.

## Требования

* Python 3.11
* cuda 13.*
* Установленные зависимости из `requirements.txt`
* Parquet-файл с эмбеддингами
* JSON-файл с параметрами кластеризации
* DuckDB-файл с метаданными

## Установка

Установите зависимости:

```bash
pip install "cudf-cu13==26.4.*" "cuml-cu13==26.4.*"
pip install -r requirements.txt
```

## Назначение скрипта

Скрипт запускает интерактивный режим, в котором можно:

* запустить кластеризацию эмбеддингов;
* посмотреть список сохранённых результатов кластеризации;
* интерпретировать выбранный результат;
* продолжить выполнение после неудачного запуска;
* выйти из интерактивного режима.

## Запуск

Пример запуска:

```bash
python main.py \
  --parquet-name data/embeddings.parquet \
  --params-name configs/cluster_params.json \
  --duckdb-name data/metadata.duckdb
```

После запуска откроется интерактивный режим:

```text
Interactive clusterization pipeline
Commands:
  cluster              - run clustering with current config
  list                 - show saved clusterizations and metrics
  interpret <num>      - interpret selected clusterization
  continue-failed      - continue latest failed checkpoint
  exit                 - stop
```

## Аргументы командной строки

| Аргумент         | Обязательный | Описание                                                |
| ---------------- | -----------: | ------------------------------------------------------- |
| `--parquet-name` |           Да | Путь к Parquet-файлу с эмбеддингами в формате `float32` |
| `--params-name`  |           Да | Путь к JSON-файлу с параметрами кластеризации           |
| `--duckdb-name`  |           Да | Путь к DuckDB-файлу с метаданными                       |

## Интерактивные команды

### `cluster`

Запускает кластеризацию с текущим конфигом.

```text
> cluster
```

При запуске команда очищает временные файлы UMAP и HDBSCAN в папке `temp/`.

Если в `temp/` уже есть результат с таким же `config.id`, скрипт автоматически создаст новый уникальный ID на основе исходного ID и случайного UUID-суффикса.

### `list`

Показывает список сохранённых кластеризаций.

```text
> list
```

Пример вывода:

```text
0. config_a
1. config_b
2. config_c
```

Номер из этого списка используется в команде `interpret <num>`.

### `interpret <num>`

Запускает интерпретацию выбранной кластеризации.

```text
> interpret 0
```

Где `0` — номер результата из команды `list`.

Перед интерпретацией скрипт ищет файл:

```text
temp/<config_id>.parquet
```

Если файл отсутствует, будет выброшена ошибка `FileNotFoundError`.

Важно: в описании команды указано, что интерпретация очищает папку `res/`. Перед запуском интерпретации рекомендуется сделать резервную копию нужных результатов.

### `continue-failed`

Продолжает выполнение последней неудачной кластеризации по последнему найденному checkpoint-файлу в папке `temp/`.

```text
> continue-failed
```

Скрипт автоматически определяет, с какого этапа нужно продолжить:

* если найден checkpoint UMAP — продолжает с этапа HDBSCAN;
* если найден checkpoint HDBSCAN — продолжает с этапа periphery;
* если найден итоговый промежуточный checkpoint — продолжает с этапа metrics.

### `exit`

Завершает интерактивный режим.

```text
> exit
```

## Папка `temp/`

Папка `temp/` используется для хранения промежуточных и итоговых файлов кластеризации.

Скрипт работает со следующими типами файлов:

```text
temp/<config_id>.parquet
temp/<config_id>_umap.parquet
temp/<config_id>_hdbscan.parquet
temp/<config_id>_hdbscan.joblib
```

Назначение файлов:

| Файл                          | Назначение                       |
| ----------------------------- | -------------------------------- |
| `<config_id>_umap.parquet`    | Checkpoint после UMAP            |
| `<config_id>_hdbscan.parquet` | Checkpoint после HDBSCAN         |
| `<config_id>_hdbscan.joblib`  | Сохранённая HDBSCAN-модель       |
| `<config_id>.parquet`         | Итоговый результат кластеризации |

При запуске команды `cluster` временные файлы с суффиксами `_umap` и `_hdbscan` очищаются.

## JSON-конфиг параметров

Параметры кластеризации задаются в JSON-файле, который передаётся через аргумент:

```bash
--params-name configs/cluster_params.json
```

Общая структура конфига:

```json
{
  "id": "first",
  "config": {
    "umap": {
      "n_neighbors": 15,
      "min_dist": 0.1
    },
    "hdbscan": {
      "min_cluster_size": 50,
      "min_samples": 10,
      "cluster_selection_method": "eom"
    },
    "periphery": {
      "core_sample_frac": 0.1,
      "core_sample_min": 100,
      "core_sample_max": 10000,
      "core_quantile": 0.9,
      "mv_batch_size": 10000,
      "batch_size": 10000
    }
  }
}
```

### Описание параметров

| Параметр                                  |     Тип | Описание                                                                                |
| ----------------------------------------- | ------: | --------------------------------------------------------------------------------------- |
| `id`                                      |   `str` | Идентификатор запуска кластеризации. Используется для именования файлов в папке `temp/` |
| `config.umap.n_neighbors`                 |   `int` | Количество соседей для UMAP                                                             |
| `config.umap.min_dist`                    | `float` | Минимальная дистанция между точками в UMAP-пространстве                                 |
| `config.hdbscan.min_cluster_size`         |   `int` | Минимальный размер кластера для HDBSCAN                                                 |
| `config.hdbscan.min_samples`              |   `int` | Минимальное количество соседей для устойчивой точки в HDBSCAN                           |
| `config.hdbscan.cluster_selection_method` |   `str` | Метод выбора кластеров. Обычно используется `"eom"`                                     |
| `config.periphery.core_sample_frac`       | `float` | Доля объектов, используемых для выбора core-sample                                      |
| `config.periphery.core_sample_min`        |   `int` | Минимальное количество объектов в core-sample                                           |
| `config.periphery.core_sample_max`        |   `int` | Максимальное количество объектов в core-sample                                          |
| `config.periphery.core_quantile`          | `float` | Квантиль для определения границ для periphery                                           |
| `config.periphery.mv_batch_size`          |   `int` | Размер батча для вычислений вероятности принадлежности точки кластеру                   |
| `config.periphery.batch_size`             |   `int` | Основной размер батча для обработки periphery-этапа                                     |

Если `id` уже используется в папке `temp/`, скрипт автоматически создаст новый уникальный идентификатор, добавив UUID-суффикс.


## Входные данные

### Parquet-файл с эмбеддингами

Файл передаётся через аргумент:

```bash
--parquet-name data/embeddings.parquet
```

Ожидается, что файл содержит:

* eid объектов;
* embedding - векторы в формате `float32`;

### DuckDB-файл с метаданными

Файл передаётся через аргумент:

```bash
--duckdb-name data/metadata.duckdb
```

## Логирование

Логи записываются в файл:

```text
logger.log
```

Формат логов:

```text
2026-01-01 12:00:00 | INFO | Starting clusterization
2026-01-01 12:00:01 | INFO | Loading config
2026-01-01 12:00:02 | INFO | Processing clusterization
```

При ошибках стек исключения также записывается в `logger.log`.

## Восстановление после сбоя

Если кластеризация завершилась с ошибкой, можно попробовать продолжить последний запуск:

```text
> continue-failed
```

Скрипт найдёт последний checkpoint в папке `temp/` и продолжит выполнение с соответствующего этапа.

Логика восстановления:

| Найденный checkpoint          | Продолжение с этапа |
| ----------------------------- | ------------------- |
| `<config_id>_umap.parquet`    | HDBSCAN             |
| `<config_id>_hdbscan.parquet` | Periphery           |
| `<config_id>.parquet`         | Metrics             |

## Важные замечания

* Перед запуском убедитесь, что папка `temp/` доступна для записи.
* Команда `cluster` очищает временные UMAP/HDBSCAN-файлы в `temp/`.
* Команда `interpret <num>` использует номер результата из команды `list`.
* Перед интерпретацией рекомендуется сохранить резервную копию папки `res/`, если в ней уже есть важные результаты.
* Если `config.id` уже используется, скрипт автоматически создаст новый уникальный ID.
* Все ошибки пишутся в `logger.log`.

## Пример полного сценария работы

Запуск скрипта:

```bash
python main.py \
  --parquet-name data/embeddings.parquet \
  --params-name configs/cluster_params.json \
  --duckdb-name data/metadata.duckdb
```

Запуск кластеризации:

```text
> cluster
```

Просмотр доступных результатов:

```text
> list
```

Интерпретация результата с номером `0`:

```text
> interpret 0
```

Завершение работы:

```text
> exit
```
