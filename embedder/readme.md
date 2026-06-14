# Embeddings Generator

Скрипт предназначен для чтения текстовых данных из таблицы DuckDB, генерации эмбеддингов и сохранения результата в parquet-файл.

В качестве входных данных используются идентификатор записи, заголовок и аннотация. Для каждой строки формируется текст вида:

```text
<title>
<abstract>
```

После этого текст передается в embedding-модель, а полученные векторы сохраняются через `EmbeddingParquetWriter`.

## Требования

* Python 3.11
* DuckDB-файл с таблицей, содержащей колонки с ID, заголовком и аннотацией
* Установленные зависимости из `requirements.txt`
* Доступные локальные модули проекта:

  * `EmbedModel`
  * `PyarrowStream`
  * `EmbeddingsBuffer`

Установите зависимости:

```bash
pip install -r requirements.txt
```

## Модель эмбеддингов

Для работы скрипта в папке проекта должна находиться локальная папка с моделью `intfloat/multilingual-e5-large`.

Пример структуры проекта:

```text
project/
├── .py files
├── test_text.txt
├── requirements.txt
├── e5_large/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
```

Папка `e5_large` должна содержать скачанную модель `intfloat/multilingual-e5-large`.

## Запуск

```bash
python <script_name>.py \
  --db-path path/to/database.duckdb \
  --table-name table_name \
  --save-name path/to/output_embeddings.parquet
```

Пример:

```bash
python main.py \
  --db-path data/articles.duckdb \
  --table-name articles \
  --save-name output/embeddings.parquet
```

## Аргументы командной строки

| Аргумент          | Обязательный | Значение по умолчанию | Описание                                      |
| ----------------- | -----------: | --------------------: | --------------------------------------------- |
| `--db-path`       |           Да |                     — | Путь к DuckDB-файлу                           |
| `--table-name`    |           Да |                     — | Название таблицы для чтения                   |
| `--save-name`     |           Да |                     — | Путь к файлу, куда будут сохранены эмбеддинги |
| `--id-name`       |          Нет |                 `eid` | Название колонки с ID                         |
| `--title-name`    |          Нет |               `title` | Название колонки с заголовком                 |
| `--abstract-name` |          Нет |            `abstract` | Название колонки с аннотацией                 |
| `--chunk-size`    |          Нет |              `100000` | Размер чанка при чтении данных из DuckDB      |
| `--batch-size`    |          Нет |                `None` | Размер батча для векторизации текстов         |
| `--stop`          |          Нет |                `None` | Ограничение на количество строк для обработки |

## Пример запуска с дополнительными параметрами

```bash
python main.py \
  --db-path data/articles.duckdb \
  --table-name articles \
  --save-name output/embeddings.parquet \
  --id-name eid \
  --title-name title \
  --abstract-name abstract \
  --chunk-size 50000 \
  --batch-size 128 \
  --stop 1000000
```

В этом примере скрипт:

1. Открывает DuckDB-файл `data/articles.duckdb`.
2. Читает данные из таблицы `articles`.
3. Берет колонки `eid`, `title` и `abstract`.
4. Обрабатывает данные чанками по `50000` строк.
5. Векторизует тексты батчами по `128`.
6. Останавливается после обработки `1000000` строк.
7. Сохраняет эмбеддинги в `output/embeddings.parquet`.

## Формат входных данных

По умолчанию таблица должна содержать следующие колонки:

| Колонка    | Описание                        |
| ---------- | ------------------------------- |
| `eid`      | Уникальный идентификатор записи |
| `title`    | Заголовок текста                |
| `abstract` | Аннотация или основной текст    |

Если в таблице используются другие названия колонок, их можно передать через параметры:

```bash
python main.py \
  --db-path data/articles.duckdb \
  --table-name articles \
  --save-name output/embeddings.parquet \
  --id-name article_id \
  --title-name article_title \
  --abstract-name article_abstract
```

## Формат результата

Результат сохраняется в parquet-файл, указанный в аргументе `--save-name`.

Файл содержит eid записей и соответствующие им embedding-векторы.
