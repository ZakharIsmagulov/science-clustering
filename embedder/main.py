from pathlib import Path
from EmbedModel import E5LargeEmbedModel
from PyarrowStream import EmbeddingParquetWriter
from EmbeddingsBuffer import EmbeddingsBuffer
from typing import Generator, List, Callable
import logging
import numpy as np
import duckdb
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="DuckDB data processing script")

    parser.add_argument("--db-path", type=Path, required=True,
                        help="Path to DuckDB file")
    parser.add_argument("--table-name", type=str, required=True,
                        help="Name of the table to read")
    parser.add_argument("--save-name", type=Path, required=True,
                        help="Name of the file to save embeddings")

    parser.add_argument("--id-name", type=str, default="eid",
                        help="Name of the ID column (default: eid)")
    parser.add_argument("--title-name", type=str, default="title",
                        help="Name of the Title column (default: title)")
    parser.add_argument("--abstract-name", type=str, default="abstract",
                        help="Name of the Abstract column (default: abstract)")

    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="Chunk size to get from DB (default: 100000)")

    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size to vectorize (default: None)")
    parser.add_argument("--stop", type=int, default=None,
                        help="Limit to the number of rows from DB to vectorize (default: None)")

    return parser.parse_args()


def set_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_texts_db(
        db_path: Path,
        table_name: str,
        id_name: str,
        title_name: str,
        abstract_name: str,
        chunk_size,
        stop=None
) -> Generator[List, None, None]:
    con = duckdb.connect(str(db_path.absolute()), read_only=True)

    try:
        rows_processed = 0

        query = f"SELECT CAST({id_name} AS UBIGINT), {title_name}, {abstract_name} FROM {table_name}"
        result = con.execute(query)

        while True:
            if stop is not None:
                if rows_processed >= stop:
                    break
                to_fetch = min(chunk_size, stop - rows_processed)
            else:
                to_fetch = chunk_size

            batch = result.fetchmany(to_fetch)

            if not batch:
                break

            yield [[np.uint64(row[0]), row[1], row[2]] for row in batch]

            rows_processed += len(batch)
    finally:
        con.close()


def get_texts_csv(csv_path, chunk_size=100_000, stop=None) -> Generator[List, None, None]:
    reader = pd.read_csv(
        csv_path,
        usecols=["eid", "title", "abstract"],
        chunksize=chunk_size
    )

    rows_processed = 0
    for chunk in reader:
        if stop is not None and rows_processed >= stop:
            break

        if stop is not None and rows_processed + len(chunk) > stop:
            yield chunk.iloc[:stop - rows_processed].values.tolist()
            break

        yield chunk.values.tolist()
        rows_processed += len(chunk)


def transform_rows(rows: List):
    ids = []
    texts = []

    for row in rows:
        ids.append(row[0])
        texts.append(f"{row[1]}\n{row[2]}")

    return np.array(ids), texts


def flush_fn(writer: EmbeddingParquetWriter) -> Callable:
    def flush(row_ids: List, embeddings: np.ndarray):
        writer.write(row_ids, embeddings)
    return flush


def process(
        db_path: Path,
        table_name: str,
        save_path: Path,
        id_name: str = "eid",
        title_name: str = "title",
        abstract_name: str = "abstract",
        chunk_size: int = 100_000,
        batch_size: int | None = None,
        stop: int | None = None,
):
    logger = set_logger()
    logger.info("Process initialized")

    with EmbeddingParquetWriter(save_path) as writer:
        texts_gen = get_texts_db(
            db_path,
            table_name,
            id_name = id_name,
            title_name = title_name,
            abstract_name = abstract_name,
            chunk_size=chunk_size,
            stop=stop)
        embedder = E5LargeEmbedModel(batch_size=batch_size, model_path=EMBED_FOLDER, max_length=MAX_LEN, prefix=PREFIX)
        batch_size = embedder.get_batch_size()
        embs_buf = EmbeddingsBuffer(flush_fn=flush_fn(writer), limit_gb=1, cols=EMBEDDING_SIZE)

        for row_num, rows in enumerate(texts_gen):
            ids, texts = transform_rows(rows)
            chunk_gen = embedder.process_texts(texts)
            logger.info(f"Batch_size = {batch_size}")

            logger.info(f"---------------PROCESSING FROM {row_num * chunk_size}---------------")
            for i, embs in enumerate(chunk_gen):
                _ids = ids[i * batch_size:i * batch_size + embs.shape[0]]
                embs_buf.append(_ids, embs)

        embs_buf.flush()

    logger.info("Process finished")


if __name__ == "__main__":
    args = parse_args()

    process(args.db_path, args.table_name, args.save_name, args.id_name, args.title_name,
            args.abstract_name, args.chunk_size, args.batch_size, args.stop)
