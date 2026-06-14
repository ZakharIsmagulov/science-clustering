from typing import List
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pathlib import Path


class EmbeddingParquetWriter:
    def __init__(
            self,
            output_path: Path,
            compression: str = "zstd",
            logger_name: str = "logger",
    ):
        self.logger = logging.getLogger(logger_name)
        self.output_path = str(output_path.absolute())
        self.compression = compression
        self.writer = None

        self.schema = pa.schema([
            ("eid", pa.int64()),
            ("embedding", pa.large_list(pa.float32())),
        ])

    def __enter__(self):
        return self

    def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb
    ):
        self.close()

    def write(
            self,
            row_ids: List,
            embeddings: np.ndarray
    ):
        if embeddings.ndim != 2:
            self.logger.error(f"Embeddings must be 2D, got shape={embeddings.shape}")
            raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")

        flat_values = pa.array(embeddings.reshape(-1), type=pa.float16())
        dim = embeddings.shape[1]
        offsets = pa.array(np.arange(0, (len(embeddings) + 1) * dim, dim, dtype=np.int64))
        embedding_array = pa.LargeListArray.from_arrays(offsets, flat_values)

        table = pa.Table.from_arrays(
            arrays=[
                pa.array(row_ids, type=pa.int64()),
                embedding_array,
            ],
            schema=self.schema,
        )

        if self.writer is None:
            self.writer = pq.ParquetWriter(
                where=self.output_path,
                schema=self.schema,
                compression=self.compression,
                use_dictionary=False,
            )

        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
