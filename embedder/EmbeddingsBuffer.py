from typing import Callable
import numpy as np
import logging


class EmbeddingsBuffer:
    def __init__(
            self,
            flush_fn: Callable,
            cols: int = 1024,
            data_dtype: np.dtype = np.float16,
            id_dtype: np.dtype = np.uint64,
            limit_gb: int = 4,
            logger_name: str = "logger",
    ):
        self.logger = logging.getLogger(logger_name)

        self.cols = cols
        self.data_dtype = data_dtype
        self.id_dtype = id_dtype

        bytes_per_row = self.cols * np.dtype(self.data_dtype).itemsize
        self.max_rows = int((limit_gb * self.cols ** 3) // bytes_per_row)

        self.buffer = np.empty((self.max_rows, self.cols), dtype=self.data_dtype)
        self.current_idx = 0

        self.ids = np.empty(self.max_rows, dtype=self.id_dtype)

        self.flush_fn = flush_fn

    def append(self, ids: np.ndarray, data: np.ndarray):
        data = np.asarray(data, dtype=self.data_dtype)
        ids = np.asarray(ids, dtype=self.id_dtype)
        num_rows = data.shape[0]

        if num_rows != ids.shape[0]:
            self.logger.error("Number of rows in ids and embeddings does not match")
            raise ValueError("Number of rows in ids and embeddings does not match")

        if self.current_idx + num_rows > self.max_rows:
            self.flush()

        self.buffer[self.current_idx: self.current_idx + num_rows] = data
        self.ids[self.current_idx: self.current_idx + num_rows] = ids
        self.current_idx += num_rows

    def flush(self):
        if self.current_idx == 0:
            return

        ready_data = self.buffer[:self.current_idx]
        ready_ids = self.ids[:self.current_idx]
        self.logger.info(f"Flushing {ready_data.nbytes / self.cols ** 2:.2f}MB")
        self.flush_fn(ready_ids, ready_data)
        self.logger.info(f"Flushing complete")
        self.current_idx = 0
