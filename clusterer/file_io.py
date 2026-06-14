from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from typing import Literal, Union
import cudf
import cupy as cp
import os
import joblib


@dataclass
class ReducedData:
    ids: list
    mat: np.ndarray

    file_type: Literal["umap"] = "umap"


@dataclass
class ClustersData:
    ids: list
    topics: cp.ndarray
    is_core: cp.ndarray
    probs: cp.ndarray

    file_type: Literal["hdbscan", "periphery"]


@dataclass
class TempData:
    value: Union[ReducedData, ClustersData]


def load_parquet_batches(path_str: str, id_col="eid", emb_col="embedding", dim=1024, batch_size=60_000):
    pf = pq.ParquetFile(path_str)

    ids = []
    rows = []

    for batch in pf.iter_batches(columns=[id_col, emb_col], batch_size=batch_size):
        id_arr = batch.column(batch.schema.get_field_index(id_col))
        emb = batch.column(batch.schema.get_field_index(emb_col))

        # Bad dims?
        lengths = pa.compute.list_value_length(emb)
        bad = pa.compute.any(pa.compute.not_equal(lengths, dim)).as_py()
        if bad:
            raise ValueError(f"Found embeddings with dim != {dim}")

        flat = emb.values.to_numpy(zero_copy_only=False)
        rows.append(flat.reshape(len(id_arr), dim).astype(np.float32, copy=False))

        ids.extend(id_arr.to_pylist())

    mat = np.vstack(rows) if rows else np.empty((0, dim), dtype=np.float32)
    return ids, mat


def save_temp_parquet(_id: str, data: TempData):
    dt = data.value
    if dt.file_type == "umap":
        table = pa.table({
            "id": pa.array(dt.ids),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(dt.mat.ravel(), type=pa.float16()),
                list_size=16,
            ),
        })
        pq.write_table(table, f"temp/{_id}_umap.parquet")
        return

    table = cudf.DataFrame({
        "id": dt.ids,
        "topic": cudf.Series(dt.topics),
        "is_core": cudf.Series(dt.is_core),
        "probability": cudf.Series(dt.probs),
    })

    if dt.file_type == "hdbscan":
        table.to_parquet(f"temp/{_id}_hdbscan.parquet", index=False)
    else:
        table.to_parquet(f"temp/{_id}.parquet", index=False)


def save_temp_joblib(_id: str, hdbscan_model):
    joblib.dump(hdbscan_model, f"temp/{_id}_hdbscan.joblib")


def del_temp_parquet(_id: str, _type: Literal["umap", "hdbscan"]):
    path_str = f"temp/{_id}_{_type}.parquet"
    if os.path.exists(path_str):
        os.remove(path_str)

    if _type == "hdbscan":
        path_str = f"temp/{_id}_hdbscan.joblib"
        if os.path.exists(path_str):
            os.remove(path_str)
