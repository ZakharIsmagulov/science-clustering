import logging
import json
import time
from annotation_prompt import get_annot_system_p, get_annot_user_p
from pydantic import BaseModel
from chatgpt_api import make_batch_line, write_jsonl, process_batch
import duckdb
import pandas as pd
from DuckDBApi import save_df_to_duckdb


class GptResult(BaseModel):
    topic_id: int
    name: str
    description: str


def get_best_pubs_for_topic(duckdb_path: str, clusters_df: pd.DataFrame, topic_id: int):
    top_10_ids = clusters_df[
        (clusters_df["topic"] == topic_id) & (clusters_df["is_core"])
    ].nlargest(10, "probability")["id"].tolist()

    ids_sql = ", ".join(str(x) for x in top_10_ids)

    query = f"""
    SELECT
        eid,
        title,
        abstract
    FROM main.docs
    WHERE CAST(eid AS VARCHAR) IN ({ids_sql})
    """

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        result_df = con.execute(query).df()
        records = result_df[["title", "abstract"]].to_dict(orient="records")
    finally:
        con.close()

    return records


def process_one(duckdb_path: str, clusters_df: pd.DataFrame, topic_id: int):
    return make_batch_line(
        system_prompt=get_annot_system_p(),
        user_prompt=get_annot_user_p(topic_id, get_best_pubs_for_topic(duckdb_path, clusters_df, topic_id)),
        response_model=GptResult,
        custom_id=str(topic_id),
        model="gpt-5-mini",
    )

def send_to_gpt(duckdb_path: str, clusters_df: pd.DataFrame):
    topics = sorted(clusters_df["topic"].unique().tolist())

    rows = []
    for topic_id in topics:
        rows.append(process_one(duckdb_path, clusters_df, topic_id))

    write_jsonl(rows, "res/llm/requests.jsonl")

    stats = process_batch(GptResult)
    return stats


def write_db():
    records = []
    with open("res/llm/success.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "topic": row["topic_id"],
                "name": row["name"],
                "description": row["description"],
            })

    df = pd.DataFrame(records, columns=["topic", "name", "description"])

    con = duckdb.connect("res/to_topic_viewer.duckdb")
    try:
        save_df_to_duckdb(con, df, "topic_annotations.annotation")
    finally:
        con.close()


def annotate_clusters(duckdb_path: str, _clusters_df: pd.DataFrame, logger_name: str = "logger"):
    timings = {}
    logger = logging.getLogger(logger_name)

    clusters_df = _clusters_df[_clusters_df["topic"] != -1].reset_index(drop=True)

    logger.info("Starting to GPT process")
    t0 = time.perf_counter()
    stats = send_to_gpt(duckdb_path, clusters_df)
    timings["gpt_process"] = time.perf_counter() - t0
    logger.info(f"GPT process is done. Total={stats.total}; Succeeded={stats.succeeded}; Failed={stats.failed}")

    logger.info("Starting to push to DB")
    t0 = time.perf_counter()
    write_db()
    timings["push_to_db"] = time.perf_counter() - t0
    logger.info("Finished pushing to DB")

    return timings
