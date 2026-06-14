import logging
from HdbscanModel import HdbscanModel
import cupy as cp
import pandas as pd
from typing import Union, Literal, Tuple
import duckdb
import time


def simple_metrics(hdbscan_model: HdbscanModel):
    metrics = {}
    labels = hdbscan_model.get_labels()
    is_core = hdbscan_model.get_is_core()
    metrics["n_total"] = str(len(labels))
    metrics["n_core"] = str(int(cp.sum(is_core)))
    metrics["n_non_noise"] = str(int(cp.sum(labels != -1)))
    metrics["core_share"] = f"""{float(
        cp.sum(is_core) / len(is_core) if len(is_core) > 0 else 0.0
    ):.4f}"""
    metrics["non_noise_share"] = f"""{float(
        cp.sum(labels != -1) / len(is_core) if len(is_core) > 0 else 0.0
    ):.4f}"""
    metrics["n_clusters"] = str(int(cp.sum(cp.unique(labels) != -1)))
    return metrics


def _register_doc_clusters_df(
    con: duckdb.DuckDBPyConnection,
    df_clusters: pd.DataFrame,
    cluster_cols: list[str],
) -> None:
    relation_name = "doc_clusters_df"

    missing_columns = set(cluster_cols) - set(df_clusters.columns)
    if missing_columns:
        raise ValueError(
            "В df_clusters отсутствуют обязательные колонки: "
            f"{sorted(missing_columns)}"
        )

    df_to_register = df_clusters.loc[:, cluster_cols].copy()

    try:
        con.unregister(relation_name)
    except duckdb.CatalogException:
        pass

    con.register(relation_name, df_to_register)


def auth_cluster_concentration(
    duckdb_path: str,
    df_clusters: pd.DataFrame,
    min_auth_pubs: int = 3,

    # Колонки датафрейма (уже без -1 кластера)
    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
    only_core: bool = False,  # True = only core, False = all

    # duckdb таблица и колонки
    auth_doc_table: str = "auth_aff_doc",
    auth_col: str = "auth_id",
    auth_doc_doc_col: str = "doc_id",
    auth_seqn_col: str = "auth_seqn",

    # Отбор авторов: первый или последний в публикации
    # None = все авторы, "first" = только первые авторы, "last" = только последние авторы
    auth_order: Literal["first", "last"] | None = None,

    # Вывод дополнительного датафрейма
    return_per_author: bool = False,
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """
    Вычисляет:
      ACC(a) = max_k n(a,k)/n(a), где k — кластер (topic), a — автор,
      n(a,k) — число публикаций автора a в кластере k,
      n(a) — число публикаций автора a.
    Считается по авторам, затем усредняется по авторам (mean ACC).

    Условие на автора в публикации:
      auth_order:
        None — учитывать всех авторов;
        "first" — учитывать только первых авторов;
        "last" — учитывать только последних авторов.

    Возвращает mean_acc, и опционально датафрейм с колонками [auth_id, ACC, n_a].
    """
    if auth_order not in {None, "first", "last"}:
        raise ValueError("auth_order должен быть None, 'first' или 'last'")

    con = duckdb.connect(duckdb_path, read_only=True)

    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""

        pos_filter_sql = ""
        if auth_order is not None:
            pos_filter_sql = (
                f"WHERE ad.{auth_seqn_col} = 0"
                if auth_order == "first"
                else f"WHERE ad.{auth_seqn_col} = m.max_seqn"
            )

        sql_q = f"""
                WITH doc_clusters AS (
                    SELECT
                        CAST({df_doc_col} AS VARCHAR) AS doc_key,
                        {df_topic_col} AS topic,
                        {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                max_seqn AS (
                    SELECT
                        CAST({auth_doc_doc_col} AS VARCHAR) AS doc_key,
                        MAX({auth_seqn_col}) AS max_seqn
                    FROM {auth_doc_table}
                    GROUP BY CAST({auth_doc_doc_col} AS VARCHAR)
                ),
                auth_docs AS (
                    SELECT
                        ad.{auth_col} AS auth_id,
                        dc.topic
                    FROM {auth_doc_table} ad
                    JOIN doc_clusters dc
                      ON dc.doc_key = CAST(ad.{auth_doc_doc_col} AS VARCHAR)
                    {"JOIN max_seqn m ON m.doc_key = CAST(ad." + auth_doc_doc_col + " AS VARCHAR)" if auth_order else ""}
                    {pos_filter_sql}
                ),
                n_ak AS (
                    SELECT auth_id, topic, COUNT(*) AS n_ak
                    FROM auth_docs
                    GROUP BY auth_id, topic
                ),
                n_a AS (
                    SELECT auth_id, COUNT(*) AS n_a
                    FROM auth_docs
                    GROUP BY auth_id
                ),
                acc AS (
                    SELECT
                        n_ak.auth_id,
                        MAX(n_ak.n_ak::DOUBLE / n_a.n_a) AS acc,
                        n_a.n_a AS n_a
                    FROM n_ak
                    JOIN n_a USING(auth_id)
                    WHERE n_a.n_a >= {min_auth_pubs}
                    GROUP BY n_ak.auth_id, n_a.n_a
                )
                SELECT AVG(acc) AS mean_acc FROM acc;
                """

        mean_acc = con.execute(sql_q).fetchone()[0]
        if mean_acc is None:
            raise Exception("MEAN ACC (авторы) не было посчитано (значение None)")

        if not return_per_author:
            return float(mean_acc)

        per_author = con.execute(
            sql_q.replace(
                "SELECT AVG(acc) AS mean_acc FROM acc;",
                "SELECT auth_id, acc AS metric, n_a FROM acc ORDER BY auth_id;"
            )
        ).df()

        return float(mean_acc), per_author

    finally:
        con.close()


def aff_cluster_concentration(
    duckdb_path: str,
    df_clusters: pd.DataFrame,
    min_aff_pubs: int = 3,

    # Колонки датафрейма (уже без -1 кластера)
    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
    only_core: bool = False,  # True = only core, False = all

    # duckdb таблица и колонки
    aff_doc_table: str = "auth_aff_doc",
    aff_col: str = "aff_id",
    aff_doc_doc_col: str = "doc_id",

    # Вывод дополнительного датафрейма
    return_per_aff: bool = False,
) -> Union[float, int, Tuple[float, pd.DataFrame]]:
    """
    Вычисляет:
      ACC(aff) = max_k n(aff,k)/n(aff)
    усредненное по аффилиациям.

    Возвращает mean_acc, и опционально датафрейм с колонками [aff_id, ACC, n_aff].
    """

    con = duckdb.connect(duckdb_path, read_only=True)

    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""

        sql_q = f"""
                WITH doc_clusters AS (
                    SELECT CAST({df_doc_col} AS VARCHAR) AS doc_key,
                           {df_topic_col} AS topic,
                           {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                aff_docs AS (
                    SELECT ad.{aff_col} AS aff_id,
                           dc.topic
                    FROM {aff_doc_table} ad
                    JOIN doc_clusters dc
                      ON dc.doc_key = CAST(ad.{aff_doc_doc_col} AS VARCHAR)
                ),
                n_ak AS (
                    SELECT aff_id, topic, COUNT(*) AS n_ak
                    FROM aff_docs
                    GROUP BY aff_id, topic
                ),
                n_a AS (
                    SELECT aff_id, COUNT(*) AS n_a
                    FROM aff_docs
                    GROUP BY aff_id
                ),
                acc AS (
                    SELECT
                        n_ak.aff_id,
                        MAX(n_ak.n_ak::DOUBLE / n_a.n_a) AS acc,
                        n_a.n_a AS n_a
                    FROM n_ak
                    JOIN n_a USING(aff_id)
                    WHERE n_a.n_a >= {min_aff_pubs}
                    GROUP BY n_ak.aff_id, n_a.n_a
                )
                SELECT AVG(acc) AS mean_acc FROM acc;
                """

        mean_acc = con.execute(sql_q).fetchone()[0]
        if mean_acc is None:
            raise Exception('MEAN ACC (аффилиации) не было посчитано')
        if not return_per_aff:
            return float(mean_acc)
        per_aff = con.execute(sql_q.replace("SELECT AVG(acc) AS mean_acc FROM acc;",
                                            "SELECT aff_id, acc AS metric, n_a FROM acc;")).df()
        return float(mean_acc), per_aff

    finally:
        con.close()


def k_cluster_auth_rate(
    duckdb_path: str,
    df_clusters: pd.DataFrame,
    K: int,
    min_auth_pubs: int = 3,

    # Колонки датафрейма (уже без -1 кластера)
    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
    only_core: bool = False,  # True = only core, False = all

    auth_doc_table: str = "auth_aff_doc",
    auth_col: str = "auth_id",
    auth_doc_doc_col: str = "doc_id",

    return_distribution: bool = False,  # опционально возвращать распределение количества кластеров на автора
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """
    CAR(K) = n(K)/n где n(K) = количество авторов в не более K кластерах,
    и n = число авторов с как минимум min_auth_pubs уникальными документами в df_clusters.

    Вывод:
     доля авторов (float), и опционально датафрейм распределения:
      columns: n_clusters, n_authors, share
    """

    con = duckdb.connect(duckdb_path, read_only=True)

    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""

        sql_share = f"""
                WITH doc_clusters AS (
                    SELECT CAST({df_doc_col} AS VARCHAR) AS doc_key,
                           {df_topic_col} AS topic,
                           {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                auth_docs AS (
                    SELECT
                        ad.{auth_col} AS auth_id,
                        dc.topic,
                        dc.doc_key
                    FROM {auth_doc_table} ad
                    JOIN doc_clusters dc
                      ON dc.doc_key = CAST(ad.{auth_doc_doc_col} AS VARCHAR)
                ),
                n_a AS (
                    SELECT auth_id, COUNT(DISTINCT doc_key) AS n_a
                    FROM auth_docs
                    GROUP BY auth_id
                ),
                author_topics AS (
                    SELECT DISTINCT
                        d.auth_id,
                        d.topic
                    FROM auth_docs d
                    JOIN n_a USING(auth_id)
                    WHERE n_a >= {int(min_auth_pubs)}
                ),
                clusters_per_author AS (
                    SELECT auth_id, COUNT(*) AS n_clusters
                    FROM author_topics
                    GROUP BY auth_id
                )
                SELECT
                    SUM(CASE WHEN n_clusters <= {int(K)} THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS kcar
                FROM clusters_per_author;
                """

        share = con.execute(sql_share).fetchone()[0]
        if share is None:
            raise Exception('K-cluster author rate не было посчитано')
        share = float(share)

        if not return_distribution:
            return share

        sql_dist = f"""
                WITH doc_clusters AS (
                    SELECT CAST({df_doc_col} AS VARCHAR) AS doc_key,
                           {df_topic_col} AS topic,
                           {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                auth_docs AS (
                    SELECT
                        ad.{auth_col} AS auth_id,
                        dc.topic,
                        dc.doc_key
                    FROM {auth_doc_table} ad
                    JOIN doc_clusters dc
                      ON dc.doc_key = CAST(ad.{auth_doc_doc_col} AS VARCHAR)
                ),
                n_a AS (
                    SELECT auth_id, COUNT(DISTINCT doc_key) AS n_a
                    FROM auth_docs
                    GROUP BY auth_id
                ),
                author_topics AS (
                    SELECT DISTINCT
                        d.auth_id,
                        d.topic
                    FROM auth_docs d
                    JOIN n_a USING(auth_id)
                    WHERE n_a >= {int(min_auth_pubs)}
                ),
                clusters_per_author AS (
                    SELECT auth_id, COUNT(*) AS n_clusters
                    FROM author_topics
                    GROUP BY auth_id
                ),
                dist AS (
                    SELECT n_clusters, COUNT(*) AS n_authors
                    FROM clusters_per_author
                    GROUP BY n_clusters
                ),
                tot AS (SELECT SUM(n_authors) AS n FROM dist)
                SELECT
                    d.n_clusters,
                    d.n_authors,
                    d.n_authors::DOUBLE / tot.n AS metric
                FROM dist d, tot
                ORDER BY d.n_clusters;
                """
        dist_df = con.execute(sql_dist).df()
        if dist_df is None:
            raise Exception('K-cluster author rate распределение не было посчитано')
        return share, dist_df

    finally:
        con.close()


def auth_overlap(
    duckdb_path: str,
    df_clusters: pd.DataFrame,

    # Колонки датафрейма (уже без -1 кластера)
    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
    only_core: bool = False,  # True = only core, False = all

    # duckdb таблица и колонки
    auth_doc_table: str = "auth_aff_doc",
    auth_col: str = "auth_id",
    auth_doc_doc_col: str = "doc_id",

    # топ-k кластеров по количеству документов в них
    top_k: int = 50,
) -> tuple[float, pd.DataFrame]:
    """
    Вычисляет схожесть Жаккара между кластерами на множествах авторов для топ-k кластеров

    Вывод:
      pairs_df columns: topic_i, topic_j, ai, aj, jaccard
      и mean_jaccard по всем парам (i<j).
    """

    con = duckdb.connect(duckdb_path, read_only=True)

    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""

        topk_cte = f"""
                top_topics AS (
                    SELECT topic
                    FROM (
                        SELECT
                            dc.topic,
                            COUNT(DISTINCT dc.doc_key) AS docs_with_authors,
                            COUNT(DISTINCT ad.{auth_col}) AS authors
                        FROM doc_clusters dc
                        JOIN {auth_doc_table} ad
                          ON dc.doc_key = CAST(ad.{auth_doc_doc_col} AS VARCHAR)
                        WHERE ad.{auth_col} IS NOT NULL
                        GROUP BY dc.topic
                        ORDER BY docs_with_authors DESC
                        LIMIT {int(top_k)}
                    )
                )"""

        sql_pairs = f"""
        WITH
        doc_clusters AS (
            SELECT CAST({df_doc_col} AS VARCHAR) AS doc_key,
                   {df_topic_col} AS topic,
                   {df_is_core_col}
            FROM doc_clusters_df
            {core_filter_sql}
        ),

        {topk_cte},

        author_docs AS (
            SELECT ad.{auth_col} AS auth_id,
                   dc.topic
            FROM {auth_doc_table} ad
            JOIN doc_clusters dc
              ON dc.doc_key = CAST(ad.{auth_doc_doc_col} AS VARCHAR)
            JOIN top_topics tt
              ON tt.topic = dc.topic
            WHERE ad.{auth_col} IS NOT NULL
        ),

        cluster_authors AS (
            SELECT DISTINCT topic, auth_id
            FROM author_docs
        ),

        a_sizes AS (
            SELECT topic, COUNT(*) AS a_size
            FROM cluster_authors
            GROUP BY topic
        ),

        topic_pairs AS (
            SELECT
                s1.topic AS topic_i,
                s2.topic AS topic_j,
                s1.a_size AS ai,
                s2.a_size AS aj
            FROM a_sizes s1
            JOIN a_sizes s2
              ON s1.topic < s2.topic
        ),

        intersections AS (
            SELECT
                ca1.topic AS topic_i,
                ca2.topic AS topic_j,
                COUNT(*) AS inter
            FROM cluster_authors ca1
            JOIN cluster_authors ca2
              ON ca1.auth_id = ca2.auth_id
             AND ca1.topic < ca2.topic
            GROUP BY ca1.topic, ca2.topic
        ),

        pairs AS (
            SELECT
                tp.topic_i,
                tp.topic_j,
                tp.ai,
                tp.aj,
                COALESCE(i.inter, 0) AS inter,
                (
                    COALESCE(i.inter, 0)::DOUBLE
                    / NULLIF(tp.ai + tp.aj - COALESCE(i.inter, 0), 0)
                ) AS metric
            FROM topic_pairs tp
            LEFT JOIN intersections i
              ON i.topic_i = tp.topic_i
             AND i.topic_j = tp.topic_j
        )

        SELECT *
        FROM pairs
        ORDER BY metric DESC;
        """

        pairs_df = con.execute(sql_pairs).df()
        if pairs_df is None:
            raise Exception('Авторное перекрытие не было посчитано')
        mean_j = float(pairs_df["metric"].mean()) if len(pairs_df) else float("nan")
        return mean_j, pairs_df

    finally:
        con.close()


def cluster_purity(
    duckdb_path: str,
    df_clusters: pd.DataFrame,
    min_cluster_docs: int = 3,

    # Колонки датафрейма (уже без -1 кластера)
    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
    only_core: bool = False,  # True = only core, False = all

    # duckdb таблица и колонки
    docs_table: str = "docs",
    docs_eid_col: str = "eid",
    field: Literal["source", "doctype"] = "source",  # что считаем s
    drop_null_field: bool = True,

    # Вывод дополнительного датафрейма
    return_per_cluster: bool = False,
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """
    Вычисляет:
      JP(c) = max_s n(c,s)/n(c), где s — docs.source или docs.doctype
    усредненное по кластерам.

    Возвращает mean_jp, и опционально датафрейм с колонками:
      [topic, JP, n_c, top_value, top_count]
    """

    if field not in ("source", "doctype"):
        raise ValueError("field must be 'source' or 'doctype'")

    con = duckdb.connect(duckdb_path, read_only=True)

    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""
        null_filter_sql = f"AND d.{field} IS NOT NULL" if drop_null_field else ""

        sql_q = f"""
                WITH doc_clusters1 AS (
                    SELECT
                        CAST({df_doc_col} AS VARCHAR) AS doc_key,
                        {df_topic_col} AS topic,
                        {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                doc_clusters AS (
                    SELECT *
                    FROM doc_clusters1 dc1
                    WHERE dc1.topic IN (
                        SELECT
                            topic
                        FROM doc_clusters1 dc11
                        GROUP BY topic
                        HAVING COUNT(dc11.doc_key) > {min_cluster_docs}
                    ) 
                ),
                cl_docs AS (
                    SELECT
                        dc.topic,
                        d.{field} AS val
                    FROM doc_clusters dc
                    JOIN {docs_table} d
                      ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
                    WHERE 1=1
                    {null_filter_sql}
                ),
                n_cs AS (
                    SELECT topic, val, COUNT(*) AS n_cs
                    FROM cl_docs
                    GROUP BY topic, val
                ),
                n_c AS (
                    SELECT topic, COUNT(*) AS n_c
                    FROM cl_docs
                    GROUP BY topic
                ),
                jp AS (
                    SELECT
                        n_cs.topic,
                        MAX(n_cs.n_cs::DOUBLE / n_c.n_c) AS jp,
                        n_c.n_c AS n_c
                    FROM n_cs
                    JOIN n_c USING(topic)
                    GROUP BY n_cs.topic, n_c.n_c
                )
                SELECT AVG(jp) AS mean_jp FROM jp;
                """

        mean_jp = con.execute(sql_q).fetchone()[0]
        if mean_jp is None:
            raise Exception("MEAN JP (кластеры) не было посчитано")

        if not return_per_cluster:
            return float(mean_jp)

        # Детализация по кластерам: JP + самый частый val (source/doctype)
        per_cluster_sql = f"""
                WITH doc_clusters AS (
                    SELECT
                        CAST({df_doc_col} AS VARCHAR) AS doc_key,
                        {df_topic_col} AS topic,
                        {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                cl_docs AS (
                    SELECT
                        dc.topic,
                        d.{field} AS val
                    FROM doc_clusters dc
                    JOIN {docs_table} d
                      ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
                    WHERE 1=1
                    {null_filter_sql}
                ),
                n_cs AS (
                    SELECT topic, val, COUNT(*) AS n_cs
                    FROM cl_docs
                    GROUP BY topic, val
                ),
                n_c AS (
                    SELECT topic, COUNT(*) AS n_c
                    FROM cl_docs
                    GROUP BY topic
                ),
                scored AS (
                    SELECT
                        n_cs.topic,
                        n_cs.val,
                        n_cs.n_cs,
                        n_c.n_c,
                        (n_cs.n_cs::DOUBLE / n_c.n_c) AS share
                    FROM n_cs
                    JOIN n_c USING(topic)
                ),
                ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (PARTITION BY topic ORDER BY share DESC, n_cs DESC, val) AS rn
                    FROM scored
                )
                SELECT
                    topic,
                    share AS metric,
                    n_c   AS n_c,
                    val   AS top_value,
                    n_cs  AS top_count
                FROM ranked
                WHERE rn = 1
                ORDER BY topic;
                """

        per_cluster = con.execute(per_cluster_sql).df()
        return float(mean_jp), per_cluster

    finally:
        con.close()


def year_iqr(
    duckdb_path: str,
    df_clusters: pd.DataFrame,

    # Колонки датафрейма (уже без -1 кластера)
    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
    only_core: bool = False,  # True = only core, False = all

    # duckdb таблица и колонки
    docs_table: str = "docs",
    docs_eid_col: str = "eid",
    docs_year_col: str = "year",
    drop_null_year: bool = True,

    # Вывод дополнительного датафрейма
    return_per_cluster: bool = False,
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """
    Вычисляет для каждого кластера:
      IQR_c = Q0.75(year) - Q0.25(year),
    где квантили берутся по распределению year внутри кластера.

    Затем усредняет IQR по кластерам (mean IQR).

    Возвращает mean_iqr, и опционально датафрейм с колонками:
      [topic, IQR, q25, q75, n_c]
    """

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""
        null_filter_sql = f"AND d.{docs_year_col} IS NOT NULL" if drop_null_year else ""

        sql_q = f"""
                WITH doc_clusters AS (
                    SELECT
                        CAST({df_doc_col} AS VARCHAR) AS doc_key,
                        {df_topic_col} AS topic,
                        {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                cl_years AS (
                    SELECT
                        dc.topic,
                        CAST(d.{docs_year_col} AS DOUBLE) AS y
                    FROM doc_clusters dc
                    JOIN {docs_table} d
                      ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
                    WHERE 1=1
                    {null_filter_sql}
                ),
                per_cluster AS (
                    SELECT
                        topic,
                        COUNT(*) AS n_c,
                        quantile_cont(y, 0.25) AS q25,
                        quantile_cont(y, 0.75) AS q75
                    FROM cl_years
                    GROUP BY topic
                ),
                iqr AS (
                    SELECT
                        topic,
                        n_c,
                        q25,
                        q75,
                        (q75 - q25) AS iqr
                    FROM per_cluster
                )
                SELECT AVG(iqr) AS mean_iqr FROM iqr;
                """

        mean_iqr = con.execute(sql_q).fetchone()[0]
        if mean_iqr is None:
            raise Exception("MEAN IQR (кластеры по годам) не было посчитано")

        if not return_per_cluster:
            return float(mean_iqr)

        per_cluster = con.execute(
            sql_q.replace(
                "SELECT AVG(iqr) AS mean_iqr FROM iqr;",
                "SELECT topic, iqr AS metric, q25, q75, n_c FROM iqr ORDER BY topic;"
            )
        ).df()

        return float(mean_iqr), per_cluster

    finally:
        con.close()


def entropy_norm(
    duckdb_path: str,
    df_clusters: pd.DataFrame,

    # Колонки датафрейма (уже без -1 кластера)
    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
    only_core: bool = False,  # True = only core, False = all

    # duckdb таблица и колонки
    docs_table: str = "docs",
    docs_eid_col: str = "eid",
    docs_year_col: str = "year",
    drop_null_year: bool = True,

    # Вывод дополнительного датафрейма
    return_per_cluster: bool = False,
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """
    Для каждого кластера считает нормированную энтропию распределения годов:

      H(c) = - SUM_y p(c,y) * log(p(c,y))
      H_norm(c) = H(c) / log(K_c),

    где:
      p(c,y) — частота года y в кластере c,
      K_c — число уникальных годов в кластере.

    Далее усредняет по кластерам: mean H_norm.

    Возвращает mean_hnorm, и опционально датафрейм по кластерам:
      [topic, H_norm, K_c, n_c]
    """

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""
        null_filter_sql = f"AND d.{docs_year_col} IS NOT NULL" if drop_null_year else ""

        sql_q = f"""
                WITH doc_clusters AS (
                    SELECT
                        CAST({df_doc_col} AS VARCHAR) AS doc_key,
                        {df_topic_col} AS topic,
                        {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                cl_years AS (
                    SELECT
                        dc.topic,
                        CAST(d.{docs_year_col} AS BIGINT) AS y
                    FROM doc_clusters dc
                    JOIN {docs_table} d
                      ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
                    WHERE 1=1
                    {null_filter_sql}
                ),
                n_cy AS (
                    SELECT topic, y, COUNT(*) AS n_cy
                    FROM cl_years
                    GROUP BY topic, y
                ),
                n_c AS (
                    SELECT
                        topic,
                        SUM(n_cy) AS n_c,
                        COUNT(*) AS k_c
                    FROM n_cy
                    GROUP BY topic
                ),
                probs AS (
                    SELECT
                        n_cy.topic,
                        n_cy.n_cy,
                        n_c.n_c,
                        n_c.k_c,
                        (n_cy.n_cy::DOUBLE / NULLIF(n_c.n_c, 0)) AS p
                    FROM n_cy
                    JOIN n_c USING(topic)
                ),
                ent AS (
                    SELECT
                        topic,
                        MAX(n_c) AS n_c,
                        MAX(k_c) AS k_c,
                        -SUM(p * log(p)) AS h
                    FROM probs
                    GROUP BY topic
                ),
                ent_norm AS (
                    SELECT
                        topic,
                        n_c,
                        k_c,
                        CASE
                            WHEN k_c <= 1 THEN 0.0
                            ELSE h / log(k_c)
                        END AS h_norm
                    FROM ent
                )
                SELECT AVG(h_norm) AS mean_h_norm FROM ent_norm;
                """

        mean_h_norm = con.execute(sql_q).fetchone()[0]
        if mean_h_norm is None:
            raise Exception("MEAN H_norm (кластеры по годам) не было посчитано")

        if not return_per_cluster:
            return float(mean_h_norm)

        per_cluster = con.execute(
            sql_q.replace(
                "SELECT AVG(h_norm) AS mean_h_norm FROM ent_norm;",
                "SELECT topic, h_norm AS metric, k_c AS K_c, n_c FROM ent_norm ORDER BY topic;"
            )
        ).df()

        return float(mean_h_norm), per_cluster

    finally:
        con.close()


def year_peak_share(
    duckdb_path: str,
    df_clusters: pd.DataFrame,

    # Колонки датафрейма (уже без -1 кластера)
    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
    only_core: bool = False,  # True = only core, False = all

    # duckdb таблица и колонки
    docs_table: str = "docs",
    docs_eid_col: str = "eid",
    docs_year_col: str = "year",
    drop_null_year: bool = True,

    # Вывод дополнительного датафрейма
    return_per_cluster: bool = False,
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """
    Для каждого кластера считает peak share по годам:

      p_share(c) = max_y p(c,y),
      p(c,y) = n(c,y)/n(c).

    Далее усредняет p_share по кластерам: mean_peak_share.

    Возвращает mean_peak_share, и опционально датафрейм по кластерам:
      [topic, peak_share, n_c, peak_year, peak_count]
    """

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""
        null_filter_sql = f"AND d.{docs_year_col} IS NOT NULL" if drop_null_year else ""

        sql_q = f"""
                WITH doc_clusters AS (
                    SELECT
                        CAST({df_doc_col} AS VARCHAR) AS doc_key,
                        {df_topic_col} AS topic,
                        {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                cl_years AS (
                    SELECT
                        dc.topic,
                        CAST(d.{docs_year_col} AS BIGINT) AS y
                    FROM doc_clusters dc
                    JOIN {docs_table} d
                      ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
                    WHERE 1=1
                    {null_filter_sql}
                ),
                n_cy AS (
                    SELECT topic, y, COUNT(*) AS n_cy
                    FROM cl_years
                    GROUP BY topic, y
                ),
                n_c AS (
                    SELECT topic, SUM(n_cy) AS n_c
                    FROM n_cy
                    GROUP BY topic
                ),
                shares AS (
                    SELECT
                        n_cy.topic,
                        n_cy.y,
                        n_cy.n_cy,
                        n_c.n_c,
                        (n_cy.n_cy::DOUBLE / NULLIF(n_c.n_c, 0)) AS share
                    FROM n_cy
                    JOIN n_c USING(topic)
                ),
                peak AS (
                    SELECT
                        topic,
                        MAX(share) AS peak_share,
                        MAX(n_c)   AS n_c
                    FROM shares
                    GROUP BY topic
                )
                SELECT AVG(peak_share) AS mean_peak_share FROM peak;
                """

        mean_peak_share = con.execute(sql_q).fetchone()[0]
        if mean_peak_share is None:
            raise Exception("MEAN peak share (кластеры по годам) не было посчитано")

        if not return_per_cluster:
            return float(mean_peak_share)

        per_cluster_sql = f"""
                WITH doc_clusters AS (
                    SELECT
                        CAST({df_doc_col} AS VARCHAR) AS doc_key,
                        {df_topic_col} AS topic,
                        {df_is_core_col}
                    FROM doc_clusters_df
                    {core_filter_sql}
                ),
                cl_years AS (
                    SELECT
                        dc.topic,
                        CAST(d.{docs_year_col} AS BIGINT) AS y
                    FROM doc_clusters dc
                    JOIN {docs_table} d
                      ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
                    WHERE 1=1
                    {null_filter_sql}
                ),
                n_cy AS (
                    SELECT topic, y, COUNT(*) AS n_cy
                    FROM cl_years
                    GROUP BY topic, y
                ),
                n_c AS (
                    SELECT topic, SUM(n_cy) AS n_c
                    FROM n_cy
                    GROUP BY topic
                ),
                shares AS (
                    SELECT
                        n_cy.topic,
                        n_cy.y,
                        n_cy.n_cy,
                        n_c.n_c,
                        (n_cy.n_cy::DOUBLE / NULLIF(n_c.n_c, 0)) AS share
                    FROM n_cy
                    JOIN n_c USING(topic)
                ),
                ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY topic
                            ORDER BY share DESC, n_cy DESC, y
                        ) AS rn
                    FROM shares
                )
                SELECT
                    topic,
                    share AS metric,
                    n_c   AS n_c,
                    y     AS peak_year,
                    n_cy  AS peak_count
                FROM ranked
                WHERE rn = 1
                ORDER BY topic;
                """

        per_cluster = con.execute(per_cluster_sql).df()

        return float(mean_peak_share), per_cluster

    finally:
        con.close()


def count_authors_with_pubs(
    duckdb_path: str,
    min_pubs: int,
    auth_seqn: None | Literal["first", "last"] = None,

    auth_doc_table: str = "auth_aff_doc",
    auth_col: str = "auth_id",
    doc_col: str = "doc_id",
    auth_seqn_col: str = "auth_seqn",
) -> int:
    con = duckdb.connect(duckdb_path, read_only=True)

    if auth_seqn is None:
        seqn_q = ""
    elif auth_seqn == "first":
        seqn_q = f"WHERE t.{auth_seqn_col} = 0"
    else:
        seqn_q = f"""
            WHERE t.{auth_seqn_col} = (
                SELECT
                    MAX(t2.{auth_seqn_col})
                FROM {auth_doc_table} t2
                WHERE t2.{doc_col} = t.{doc_col}
            )    
        """

    try:
        sql_q = f"""
            SELECT
                COUNT(*) as n_authors
            FROM(
                SELECT
                    DISTINCT t.{auth_col} as auths
                FROM {auth_doc_table} t
                {seqn_q}
                GROUP BY t.{auth_col}
                HAVING COUNT(t.{doc_col}) > {min_pubs}
            )
        """

        n_authors = con.execute(sql_q).fetchone()[0]
        return int(n_authors)

    finally:
        con.close()


def count_affiliations_with_pubs(
    duckdb_path: str,
    min_pubs: int,

    aff_doc_table: str = "auth_aff_doc",
    aff_col: str = "aff_id",
    doc_col: str = "doc_id",
) -> int:
    con = duckdb.connect(duckdb_path, read_only=True)

    try:
        sql_q = f"""
            SELECT
                COUNT(*) as n_affiliations
            FROM(
                SELECT
                    DISTINCT t.{aff_col} as affs
                FROM {aff_doc_table} t
                GROUP BY t.{aff_col}
                HAVING COUNT(t.{doc_col}) > {min_pubs}
            )
        """

        n_affiliations = con.execute(sql_q).fetchone()[0]
        return int(n_affiliations)

    finally:
        con.close()


def count_clusters_with_pubs(
    df_clusters: pd.DataFrame,
    min_cluster_docs: int,

    df_doc_col: str = "id",
    df_topic_col: str = "topic",
    df_is_core_col: str = "is_core",
):
    result = (
        df_clusters.groupby([df_is_core_col, df_topic_col])[df_doc_col]
        .nunique()
        .reset_index(name="id_count")
        .query("id_count > @min_cluster_docs")
        .groupby(df_is_core_col)[df_topic_col]
        .nunique()
        .reset_index(name="topic_count")
    )
    return result


def get_interpretable_metrics(duckdb_path: str, _clusters_df: pd.DataFrame, logger_name: str = "logger"):
    logger = logging.getLogger(logger_name)

    timings = {}
    metrics = {}

    clusters = _clusters_df[_clusters_df["topic"] != -1].reset_index(drop=True)

    unique_authors = count_authors_with_pubs(duckdb_path, 0)
    unique_affiliations = count_affiliations_with_pubs(duckdb_path, 0)

    t0 = time.perf_counter()
    try:
        val = auth_cluster_concentration(duckdb_path, clusters, min_auth_pubs=2, return_per_author=False)
        n_eval = count_authors_with_pubs(duckdb_path, min_pubs=2)
        metrics["AACC"] = f"{val:.4f}"
        metrics["auth_ratio"] = f"{n_eval}/{unique_authors}"
    except Exception as e:
        logger.error("--------AACC--------" + "\n" + str(e))
        metrics["AACC"] = f"-1"
    timings["AACC"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = auth_cluster_concentration(duckdb_path, clusters, min_auth_pubs=2, auth_order="first",
                                              return_per_author=False)
        n_eval = count_authors_with_pubs(duckdb_path, min_pubs=2, auth_seqn="first")
        metrics["AACC_first"] = f"{val:.4f}"
        metrics["auth_first_ratio"] = f"{n_eval}/{unique_authors}"
    except Exception as e:
        logger.error("--------AACC_first--------" + "\n" + str(e))
        metrics["AACC_first"] = f"-1"
    timings["AACC_first"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = auth_cluster_concentration(duckdb_path, clusters, min_auth_pubs=2, auth_order="last",
                                                 return_per_author=False)
        n_eval = count_authors_with_pubs(duckdb_path, min_pubs=2, auth_seqn="last")
        metrics["AACC_last"] = f"{val:.4f}"
        metrics["auth_last_ratio"] = f"{n_eval}/{unique_authors}"
    except Exception as e:
        logger.error("--------AACC_last--------" + "\n" + str(e))
        metrics["AACC_last"] = f"-1"
    timings["AACC_last"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = aff_cluster_concentration(duckdb_path, clusters, min_aff_pubs=2, return_per_aff=False)
        n_eval = count_affiliations_with_pubs(duckdb_path, min_pubs=2)
        metrics["AAffCC"] = f"{val:.4f}"
        metrics["aff_ratio"] = f"{n_eval}/{unique_affiliations}"
    except Exception as e:
        logger.error("--------AAffCC--------" + "\n" + str(e))
        metrics["AAffCC"] = f"-1"
    timings["AAffCC"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = k_cluster_auth_rate(duckdb_path, clusters, K=1, min_auth_pubs=2,
                                  return_distribution=False)
        n_eval = count_authors_with_pubs(duckdb_path, min_pubs=2)
        metrics["CAR_1"] = f"{val:.4f}"
        metrics["CAR_1_ratio"] = f"{n_eval}/{unique_authors}"
    except Exception as e:
        logger.error("--------CAR_1--------" + "\n" + str(e))
        metrics["CAR_1"] = f"-1"
    timings["CAR_1"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = k_cluster_auth_rate(duckdb_path, clusters, K=2, min_auth_pubs=3,
                                          return_distribution=False)
        n_eval = count_authors_with_pubs(duckdb_path, min_pubs=3)
        metrics["CAR_2"] = f"{val:.4f}"
        metrics["CAR_2_ratio"] = f"{n_eval}/{unique_authors}"
    except Exception as e:
        logger.error("--------CAR_2--------" + "\n" + str(e))
        metrics["CAR_2"] = f"-1"
    timings["CAR_2"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = auth_overlap(duckdb_path, clusters)[0]
        metrics["AJ"] = f"{val:.8f}"
    except Exception as e:
        logger.error("--------AJ--------" + "\n" + str(e))
        metrics["AJ"] = f"-1"
    timings["AJ"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = cluster_purity(duckdb_path, clusters, field="source", return_per_cluster=False)
        metrics["CP_source"] = f"{val:.4f}"
    except Exception as e:
        logger.error("--------CP_source--------" + "\n" + str(e))
        metrics["CP_source"] = f"-1"
    timings["CP_source"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = cluster_purity(duckdb_path, clusters, field="doctype", return_per_cluster=False)
        metrics["CP_doctype"] = f"{val:.4f}"
    except Exception as e:
        logger.error("--------CP_doctype--------" + "\n" + str(e))
        metrics["CP_doctype"] = f"-1"
    timings["CP_doctype"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = year_iqr(duckdb_path, clusters, return_per_cluster=False)
        metrics["YearIQR"] = f"{val:.4f}"
    except Exception as e:
        logger.error("--------YearIQR--------" + "\n" + str(e))
        metrics["YearIQR"] = f"-1"
    timings["YearIQR"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = entropy_norm(duckdb_path, clusters, return_per_cluster=False)
        metrics["AH"] = f"{val:.4f}"
    except Exception as e:
        logger.error("--------AH--------" + "\n" + str(e))
        metrics["AH"] = f"-1"
    timings["AH"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        val = year_peak_share(duckdb_path, clusters, return_per_cluster=False)
        metrics["YPS"] = f"{val:.4f}"
    except Exception as e:
        logger.error("--------YPS--------" + "\n" + str(e))
        metrics["YPS"] = f"-1"
    timings["YPS"] = str(int(time.perf_counter() - t0))

    return timings, metrics
