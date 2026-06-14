import logging
import time
import pandas as pd
from typing import Tuple, Union, Literal
import duckdb
from DuckDBApi import save_df_to_duckdb


def _register_doc_clusters_df(
    con: duckdb.DuckDBPyConnection,
    df_clusters: pd.DataFrame,
    cluster_cols: list[str],
) -> None:
    """
    Регистрирует в DuckDB только нужные колонки из df_clusters.

    cluster_cols:
        список колонок, которые нужно передать в DuckDB.
        Например: ["id", "topic", "is_core"].
    """
    relation_name = "doc_clusters_df"

    missing_columns = set(cluster_cols) - set(df_clusters.columns)

    if missing_columns:
        raise ValueError(
            "В df_clusters отсутствуют обязательные колонки: "
            f"{sorted(missing_columns)}"
        )

    existing_tables = {
        row[0]
        for row in con.execute("PRAGMA show_tables").fetchall()
    }

    if relation_name in existing_tables:
        con.unregister(relation_name)

    con.register(
        relation_name,
        df_clusters[cluster_cols],
    )


def impact_score(
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
        docs_cited_col: str = "cited",
) -> dict:
    """
    Для каждого кластера считает z-оценку Impact (подробности в "отчет прототипы.docx")

    Возвращает Impact dict:
      {topic: Z_Impact(topic)}
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
        WITH docs AS (
            SELECT
                d.{docs_eid_col} AS eid,
                d.{docs_year_col} AS year,
                d.{docs_cited_col}::DOUBLE AS cited
            FROM {docs_table} d
            WHERE d.{docs_eid_col} IS NOT NULL
                AND d.{docs_year_col} IS NOT NULL
                AND d.{docs_cited_col} IS NOT NULL
        ),

        q99 AS (
            SELECT
                year,
                quantile_cont(cited, 0.99) AS q99_year
            FROM docs
            GROUP BY year
        ),

        trimmed AS (
            SELECT
                d.eid,
                d.year,
                d.cited,
                least(d.cited, q.q99_year) AS cited_trim
            FROM docs d
            JOIN q99 q USING(year)
        ),

        mu AS (
            SELECT
                year,
                avg(cited_trim) AS mu_trim_year
            FROM trimmed
            GROUP BY year
        ),

        ncs_all AS (
            SELECT
                t.eid,
                t.year,
                t.cited,
                m.mu_trim_year,
                t.cited / NULLIF(m.mu_trim_year, 0) AS ncs
            FROM trimmed t
            JOIN mu m USING(year)
            WHERE m.mu_trim_year IS NOT NULL
        ),

        doc_clusters AS (
            SELECT
                CAST({df_doc_col} AS VARCHAR) AS id,
                {df_topic_col} AS topic,
                {df_is_core_col}
            FROM doc_clusters_df
            {core_filter_sql}
        ),

        ncs_topics AS (
            SELECT
                tp.topic,
                n.ncs
            FROM ncs_all n
            JOIN doc_clusters tp
              ON n.eid = tp.id
            WHERE n.ncs IS NOT NULL
        ),

        impact AS (
            SELECT
                topic,
                avg(ncs) AS impact_raw
            FROM ncs_topics
            GROUP BY topic
        )

        SELECT
            topic,
            impact_raw,
            (impact_raw - avg(impact_raw) OVER ())
              / NULLIF(stddev_pop(impact_raw) OVER (), 0) AS impact_z
        FROM impact
        ORDER BY topic
        """

        res = con.execute(sql_q).df()

        if res is None:
            raise Exception("Z_Impact не было посчитано")

        return dict(zip(res["topic"], res["impact_z"]))

    finally:
        con.close()


def top_k_pubs(
        duckdb_path: str,
        df_clusters: pd.DataFrame,
        K: int,

        # Колонки датафрейма (уже без -1 кластера)
        df_doc_col: str = "id",
        df_topic_col: str = "topic",
        df_is_core_col: str = "is_core",
        only_core: bool = False,  # True = only core, False = all

        # duckdb таблица и колонки
        docs_table: str = "docs",
        docs_eid_col: str = "eid",
        docs_year_col: str = "year",
        docs_cited_col: str = "cited",
):
    """
    Для каждого кластера считает z-оценку TOPK (подробности в "отчет прототипы.docx")

    Возвращает TOPK dict:
      {topic: TOPK(topic)}
    """
    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter_sql = f"WHERE {df_is_core_col} = TRUE" if only_core else ""
        threshold = 100 - K

        sql_q = f"""
        WITH docs AS (
            SELECT
                d.{docs_eid_col} AS eid,
                d.{docs_year_col} AS year,
                d.{docs_cited_col}::DOUBLE AS cited
            FROM {docs_table} d
            WHERE d.{docs_eid_col} IS NOT NULL
                AND d.{docs_year_col} IS NOT NULL
                AND d.{docs_cited_col} IS NOT NULL
        ),

        pr_all AS (
            SELECT
                eid,
                100.0 * cume_dist() OVER (PARTITION BY year ORDER BY cited) AS pr
            FROM docs
        ),

        doc_clusters AS (
            SELECT
                CAST({df_doc_col} AS VARCHAR) AS id,
                {df_topic_col} AS topic,
                {df_is_core_col}
            FROM doc_clusters_df
            {core_filter_sql}
        ),

        pr_topics AS (
            SELECT
                tp.topic,
                p.pr
            FROM pr_all p
            JOIN doc_clusters tp
              ON p.eid = tp.id
            WHERE p.pr IS NOT NULL
        ),

        topk AS (
            SELECT
                topic,
                avg(CASE WHEN pr >= {threshold} THEN 1.0 ELSE 0.0 END) AS topk_raw,
                count(*) AS n_t
            FROM pr_topics
            GROUP BY topic
        )

        SELECT
            topic,
            topk_raw,
            (topk_raw - avg(topk_raw) OVER ())
              / NULLIF(stddev_pop(topk_raw) OVER (), 0) AS topk_z,
            n_t
        FROM topk
        ORDER BY topic
        """

        res = con.execute(sql_q).df()

        if res is None:
            raise Exception(f"Z_TOP{K} не было посчитано")

        return dict(zip(res["topic"], res["topk_z"]))

    finally:
        con.close()


def cs_mean(
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
        docs_cs_col: str = "sjr",
):
    """
    Для каждого кластера считает z-оценку CS_mean (подробности в "отчет прототипы.docx")

    Возвращает CS_mean dict:
      {topic: CS_mean(topic)}

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
            SELECT
                CAST({df_doc_col} AS VARCHAR) AS id,
                {df_topic_col} AS topic,
                {df_is_core_col}
            FROM doc_clusters_df
            {core_filter_sql}
        ),

        docs AS (
            SELECT
                d.{docs_eid_col} AS eid,
                d.{docs_cs_col}::DOUBLE AS cs
            FROM {docs_table} d
            WHERE d.{docs_eid_col} IS NOT NULL
                AND d.{docs_cs_col} IS NOT NULL
        ),

        topic_docs AS (
            SELECT
                t.topic,
                COALESCE(d.cs::DOUBLE, 0.0) AS cs
            FROM docs d
            JOIN doc_clusters t
              ON d.eid = t.id
            WHERE d.eid IS NOT NULL
        ),

        cs AS (
            SELECT
                topic,
                avg(cs) AS cs_mean,
            FROM topic_docs
            GROUP BY topic
        )

        SELECT
            topic,
            cs_mean,
            (cs_mean - avg(cs_mean) OVER ())
              / NULLIF(stddev_pop(cs_mean) OVER (), 0) AS cs_z,
        FROM cs
        ORDER BY topic
        """

        res = con.execute(sql_q).df()

        if res is None:
            raise Exception(f"CS_mean не было посчитано")

        return dict(zip(res["topic"], res["cs_z"]))

    finally:
        con.close()


def ztrend_cs_sum(
        duckdb_path: str,
        df_clusters: pd.DataFrame,

        # Колонки df_clusters (уже без -1 кластера)
        df_doc_col: str = "id",  # eid
        df_topic_col: str = "topic",
        df_is_core_col: str = "is_core",
        only_core: bool = False,

        # DuckDB: docs
        docs_table: str = "docs",
        docs_eid_col: str = "eid",
        docs_year_col: str = "year",
        docs_cs_col: str = "sjr",

        # Окно лет
        y1: int | None = None,  # если None -> min(year) по выбранным документам
        ymax: int | None = None,  # если None -> max(year) по выбранным документам
        n_last_years: int = 3,  # последний период = n последних лет до ymax (включая)

        # Веса тренда
        w_full: float = 0.4,
        w_last: float = 0.6,
) -> pd.DataFrame:
    """
    Возвращает:
      zscore_trend_mean (всегда 0 при std>0, но оставлено для совместимости формата),
      и опционально df с колонками:
        [topic, metric, beta_full, beta_last]

    Примечания:
    - Для регрессии используются ВСЕ годы окна; в годы без статей cs_sum=0 => log(1)=0.
    - Особое правило: если в last-окне ровно один год с cs_sum>0, то beta_last = 0.
    - Если std(Trend_CS)==0, ztrend будет NULL (NaN).
    """

    if n_last_years <= 0:
        raise ValueError("n_last_years должен быть >= 1")

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter = f"WHERE {df_is_core_col} = TRUE" if only_core else ""

        yr_bounds_sql = f"""
        WITH dc AS (
            SELECT CAST({df_doc_col} AS VARCHAR) AS doc_key,
                   {df_topic_col} AS topic,
                   {df_is_core_col} AS is_core
            FROM doc_clusters_df
            {core_filter}
        )
        SELECT
            MIN(d.{docs_year_col}) AS y_min,
            MAX(d.{docs_year_col}) AS y_max
        FROM dc
        JOIN {docs_table} d
          ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
        WHERE d.{docs_year_col} IS NOT NULL;
        """
        if y1 is None or ymax is None:
            y_min_db, y_max_db = con.execute(yr_bounds_sql).fetchone()
            if y_min_db is None or y_max_db is None:
                raise RuntimeError("Не удалось определить границы лет: нет пересечения docs и df_clusters по eid.")

            y1 = int(y_min_db)
            ymax = int(y_max_db)

        if y1 > ymax:
            raise ValueError(f"Некорректное окно лет: y1={y1} > ymax={ymax}")

        y_last_start = ymax - n_last_years + 1

        sql = f"""
            WITH dc AS (
                SELECT
                    CAST({df_doc_col} AS VARCHAR) AS doc_key,
                    {df_topic_col} AS topic,
                    {df_is_core_col} AS is_core
                FROM doc_clusters_df
                {core_filter}
            ),
            docs_in AS (
                SELECT
                    dc.topic,
                    d.{docs_year_col}::INT AS year,
                    COALESCE(d.{docs_cs_col}, 0)::DOUBLE AS cs
                FROM dc
                JOIN {docs_table} d
                  ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
                WHERE d.{docs_year_col} IS NOT NULL
                  AND d.{docs_year_col} BETWEEN {y1} AND {ymax}
            ),
            topics AS (
                SELECT DISTINCT topic FROM dc
            ),
            years AS (
                SELECT * FROM range({y1}, {ymax}+1) AS t(year)
            ),
            grid AS (
                SELECT topics.topic, years.year
                FROM topics
                CROSS JOIN years
            ),
            cs_sum AS (
                SELECT
                    topic,
                    year,
                    SUM(cs) AS cs_sum
                FROM docs_in
                GROUP BY topic, year
            ),
            panel AS (
                SELECT
                    g.topic,
                    g.year,
                    COALESCE(c.cs_sum, 0)::DOUBLE AS cs_sum,
                    LN(1.0 + COALESCE(c.cs_sum, 0)::DOUBLE) AS z_cs,
                    CASE WHEN COALESCE(c.cs_sum, 0) > 0 THEN 1 ELSE 0 END AS has_data
                FROM grid g
                LEFT JOIN cs_sum c
                  ON c.topic = g.topic AND c.year = g.year
            ),

            -- агрегаты для full окна
            full_agg AS (
                SELECT
                    topic,
                    COUNT(*)::DOUBLE AS n,
                    SUM(year)::DOUBLE AS sx,
                    SUM(year*year)::DOUBLE AS sxx,
                    SUM(z_cs)::DOUBLE AS sy,
                    SUM(year*z_cs)::DOUBLE AS sxy,
                    SUM(cs_sum)::DOUBLE AS cs_sum_total,
                    SUM(has_data)::INT AS years_with_data_full
                FROM panel
                GROUP BY topic
            ),
            beta_full AS (
                SELECT
                    topic,
                    CASE
                        WHEN (sxx - (sx*sx)/n) = 0 THEN 0.0
                        ELSE (sxy - (sx*sy)/n) / (sxx - (sx*sx)/n)
                    END AS beta_full,
                    cs_sum_total,
                    years_with_data_full
                FROM full_agg
            ),

            -- агрегаты для last окна
            last_panel AS (
                SELECT *
                FROM panel
                WHERE year BETWEEN {y_last_start} AND {ymax}
            ),
            last_agg AS (
                SELECT
                    topic,
                    COUNT(*)::DOUBLE AS n,
                    SUM(year)::DOUBLE AS sx,
                    SUM(year*year)::DOUBLE AS sxx,
                    SUM(z_cs)::DOUBLE AS sy,
                    SUM(year*z_cs)::DOUBLE AS sxy,
                    SUM(has_data)::INT AS years_with_data_last
                FROM last_panel
                GROUP BY topic
            ),
            beta_last AS (
                SELECT
                    topic,
                    CASE
                        WHEN years_with_data_last = 1 THEN 0.0
                        WHEN (sxx - (sx*sx)/n) = 0 THEN 0.0
                        ELSE (sxy - (sx*sy)/n) / (sxx - (sx*sx)/n)
                    END AS beta_last,
                    years_with_data_last
                FROM last_agg
            ),

            trend AS (
                SELECT
                    f.topic,
                    f.beta_full,
                    l.beta_last,
                    ({w_full})*f.beta_full + ({w_last})*l.beta_last AS trend,
                    f.cs_sum_total,
                    f.years_with_data_full,
                    l.years_with_data_last
                FROM beta_full f
                JOIN beta_last l USING(topic)
            ),
            stats AS (
                SELECT
                    AVG(trend)::DOUBLE AS mean_trend,
                    STDDEV_SAMP(trend)::DOUBLE AS std_trend
                FROM trend
            )
            SELECT
                t.topic,
                CASE
                    WHEN s.std_trend IS NULL OR s.std_trend = 0 THEN NULL
                    ELSE (t.trend - s.mean_trend) / s.std_trend
                END AS metric,
                t.beta_full,
                t.beta_last,
            FROM trend t
            CROSS JOIN stats s
            ORDER BY t.topic;
            """

        per_topic = con.execute(sql)

        if not per_topic:
            raise Exception("Z_trend_cs_t_impulse не был посчитан")

        per_topic = per_topic.df()

        return per_topic

    finally:
        con.close()


def ztrend_diversity(
        duckdb_path: str,
        df_clusters: pd.DataFrame,

        # df_clusters
        df_doc_col: str = "id",
        df_topic_col: str = "topic",
        df_is_core_col: str = "is_core",
        only_core: bool = False,

        # docs
        docs_table: str = "docs",
        docs_eid_col: str = "eid",
        docs_year_col: str = "year",

        # окно лет
        y1: int | None = None,
        ymax: int | None = None,
        n_last_years: int = 3,

        # веса
        w_full: float = 0.4,
        w_last: float = 0.6,

        entity_rows_cte_sql: str = "",

        beta_last_zero_if_one_data_year: bool = True,
) -> pd.DataFrame:
    """
    Универсальная метрика:
      D_t(y)=1 - Σ_e (n_e,t(y)/n_t(y))^2, если n_t(y)>0 иначе 0
      Z_t(y)=log(1 + D_t(y))
      beta_full / beta_last по Z_t(y) ~ year
      Trend = w_full*beta_full + w_last*beta_last
      ztrend = z-score(Trend) по топикам

    entity_rows_cte_sql должен быть валидным SQL-фрагментом CTE вида:

      entity_rows AS (
        SELECT <topic> AS topic, <year>::INT AS year, <entity_id> AS entity_id
        FROM ...
        WHERE <year> BETWEEN y1 AND ymax
          AND <entity_id> IS NOT NULL
      )

    Возвращает:
      mean_ztrend (обычно ~0 при std>0)
      и опционально df per-topic.
    """
    if not entity_rows_cte_sql.strip():
        raise ValueError("Нужно передать entity_rows_cte_sql")

    if n_last_years <= 0:
        raise ValueError("n_last_years должен быть >= 1")

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        _register_doc_clusters_df(
            con=con,
            df_clusters=df_clusters,
            cluster_cols=[df_doc_col, df_topic_col, df_is_core_col],
        )

        core_filter = f"WHERE {df_is_core_col} = TRUE" if only_core else ""

        yr_bounds_sql = f"""
        WITH dc AS (
            SELECT CAST({df_doc_col} AS VARCHAR) AS doc_key,
                   {df_topic_col} AS topic,
                   {df_is_core_col} AS is_core
            FROM doc_clusters_df
            {core_filter}
        )
        SELECT MIN(d.{docs_year_col}) AS y_min,
               MAX(d.{docs_year_col}) AS y_max
        FROM dc
        JOIN {docs_table} d
          ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
        WHERE d.{docs_year_col} IS NOT NULL;
        """
        if y1 is None or ymax is None:
            y_min_db, y_max_db = con.execute(yr_bounds_sql).fetchone()
            if y_min_db is None or y_max_db is None:
                raise RuntimeError("Нет пересечения docs и df_clusters по eid или нет годов.")

            y1 = int(y_min_db)
            ymax = int(y_max_db)
        if y1 > ymax:
            raise ValueError(f"Некорректное окно лет: y1={y1} > ymax={ymax}")

        y_last_start = ymax - n_last_years + 1

        beta_last_rule = "WHEN years_with_data_last = 1 THEN 0.0" if beta_last_zero_if_one_data_year else ""

        sql = f"""
        WITH dc AS (
            SELECT
                CAST({df_doc_col} AS VARCHAR) AS doc_key,
                {df_topic_col} AS topic,
                {df_is_core_col} AS is_core
            FROM doc_clusters_df
            {core_filter}
        ),
        docs_in AS (
            SELECT
                dc.topic,
                d.{docs_year_col}::INT AS year,
                CAST(d.{docs_eid_col} AS VARCHAR) AS doc_key
            FROM dc
            JOIN {docs_table} d
              ON CAST(d.{docs_eid_col} AS VARCHAR) = dc.doc_key
            WHERE d.{docs_year_col} IS NOT NULL
              AND d.{docs_year_col} BETWEEN {y1} AND {ymax}
        ),

        -- пользовательский CTE: (topic, year, entity_id)
        {entity_rows_cte_sql},

        topics AS (
            SELECT DISTINCT topic FROM dc
        ),
        years AS (
            SELECT * FROM range({y1}, {ymax}+1) AS t(year)
        ),
        grid AS (
            SELECT topics.topic, years.year
            FROM topics
            CROSS JOIN years
        ),

        -- n_e,t(y)
        n_ety AS (
            SELECT
                er.topic,
                er.year,
                er.entity_id,
                COUNT(*)::DOUBLE AS n_ety
            FROM entity_rows er
            GROUP BY er.topic, er.year, er.entity_id
        ),
        -- n_t(y)
        n_ty AS (
            SELECT topic, year, SUM(n_ety)::DOUBLE AS n_ty
            FROM n_ety
            GROUP BY topic, year
        ),
        -- HHI = sum (n_ety/n_ty)^2
        hhi AS (
            SELECT
                n.topic,
                n.year,
                SUM( (n.n_ety / t.n_ty) * (n.n_ety / t.n_ty) )::DOUBLE AS hhi
            FROM n_ety n
            JOIN n_ty t USING(topic, year)
            GROUP BY n.topic, n.year
        ),

        metric AS (
            SELECT
                g.topic,
                g.year,
                COALESCE(t.n_ty, 0)::DOUBLE AS n_ty,
                CASE
                    WHEN COALESCE(t.n_ty, 0) > 0 THEN 1.0 - COALESCE(h.hhi, 0)
                    ELSE 0.0
                END AS d_div,
                LN(1.0 + CASE
                    WHEN COALESCE(t.n_ty, 0) > 0 THEN 1.0 - COALESCE(h.hhi, 0)
                    ELSE 0.0
                END) AS z_metric,
                CASE WHEN COALESCE(t.n_ty, 0) > 0 THEN 1 ELSE 0 END AS has_data
            FROM grid g
            LEFT JOIN n_ty t
              ON t.topic = g.topic AND t.year = g.year
            LEFT JOIN hhi h
              ON h.topic = g.topic AND h.year = g.year
        ),

        full_agg AS (
            SELECT
                topic,
                COUNT(*)::DOUBLE AS n,
                SUM(year)::DOUBLE AS sx,
                SUM(year*year)::DOUBLE AS sxx,
                SUM(z_metric)::DOUBLE AS sy,
                SUM(year*z_metric)::DOUBLE AS sxy,
                SUM(has_data)::INT AS years_with_data_full
            FROM metric
            GROUP BY topic
        ),
        beta_full AS (
            SELECT
                topic,
                CASE
                    WHEN (sxx - (sx*sx)/n) = 0 THEN 0.0
                    ELSE (sxy - (sx*sy)/n) / (sxx - (sx*sx)/n)
                END AS beta_full,
                years_with_data_full
            FROM full_agg
        ),

        last_metric AS (
            SELECT *
            FROM metric
            WHERE year BETWEEN {y_last_start} AND {ymax}
        ),
        last_agg AS (
            SELECT
                topic,
                COUNT(*)::DOUBLE AS n,
                SUM(year)::DOUBLE AS sx,
                SUM(year*year)::DOUBLE AS sxx,
                SUM(z_metric)::DOUBLE AS sy,
                SUM(year*z_metric)::DOUBLE AS sxy,
                SUM(has_data)::INT AS years_with_data_last
            FROM last_metric
            GROUP BY topic
        ),
        beta_last AS (
            SELECT
                topic,
                CASE
                    {beta_last_rule}
                    WHEN (sxx - (sx*sx)/n) = 0 THEN 0.0
                    ELSE (sxy - (sx*sy)/n) / (sxx - (sx*sx)/n)
                END AS beta_last,
                years_with_data_last
            FROM last_agg
        ),

        trend AS (
            SELECT
                f.topic,
                f.beta_full,
                l.beta_last,
                ({w_full})*f.beta_full + ({w_last})*l.beta_last AS trend,
                f.years_with_data_full,
                l.years_with_data_last
            FROM beta_full f
            JOIN beta_last l USING(topic)
        ),
        stats AS (
            SELECT AVG(trend)::DOUBLE AS mean_trend,
                   STDDEV_SAMP(trend)::DOUBLE AS std_trend
            FROM trend
        )
        SELECT
            t.topic,
            CASE
                WHEN s.std_trend IS NULL OR s.std_trend = 0 THEN NULL
                ELSE (t.trend - s.mean_trend) / s.std_trend
            END AS metric,
            t.beta_full,
            t.beta_last,
        FROM trend t
        CROSS JOIN stats s
        ORDER BY t.topic;
        """

        per_topic = con.execute(sql)

        if not per_topic:
            raise Exception("Z_diversity (один из двух) не был посчитан")

        per_topic = per_topic.df()

        return per_topic

    finally:
        con.close()


def make_entity_rows_auth_cte(
        auth_aff_docs_table: str = "auth_aff_doc",
        doc_id_col: str = "doc_id",
        auth_id_col: str = "auth_id",
) -> str:
    return f"""
    entity_rows AS (
        SELECT
            di.topic AS topic,
            di.year  AS year,
            aad.{auth_id_col} AS entity_id
        FROM docs_in di
        JOIN {auth_aff_docs_table} aad
          ON CAST(aad.{doc_id_col} AS VARCHAR) = di.doc_key
        WHERE aad.{auth_id_col} IS NOT NULL
    )
    """


def make_entity_rows_country_cte(
        auth_aff_docs_table: str = "auth_aff_doc",
        doc_id_col: str = "doc_id",
        aff_id_col: str = "aff_id",
        affiliations_table: str = "affiliations",
        affiliations_id_col: str = "id",
        affiliations_country_col: str = "Country",
) -> str:
    return f"""
    entity_rows AS (
        SELECT
            di.topic AS topic,
            di.year  AS year,
            a.{affiliations_country_col} AS entity_id
        FROM docs_in di
        JOIN {auth_aff_docs_table} aad
          ON CAST(aad.{doc_id_col} AS VARCHAR) = di.doc_key
        JOIN {affiliations_table} a
          ON a.{affiliations_id_col} = aad.{aff_id_col}
        WHERE a.{affiliations_country_col} IS NOT NULL
          AND TRIM(a.{affiliations_country_col}) <> ''
    )
    """


def build_impact_index_df(
        z_impact: dict,
        z_topk: dict,
        z_cs_mean: dict,
        w1: float,
        w2: float,
        w3: float,
        fill_missing: float = 0.0,
        sort_desc: bool = True,
) -> Tuple[float, pd.DataFrame]:
    topics = set(z_impact) | set(z_topk) | set(z_cs_mean)

    rows = []
    for t in topics:
        z_imp = z_impact.get(t, fill_missing)
        z_top = z_topk.get(t, fill_missing)
        z_cs = z_cs_mean.get(t, fill_missing)
        impact_index = w1 * z_imp + w2 * z_top + w3 * z_cs
        rows.append((t, impact_index))

    out = pd.DataFrame(rows, columns=["topic", "metric"])
    mean_impact_index = float(out["metric"].mean()) if not out.empty else float("nan")

    if sort_desc:
        out = out.sort_values("metric", ascending=False, ignore_index=True)

    return mean_impact_index, out


def build_impulse_index(
        df_cs: pd.DataFrame,  # from ztrend_cs_sum(..., return_per_topic=True)
        df_auth: pd.DataFrame,  # from ztrend_diversity(... authors ..., return_per_topic=True)
        df_country: pd.DataFrame,  # from ztrend_diversity(... countries ..., return_per_topic=True)
        w1: float,
        w2: float,
        w3: float,
        fill_missing: float = 0.0,
        sort_desc: bool = True,
        topic_col: str = "topic",
        value_col: str = "metric",  # у наших метрик это "ztrend"
) -> Tuple[float, pd.DataFrame]:
    """
    ImpulseIndex_t = w1*ZTrend_CS_t + w2*ZTrend_AuthDiv_t + w3*ZTrend_CountryDiv_t

    Возвращает:
      mean_index, df(topic, metric)
    """
    a = df_cs[[topic_col, value_col]].rename(columns={value_col: "z_cs"})
    b = df_auth[[topic_col, value_col]].rename(columns={value_col: "z_auth"})
    c = df_country[[topic_col, value_col]].rename(columns={value_col: "z_country"})

    out = a.merge(b, on=topic_col, how="outer").merge(c, on=topic_col, how="outer")
    out[["z_cs", "z_auth", "z_country"]] = out[["z_cs", "z_auth", "z_country"]].fillna(fill_missing)

    out["metric"] = w1 * out["z_cs"] + w2 * out["z_auth"] + w3 * out["z_country"]

    mean_metric = float(out["metric"].mean()) if not out.empty else float("nan")
    out = out[[topic_col, "metric"]]

    if sort_desc:
        out = out.sort_values("metric", ascending=False, ignore_index=True)

    return mean_metric, out


def get_topic_metrics(duckdb_path: str, _clusters_df: pd.DataFrame,
                      impact_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
                      impulse_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
                      logger_name: str = "logger"
):
    timings = {}
    logger = logging.getLogger(logger_name)

    con = duckdb.connect("res/to_topic_viewer.duckdb")

    clusters = _clusters_df[_clusters_df["topic"] != -1].reset_index(drop=True)

    t0 = time.perf_counter()
    try:
        z_impact = impact_score(duckdb_path, clusters)
    except Exception as e:
        logger.error("--------z_impact--------" + "\n" + str(e))
        if con:
            con.close()
        raise e
    timings["z_impact"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        z_top10 = top_k_pubs(duckdb_path, clusters, K=10)
    except Exception as e:
        logger.error("--------top10--------" + "\n" + str(e))
        if con:
            con.close()
        raise e
    timings["top10"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        z_cs = cs_mean(duckdb_path, clusters)
    except Exception as e:
        logger.error("--------cs_mean--------" + "\n" + str(e))
        if con:
            con.close()
        raise e
    timings["cs_mean"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        _, df = build_impact_index_df(z_impact, z_top10, z_cs, *impact_weights)
        save_df_to_duckdb(con, df, f"topic_metrics.impact_index")
    except Exception as e:
        logger.error("--------impact_index--------" + "\n" + str(e))
        if con:
            con.close()
        raise e
    timings["impact_index"] = str(int(time.perf_counter() - t0))

    del z_impact, z_top10, z_cs, df

    t0 = time.perf_counter()
    try:
        cs_impulse = ztrend_cs_sum(duckdb_path, clusters)
    except Exception as e:
        logger.error("--------z_cs_impulse--------" + "\n" + str(e))
        if con:
            con.close()
        raise e
    timings["z_cs_impulse"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        auth_impulse = ztrend_diversity(duckdb_path, clusters, entity_rows_cte_sql=make_entity_rows_auth_cte())
    except Exception as e:
        logger.error("--------auth_impulse--------" + "\n" + str(e))
        if con:
            con.close()
        raise e
    timings["auth_impulse"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        country_impulse = ztrend_diversity(duckdb_path, clusters, entity_rows_cte_sql=make_entity_rows_country_cte())
    except Exception as e:
        logger.error("--------country_impulse--------" + "\n" + str(e))
        if con:
            con.close()
        raise e
    timings["country_impulse"] = str(int(time.perf_counter() - t0))

    t0 = time.perf_counter()
    try:
        _, df = build_impulse_index(cs_impulse, auth_impulse, country_impulse, *impulse_weights)
        save_df_to_duckdb(con, df, f"topic_metrics.impulse_index")
    except Exception as e:
        logger.error("--------impulse_index--------" + "\n" + str(e))
        if con:
            con.close()
        raise e
    timings["impulse_index"] = str(int(time.perf_counter() - t0))

    if con:
        con.close()

    return timings
