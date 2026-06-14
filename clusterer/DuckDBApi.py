import duckdb


def qident(name: str) -> str:
    """
    Безопасно экранирует идентификатор (schema/table/column) для DuckDB.
    """
    return '"' + name.replace('"', '""') + '"'


def save_df_to_duckdb(con: duckdb.DuckDBPyConnection, df, table_name: str):
    schema = None
    table = table_name
    if "." in table_name:
        schema, table = table_name.split(".", 1)

    tmp_view = "_tmp_metric_df"
    con.register(tmp_view, df)

    if schema:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {qident(schema)}")
        fqtn = f"{qident(schema)}.{qident(table)}"
    else:
        fqtn = qident(table)

    con.execute(f"CREATE OR REPLACE TABLE {fqtn} AS SELECT * FROM {qident(tmp_view)}")

    con.unregister(tmp_view)
