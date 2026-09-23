"""Reads the database response views."""

import io
from typing import Any

import polars as pl
import psycopg
from psycopg.rows import dict_row

VIEWS = ("v_response_acc", "v_response_score")

CELLS_COPY = """COPY (
	SELECT a.user_id, a.beatmap_id, a.rate_group, a.keys, a.response AS acc, s.response AS score
	FROM v_response_acc a
	JOIN v_response_score s USING (user_id, beatmap_id, rate_group)
	WHERE a.in_random
) TO STDOUT (FORMAT csv, HEADER)"""

_CELLS_SCHEMA = {
	"user_id": pl.Int64,
	"beatmap_id": pl.Int64,
	"rate_group": pl.String,
	"keys": pl.Int64,
	"acc": pl.Float64,
	"score": pl.Float64,
}


def connect(dsn: str) -> psycopg.Connection:
	"""Opens a connection whose transactions are read-only."""
	# extra_float_digits=3 makes float8 text output round-trip exactly on any server version.
	return psycopg.connect(
		dsn, options="-c default_transaction_read_only=on -c extra_float_digits=3"
	)


def fetch_cells(conn: psycopg.Connection) -> pl.DataFrame:
	"""Every random-pool cell with both responses, streamed through COPY (millions of rows)."""
	buffer = io.BytesIO()
	with conn.cursor() as cur, cur.copy(CELLS_COPY) as copy:
		for block in copy:
			buffer.write(block)
	buffer.seek(0)
	return pl.read_csv(buffer, schema=_CELLS_SCHEMA)


def fetch_view_definitions(conn: psycopg.Connection) -> dict[str, str]:
	"""Current SQL of each view. The views live outside this repository and can change."""
	with conn.cursor() as cur:
		rows = cur.execute(
			"SELECT v, pg_get_viewdef(v::regclass, true) FROM unnest(%s::text[]) AS v",
			(list(VIEWS),),
		).fetchall()
	return dict(rows)


def fetch_ingest_log(conn: psycopg.Connection) -> list[dict[str, Any]]:
	"""The dump files loaded into the database, for the manifest."""
	with conn.cursor(row_factory=dict_row) as cur:
		return cur.execute(
			"SELECT dump_file, member, target_table, in_top, in_random, rows_written"
			" FROM ingest_log ORDER BY id"
		).fetchall()
