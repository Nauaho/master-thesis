# benches/age_bench.py
import time
import csv
import psycopg
from base import (
    GraphBenchmarks,
    BenchmarkImportError,
    _timed_repeated,
    _timed_per_input,
    EXPECTED_NODE_COUNT,
    EXPECTED_EDGE_COUNT,
    EXPECTED_EDGE_AGG_COUNT,
)

GRAPH_NAME = "subreddits"


class AGEBenchmark(GraphBenchmarks):
    def __init__(self, port: int):
        self._conn = psycopg.connect(
            host="localhost", port=port, dbname="postgres", user="postgres", password="password",
            autocommit=True,
        )
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS age")
        self._conn.execute("LOAD 'age'")
        self._conn.execute('SET search_path = ag_catalog, "$user", public')
        try:
            self._conn.execute(f"SELECT create_graph('{GRAPH_NAME}')")
        except psycopg.errors.UniqueViolation:
            pass  # graph already exists — fine on a fresh container, defensive otherwise
        self.db_name = "age"

    def _exec(self, cypher: str, out_cols: str = "result agtype"):
        """Wraps a raw Cypher string in AGE's cypher() SQL function call.
        out_cols must match the RETURN arity exactly, e.g. 'a agtype, b agtype'."""
        return self._conn.execute(
            f"SELECT * FROM cypher('{GRAPH_NAME}', $$ {cypher} $$) AS ({out_cols})"
        ).fetchall()

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "\\'")

    def import_data(self):
        try:
            for filename in ["soc-redditHyperlinks-body.tsv", "soc-redditHyperlinks-title.tsv"]:
                with open(f"data/{filename}") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    batch = []
                    for row in reader:
                        batch.append(
                            f"{{source: '{self._escape(row['SOURCE_SUBREDDIT'])}', "
                            f"target: '{self._escape(row['TARGET_SUBREDDIT'])}', "
                            f"sentiment: {float(row['LINK_SENTIMENT'])}}}"
                        )
                        if len(batch) >= 2000:
                            self._flush_hyperlink_batch(batch)
                            batch = []
                    if batch:
                        self._flush_hyperlink_batch(batch)
        except Exception as e:
            raise BenchmarkImportError(f"AGE import failed: {e}") from e

        node_count = self._exec("MATCH (n:Subreddit) RETURN count(n)")[0][0]
        edge_count = self._exec("MATCH ()-[r:LINK_TO]->() RETURN count(r)")[0][0]

        errors = []
        if node_count != EXPECTED_NODE_COUNT:
            errors.append(f"node count: expected {EXPECTED_NODE_COUNT}, got {node_count}")
        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(f"edge count: expected {EXPECTED_EDGE_COUNT}, got {edge_count}")
        if errors:
            raise BenchmarkImportError("AGE import validation failed:\n" + "\n".join(errors))
        return node_count, edge_count

    def _flush_hyperlink_batch(self, batch: list[str]):
        rows_literal = "[" + ", ".join(batch) + "]"
        self._exec(f"""
            UNWIND {rows_literal} AS row
            MERGE (source:Subreddit {{name: row.source}})
            MERGE (target:Subreddit {{name: row.target}})
            CREATE (source)-[r:LINK_TO {{sentimentScore: row.sentiment}}]->(target)
        """, out_cols="result agtype")

    def persist_aggregation(self):
        self._exec("""
            MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)
            WITH s, t, avg(r.sentimentScore) AS sentiment, count(r) AS linkCount
            MERGE (s)-[agg:LINK_TO_AGG]->(t)
            SET agg.sentiment = sentiment, agg.linkCount = linkCount
        """)
        edge_agg_count = self._exec("MATCH ()-[r:LINK_TO_AGG]->() RETURN count(r)")[0][0]
        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:
            raise BenchmarkImportError(
                f"AGE aggregation validation failed: expected {EXPECTED_EDGE_AGG_COUNT}, got {edge_agg_count}"
            )

    def aggregate_graph(self):
        def run():
            return self._exec("""
                MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)
                RETURN s.name, t.name, sum(r.sentimentScore), count(r)
                ORDER BY sum(r.sentimentScore) DESC
            """, out_cols="source agtype, target agtype, sentiment agtype, linkCount agtype")
        return _timed_repeated(run, n=5)

    def common_neighbour_match(self, subreddit_names: list[str]):
        def run(name: str):
            return self._exec(f"""
                MATCH (s:Subreddit {{name: '{self._escape(name)}'}})-[r1:LINK_TO_AGG]->(common:Subreddit)<-[r2:LINK_TO_AGG]-(newFriend:Subreddit)
                WHERE r1.sentiment > 0.5 AND r2.sentiment > 0.5
                  AND s <> newFriend
                  AND NOT (s)-[:LINK_TO_AGG]->(newFriend)
                  AND NOT (newFriend)-[:LINK_TO_AGG]->(s)
                RETURN newFriend.name, r2.sentiment - r1.sentiment
                ORDER BY r2.sentiment - r1.sentiment DESC LIMIT 100
            """, out_cols="newFriend agtype, delta_interest agtype")
        return _timed_per_input(run, subreddit_names)

    def cycle_detection(self, subreddit_names: list[str], category: str = "positive"):
        op = self._sentiment_op(category)
        def run(name: str):
            return self._exec(f"""
                MATCH p = (s:Subreddit {{name: '{self._escape(name)}'}})-[:LINK_TO_AGG]->(a:Subreddit)-[:LINK_TO_AGG]->(b:Subreddit)-[:LINK_TO_AGG]->(s)
                WHERE all(r IN relationships(p) WHERE r.sentiment {op}) AND a <> b
                RETURN p LIMIT 500
            """, out_cols="p agtype")
        return _timed_per_input(run, subreddit_names)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        return False


def wait_age_ready(port: int, timeout: int = 60):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            conn = psycopg.connect(
                host="localhost", port=port, dbname="postgres", user="postgres", password="password",
                connect_timeout=3,
            )
            conn.close()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"AGE (postgres) not ready on port {port}") from last_err