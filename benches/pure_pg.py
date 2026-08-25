import time
import csv
import psycopg
from .base import (
    GraphBenchmarks,
    BenchmarkImportError,
    _timed_repeated,
    _timed_per_input,
    EXPECTED_NODES_WITHOUT_EMBEDDED_DATASET,
    EXPECTED_EDGE_COUNT,
    EXPECTED_EDGE_AGG_COUNT,
)

class PostgresGraphBenchmark(GraphBenchmarks):
    def __init__(self, port: int):
        self._conn = psycopg.connect(
            host="localhost", port=port, dbname="postgres", user="postgres", password="password",
            autocommit=True,
        )
        self.db_name = "postgres_sql"

    def import_data(self):
        try:
            # Create subreddit (node) table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS subreddit (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                )
            """)

            # Create normalized link_to table with foreign keys
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS link_to (
                    id SERIAL PRIMARY KEY,
                    source_id INTEGER NOT NULL REFERENCES subreddit(id),
                    target_id INTEGER NOT NULL REFERENCES subreddit(id),
                    post_id TEXT,
                    ts TIMESTAMP,
                    sentiment_score REAL,
                    properties REAL[]
                )
            """)

            # First pass: collect all unique subreddits
            all_subreddits = set()
            for filename in ["soc-redditHyperlinks-body.tsv", "soc-redditHyperlinks-title.tsv"]:
                with open(f"data/{filename}") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        all_subreddits.add(row["SOURCE_SUBREDDIT"])
                        all_subreddits.add(row["TARGET_SUBREDDIT"])

            # Insert unique subreddits
            with self._conn.cursor() as cur:
                for subreddit_name in sorted(all_subreddits):
                    cur.execute(
                        "INSERT INTO subreddit (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                        (subreddit_name,)
                    )
            
            # Fetch the subreddit ID mapping
            subreddit_id_map = {}
            with self._conn.cursor() as cur:
                cur.execute("SELECT id, name FROM subreddit")
                for row in cur.fetchall():
                    subreddit_id_map[row[1]] = row[0]

            # Second pass: load edges using subreddit IDs
            for filename in ["soc-redditHyperlinks-body.tsv", "soc-redditHyperlinks-title.tsv"]:
                with open(f"data/{filename}") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    with self._conn.cursor() as cur:
                        with cur.copy(
                            "COPY link_to (source_id, target_id, post_id, ts, sentiment_score, properties) FROM STDIN"
                        ) as copy:
                            for row in reader:
                                source_id = subreddit_id_map[row["SOURCE_SUBREDDIT"]]
                                target_id = subreddit_id_map[row["TARGET_SUBREDDIT"]]
                                properties = [float(num) for num in row["PROPERTIES"].split(",")]
                                copy.write_row((
                                    source_id,
                                    target_id,
                                    row["POST_ID"],
                                    row["TIMESTAMP"],
                                    float(row["LINK_SENTIMENT"]),
                                    properties
                                ))

            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_link_source ON link_to (source_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_link_target ON link_to (target_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_link_group ON link_to (source_id, target_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_subreddit_lookup ON subreddit (id)")
            self._conn.execute("ANALYZE link_to;")
            self._conn.execute("ANALYZE subreddit;")
        except Exception as e:
            raise BenchmarkImportError(f"Postgres (SQL) import failed: {e}") from e

        node_count = self._conn.execute("""
            SELECT count(*) FROM subreddit;
        """).fetchone()[0]
        edge_count = self._conn.execute("SELECT count(*) FROM link_to").fetchone()[0]

        errors = []
        if node_count != EXPECTED_NODES_WITHOUT_EMBEDDED_DATASET:
            errors.append(f"node count: expected {EXPECTED_NODES_WITHOUT_EMBEDDED_DATASET}, got {node_count}")
        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(f"edge count: expected {EXPECTED_EDGE_COUNT}, got {edge_count}")
        if errors:
            raise BenchmarkImportError("Postgres (SQL) import validation failed:\n" + "\n".join(errors))
        return node_count, edge_count

    def persist_aggregation(self):
        self._conn.execute("DROP TABLE IF EXISTS link_to_agg")
        self._conn.execute("""
            CREATE TABLE link_to_agg AS
            SELECT source_id, target_id,
                   avg(sentiment_score) AS sentiment,
                   count(*) AS link_count
            FROM link_to
            GROUP BY source_id, target_id
        """)
        self._conn.execute("CREATE INDEX ON link_to_agg (source_id)")
        self._conn.execute("CREATE INDEX ON link_to_agg (target_id)")

        edge_agg_count = self._conn.execute("SELECT count(*) FROM link_to_agg").fetchone()[0]
        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:
            raise BenchmarkImportError(
                f"Postgres (SQL) aggregation validation failed: expected {EXPECTED_EDGE_AGG_COUNT}, got {edge_agg_count}"
            )

    def aggregate_graph(self):
        def run():
            return self._conn.execute("""
                SELECT source_id, target_id, sum(sentiment_score), count(*)
                FROM link_to
                GROUP BY source_id, target_id
                ORDER BY sum(sentiment_score) DESC
            """).fetchall()
        return _timed_repeated(run, n=5)

    def common_neighbour_match(self, subreddit_names: list[str]):
        # First, get the IDs for the provided subreddit names
        def run(name: str):
            # Get source subreddit ID
            source_id_row = self._conn.execute(
                "SELECT id FROM subreddit WHERE name = %s",
                (name,)
            ).fetchone()
            if not source_id_row:
                return []
            source_id = source_id_row[0]
            
            return self._conn.execute("""
                SELECT s2.name AS new_friend, (r2.sentiment - r1.sentiment) AS delta_interest
                FROM link_to_agg r1
                JOIN link_to_agg r2 ON r1.target_id = r2.target_id
                JOIN subreddit s2 ON r2.source_id = s2.id
                WHERE r1.source_id = %s
                  AND r1.sentiment > 0.33 AND r2.sentiment > 0.33
                  AND r2.source_id <> r1.source_id
                  AND NOT EXISTS (
                      SELECT 1 FROM link_to_agg x
                      WHERE x.source_id = r1.source_id AND x.target_id = r2.source_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM link_to_agg x
                      WHERE x.source_id = r2.source_id AND x.target_id = r1.source_id
                  )
                ORDER BY delta_interest DESC
                LIMIT 100
            """, (source_id,)).fetchall()
        return _timed_per_input(run, subreddit_names)

    def cycle_detection(self, subreddit_names: list[str], category: str = "positive"):
        op = self._sentiment_op(category)  # returns e.g. ">= 0" / "< 0"
        def run(name: str):
            # Get source subreddit ID
            source_id_row = self._conn.execute(
                "SELECT id FROM subreddit WHERE name = %s",
                (name,)
            ).fetchone()
            if not source_id_row:
                return []
            source_id = source_id_row[0]
            
            return self._conn.execute(f"""
                SELECT s1.name, s2.name AS a, s3.name AS b
                FROM link_to_agg e1
                JOIN link_to_agg e2 ON e1.target_id = e2.source_id
                JOIN link_to_agg e3 ON e2.target_id = e3.source_id
                                    AND e3.target_id = e1.source_id
                JOIN subreddit s1 ON e1.source_id = s1.id
                JOIN subreddit s2 ON e1.target_id = s2.id
                JOIN subreddit s3 ON e2.target_id = s3.id
                WHERE e1.source_id = %s
                  AND e1.target_id <> e2.target_id
                  AND e1.sentiment {op} AND e2.sentiment {op} AND e3.sentiment {op}
                LIMIT 500
            """, (source_id,)).fetchall()
        return _timed_per_input(run, subreddit_names)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        return False


def wait_postgres_sql_ready(port: int, timeout: int = 60):
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
    raise TimeoutError(f"postgres (sql) not ready on port {port}") from last_err