# benches/postgres_bench.py
import time
import csv
import psycopg
import math
from pgvector.psycopg import register_vector
from pgvector.vector import Vector
from .base import (
    VectorBenchmarks,
    BenchmarkImportError,
    _timed_per_input,
    _timed_index_build,
    EXPECTED_EMBEDDED_NODE_COUNT,
)

VECTOR_DIM = 300
HNSW_M = 16
HNSW_EF_CONSTRCUTION = 64
HNSW_INDEX = "idx_hnsw"
IVF_INDEX = "idx_ivfflat"

class PgVectorBenchmark(VectorBenchmarks):
    def __init__(self, port: int):
        self._conn = psycopg.connect(
            host="localhost", port=port, dbname="postgres", user="postgres", password="password",
            autocommit=True,
        )
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self.db_name = "pgvector"

    def import_data(self):
        try:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS subreddits (
                    name TEXT PRIMARY KEY,
                    embedding VECTOR(300)
                )
            """)

            with self._conn.cursor() as cur:
                with cur.copy("COPY subreddits (name, embedding) FROM STDIN (FORMAT BINARY)") as copy:
                    with open("data/web-redditEmbeddings-subreddits.csv") as f:
                        reader = csv.reader(f)
                        for name, *raw_vec in reader:
                            vec = [float(x) for x in raw_vec]
                            if len(vec) != VECTOR_DIM or not any(vec):
                                continue
                            copy.write_row((name, Vector(vec)))
        except Exception as e:
            raise BenchmarkImportError(f"pgvector import failed: {e}") from e

        count = self._conn.execute("SELECT count(*) FROM subreddits WHERE embedding IS NOT NULL").fetchone()[0]
        if count != EXPECTED_EMBEDDED_NODE_COUNT:
            raise BenchmarkImportError(
                f"pgvector import count mismatch: expected {EXPECTED_EMBEDDED_NODE_COUNT}, got {count}"
            )
        
        self.optimal_ivf_lists_number = 0
        if count < 1_000_000:
            self.optimal_ivf_lists_number = math.floor(count / 1000)
        else:
            self.optimal_ivf_lists_number = math.floor(math.sqrt(count))

        self.optimal_ivf_probes_number = math.floor(math.sqrt(self.optimal_ivf_lists_number))

        self._conn.execute(f"SET ivfflat.probes = {self.optimal_ivf_probes_number};")
        return count

    def _drop_all_indexes(self):
        self._conn.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX}")
        self._conn.execute(f"DROP INDEX IF EXISTS {IVF_INDEX}")

    def hnsw_index_build(self):
        def build():
            self._conn.execute(
                f"CREATE INDEX {HNSW_INDEX} ON subreddits USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRCUTION})"
            )
        def drop():
            self._conn.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX}")
        return _timed_index_build(build, n=5, cleanup=drop)

    def ivf_index_build(self):
        def build():
            self._conn.execute(
                f"CREATE INDEX {IVF_INDEX} ON subreddits USING ivfflat (embedding vector_cosine_ops) "
                f"WITH (lists = {self.optimal_ivf_lists_number})"
            )
            self._conn.execute("ANALYZE subreddits")
        def drop():
            self._conn.execute(f"DROP INDEX IF EXISTS {IVF_INDEX}")
        return _timed_index_build(build, n=5, cleanup=drop)

    def knn(self, query_vectors: dict, k: int = 10):
        self._drop_all_indexes()  # guarantee exact search — no ANN index present at all
        def run_query(vec):
            return self._conn.execute(
                "SELECT name, embedding <=> %s AS distance FROM subreddits "
                "ORDER BY distance LIMIT %s", (Vector(vec), k)
            ).fetchall()

        return _timed_per_input(run_query, list(query_vectors.values()))

    def ann(self, index_type: str, query_vectors: dict, k: int = 10):
        self._drop_all_indexes()
        if index_type == "hnsw":
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX} ON subreddits USING hnsw "
                f"(embedding vector_cosine_ops) WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRCUTION})"
            )
            index_name = HNSW_INDEX
        elif index_type == "ivf":
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {IVF_INDEX} ON subreddits USING ivfflat "
                f"(embedding vector_cosine_ops) WITH (lists = {self.optimal_ivf_lists_number})"
            )
            self._conn.execute("ANALYZE subreddits")
            index_name = IVF_INDEX
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        def run_query(vec):
            return self._conn.execute(
                "SELECT name, embedding <=> %s AS distance FROM subreddits "
                "ORDER BY distance LIMIT %s", (Vector(vec), k)
            ).fetchall()

        try:
            return _timed_per_input(run_query, list(query_vectors.values()))
        finally:
            self._conn.execute(f"DROP INDEX IF EXISTS {index_name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        return False


def wait_pgvector_ready(port: int, timeout: int = 60):
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
    raise TimeoutError(f"pgvector not ready on port {port}") from last_err