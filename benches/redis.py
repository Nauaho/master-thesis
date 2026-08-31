# benches/redis_bench.py
import time
import numpy as np
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from .base import (
    VectorBenchmarks,
    BenchmarkImportError,
    _timed_index_build,
    _timed_per_input,
    EXPECTED_EMBEDDED_NODE_COUNT,
)

VECTOR_DIM = 300
HNSW_INDEX = "subreddit_hnsw"
FLAT_INDEX = "subreddit_flat"
DOC_PREFIX = "subreddit:"


class RedisBenchmark(VectorBenchmarks):
    def __init__(self, port: int):
        self._r = redis.Redis(host="localhost", port=port, decode_responses=False)
        self.db_name = "redis"

    def import_data(self):
        try:
            import csv

            with open("data/web-redditEmbeddings-subreddits.csv") as f:
                reader = csv.reader(f)
                pipe = self._r.pipeline(transaction=False)
                count = 0
                for name, *raw_vec in reader:
                    vec = np.array([float(x) for x in raw_vec], dtype=np.float32)
                    if vec.shape[0] != VECTOR_DIM or not np.any(vec):
                        continue  # skip malformed / zero vectors — same filter as Neo4j
                    pipe.hset(
                        f"{DOC_PREFIX}{name}", mapping={"embedding": vec.tobytes()}
                    )
                    count += 1
                    if count % 1000 == 0:
                        pipe.execute()
                        pipe = self._r.pipeline(transaction=False)
                pipe.execute()
        except Exception as e:
            raise BenchmarkImportError(f"Redis import failed: {e}") from e

        actual = len(list(self._r.scan_iter(f"{DOC_PREFIX}*")))
        if actual != EXPECTED_EMBEDDED_NODE_COUNT:
            raise BenchmarkImportError(
                "Redis import validation failed:\n number of embeddings expected {EXPECTED_EMBEDDED_NODE_COUNT}, got {embedded_count}"
            )
        return actual

    def ivf_index_build(self):
        print(f"[{self.db_name}] IVF index build not supported, skipping.")
        return None

    def hnsw_index_build(self):
        def build():
            self._r.ft(HNSW_INDEX).create_index(
                [
                    VectorField(
                        "embedding",
                        "HNSW",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": VECTOR_DIM,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    )
                ],
                definition=IndexDefinition(
                    prefix=[DOC_PREFIX], index_type=IndexType.HASH
                ),
            )
            wait_for_hnsw_index(
                self._r,
                HNSW_INDEX,
                interval=0.005,
            )

        def drop():
            try:
                self._r.ft(HNSW_INDEX).dropindex(delete_documents=False)
            except redis.exceptions.ResponseError:
                pass

        return _timed_index_build(build, n=5, cleanup=drop)

    def _flat_index_build(self):
        """FLAT = exact/brute-force — required for knn, not separately benchmarked."""
        try:
            self._r.ft(FLAT_INDEX).dropindex(delete_documents=False)
        except redis.exceptions.ResponseError:
            pass
        self._r.ft(FLAT_INDEX).create_index(
            [
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                    },
                )
            ],
            definition=IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH),
        )

    def knn(self, query_vectors: dict, k: int = 10):
        self._flat_index_build()

        def run_query(vec):
            vec_bytes = np.array(vec, dtype=np.float32).tobytes()
            q = (
                Query(f"*=>[KNN {k} @embedding $vec AS score]")
                .sort_by("score")
                .dialect(2)
            )
            return self._r.ft(FLAT_INDEX).search(q, query_params={"vec": vec_bytes})

        try:
            return _timed_per_input(run_query, inputs=list(query_vectors.values()))
        finally:
            self._r.ft(FLAT_INDEX).dropindex(delete_documents=False)

    def ann(self, index_type: str, query_vectors: dict, k: int = 10):
        if index_type == "ivf":
            print(f"[{self.db_name}] IVF not supported, skipping ANN-IVF.")
            return None
        if index_type != "hnsw":
            raise ValueError(f"Unknown index_type: {index_type}")

        try:
            self._r.ft(HNSW_INDEX).dropindex(delete_documents=False)
        except redis.exceptions.ResponseError:
            pass
        self._r.ft(HNSW_INDEX).create_index(
            [
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                    },
                )
            ],
            definition=IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH),
        )

        wait_for_hnsw_index(
            self._r,
            HNSW_INDEX,
            interval=0.005,
        )

        def run_query(vec):
            vec_bytes = np.array(vec, dtype=np.float32).tobytes()
            q = (
                Query(f"*=>[KNN {k} @embedding $vec AS score]")
                .sort_by("score")
                .dialect(2)
            )
            return self._r.ft(HNSW_INDEX).search(q, query_params={"vec": vec_bytes})

        try:
            return _timed_per_input(run_query, inputs=list(query_vectors.values()))
        finally:
            self._r.ft(HNSW_INDEX).dropindex(delete_documents=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._r.close()
        return False


def wait_redis_ready(port: int, timeout: int = 60):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = redis.Redis(host="localhost", port=port)
            r.ping()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"redis not ready on port {port}") from last_err

def wait_for_hnsw_index(
    redis_client,
    index_name: str,
    interval: float = 0.010,
    timeout: float = 600.0,
):
    """
    Wait until a Redis/RediSearch HNSW index is fully constructed.

    Polls FT.INFO every `interval` seconds.

    Returns:
        elapsed time in seconds from the first FT.INFO check
        until the index reports 100% indexed and is no longer indexing.

    Raises:
        TimeoutError: if the index does not become ready within `timeout`.
    """
    start = time.perf_counter()
    deadline = start + timeout

    while True:
        info = redis_client.ft(index_name).info()

        indexing = int(info["indexing"])
        percent_indexed = float(info["percent_indexed"])

        if indexing == 0 and percent_indexed >= 1.0:
            return time.perf_counter() - start

        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"Redis HNSW index '{index_name}' did not finish "
                f"within {timeout:.1f}s "
                f"(indexing={indexing}, "
                f"percent_indexed={percent_indexed:.4f})"
            )

        time.sleep(interval)
