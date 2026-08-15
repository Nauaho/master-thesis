# benches/chroma_bench.py
import time
import csv
import chromadb
from base import (
    _timed_per_input,
    VectorBenchmarks,
    BenchmarkImportError,
    EXPECTED_EMBEDDED_NODE_COUNT
)

VECTOR_DIM = 300
COLLECTION_NAME = "subreddits"


class ChromaBenchmark(VectorBenchmarks):
    def __init__(self, port: int):
        self._client = chromadb.HttpClient(host="localhost", port=port)
        self.db_name = "chroma"
        self._collection = None

    def import_data(self):
        try:
            self._collection = self._client.create_collection(
                name=COLLECTION_NAME,
                configuration={
                    "hnsw": {
                        "space": "cosine",
                        "batch_size": 100,       # frequent migration into real HNSW graph
                        "sync_threshold": 100,    # frequent disk flush, same as batch_size
                    }
                },
            )
            ids, embeddings, first_id = [], [], None
            with open("data/web-redditEmbeddings-subreddits.csv") as f:
                reader = csv.reader(f)
                for name, *raw_vec in reader:
                    vec = [float(x) for x in raw_vec]
                    if len(vec) != VECTOR_DIM or not any(vec):
                        continue
                    if first_id is None:
                        first_id = name
                    ids.append(name)
                    embeddings.append(vec)
                    if len(ids) >= 1000:
                        self._collection.add(ids=ids, embeddings=embeddings)
                        ids, embeddings = [], []
                if ids:
                    self._collection.add(ids=ids, embeddings=embeddings)

            self._wait_fully_indexed(first_id)
        except Exception as e:
            raise BenchmarkImportError(f"Chroma import failed: {e}") from e

        count = self._collection.count()
        if count != EXPECTED_EMBEDDED_NODE_COUNT:
            raise BenchmarkImportError(
                f"Chroma import count mismatch: expected {EXPECTED_EMBEDDED_NODE_COUNT}, got {count}"
            )
        return count

    def hnsw_index_build(self):
        print(f"[{self.db_name}] HNSW build not independently controllable "
              f"(background compaction, not a client-triggered op) — skipping.")
        return None

    def ivf_index_build(self):
        print(f"[{self.db_name}] IVF index build not supported, skipping.")
        return None

    def knn(self, query_vectors: dict, k: int = 10):
        print(f"[{self.db_name}] No exact/brute-force search mode available "
              f"(HNSW-only) — skipping KNN.")
        return None

    def ann(self, index_type: str, query_vectors: dict, k: int = 10):
        if index_type != "hnsw":
            print(f"[{self.db_name}] Only HNSW is supported (no IVF) — skipping {index_type}.")
            return None

        def run(vec: list[float]):
            self._collection.query(query_embeddings=[vec], n_results=k)

        return _timed_per_input(run, query_vectors.items())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # HttpClient has no persistent connection/session to close

    def _wait_fully_indexed(self, probe_id: str, timeout: int = 120):
        """Belt-and-suspenders: config alone doesn't guarantee the final partial
        batch flushed, so poll until the first-inserted vector is queryable."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self._collection.get(ids=[probe_id], include=["embeddings"])
            if result["ids"]:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Chroma did not finish indexing within {timeout}s")

def wait_chroma_ready(port: int, timeout: int = 60):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            client = chromadb.HttpClient(host="localhost", port=port)
            client.heartbeat()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"chroma not ready on port {port}") from last_err