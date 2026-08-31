import time
import csv
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from functools import cached_property
from pathlib import Path

EXPECTED_NODE_COUNT = 87_211
EXPECTED_EDGE_COUNT = 858_488
EXPECTED_EMBEDDED_NODE_COUNT = 51_269
EXPECTED_EDGE_AGG_COUNT = 339_643
EXPECTED_NODES_WITHOUT_EMBEDDED_DATASET = 67_180

FRIENDS_OF_FRIENDS_SENTIMENT = 0.5
MATCH_AGG_MAX = 5
TRAVERSAL_LIMIT = 500
GRAPH_QUERY_SUBREDDITS = [
    "shitamericanssay",
    "botsrights",
    "gaming",
    "shitpost",
    "conspiracy",
]
LINKS_CATEGORIES = ("positive", "negative")

@dataclass
class BenchMarkResult:
    cold_run: float | None
    hot_runs: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def avg(self) -> float:
        return float(np.mean(self.hot_runs)) if self.hot_runs else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.hot_runs)) if self.hot_runs else float("nan")

    @property
    def runs(self) -> list[float]:
        return ([self.cold_run] if self.cold_run is not None else []) + self.hot_runs


@dataclass
class MatchResult:
    """For match_pattern: k different pattern lengths, each measured n times.
    Keyed by pattern length -> BenchMarkResult for that length."""

    by_pattern_length: dict[int, BenchMarkResult]


def _timed_repeated(func, n: int = 5, cleanup: callable = None) -> BenchMarkResult:
    times, errors = [], []
    for _ in range(n):
        try:
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)
        except Exception as e:
            errors.append(str(e))
        finally:
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as e:
                    errors.append(f"cleanup failed: {e}")
    if not times:
        return BenchMarkResult(cold_run=None, hot_runs=[], errors=errors)
    cold_run, *hot_runs = times
    return BenchMarkResult(cold_run=cold_run, hot_runs=hot_runs, errors=errors)


def _timed_per_input(func, inputs: list) -> BenchMarkResult:
    times, errors = [], []
    for item in inputs:
        try:
            start = time.perf_counter()
            func(item)
            times.append(time.perf_counter() - start)
        except Exception as e:
            errors.append(f"input={item!r}: {e}")
    if not times:
        return BenchMarkResult(cold_run=None, hot_runs=[], errors=errors)
    cold_run, *hot_runs = times
    return BenchMarkResult(cold_run=cold_run, hot_runs=hot_runs, errors=errors)


def _timed_index_build(func, n: int = 5, cleanup: callable = None) -> BenchMarkResult:
    # identical shape to _timed_repeated — kept separate per your existing naming convention
    return _timed_repeated(func, n=n, cleanup=cleanup)


def _timed_match(func, inputs: list[str], max_range: int = MATCH_AGG_MAX) -> MatchResult:
    by_length = {}
    if max_range < 2:
        return MatchResult(by_pattern_length={})
    for k in range(1, max_range + 1):
        times, errors = [], []
        for item in inputs:
            try:
                start = time.perf_counter()
                func(item, k)
                times.append(time.perf_counter() - start)
            except Exception as e:
                errors.append(f"input={item!r}, length={k}: {e}")
        if times:
            cold_run, *hot_runs = times
        else:
            cold_run, hot_runs = None, []
        by_length[k] = BenchMarkResult(cold_run=cold_run, hot_runs=hot_runs, errors=errors)
    return MatchResult(by_pattern_length=by_length)

class BenchmarkImportError(Exception):
    """Raised when import_data fails to populate the database."""


class BaseBenchmarks(ABC):
    def _save(self, metric_name: str, results):
        output_dir = Path("results") / self.db_name
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{metric_name}_series.json"
        with open(path, "w") as f:
            json.dump(asdict(results), f, indent=2)

    @abstractmethod
    def import_data(self):
        print("Implement the graph import")

    def perform_benchmark(self):
        try:
            self.import_data()
        except BenchmarkImportError as e:
            print(f"[{self.db_name}] Import failed, skipping benchmarks: {e}")
            return
        except Exception as e:
            print(
                f"[{self.db_name}] Unexpected error during import, skipping benchmarks: {e}"
            )
            return

        # if isinstance(self, GraphBenchmarks):
        #     self.perform_graph_benchmarks()

        if isinstance(self, VectorBenchmarks):
            self.perform_vector_benchmarks()

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb): ...


class GraphBenchmarks(BaseBenchmarks):
    @abstractmethod
    def aggregate_graph(self):
        print("Implement the graph aggregation")

    @abstractmethod
    def common_neighbour_match(self, subreddit_names: list[str]):
        print("Implement the neighbour search")

    @abstractmethod
    def cycle_detection(self, subreddit_names: list[str], category: str):
        print("Implement the cycle detection")

    @abstractmethod
    def friends_of_friends(self, subreddit_name: list[str]):
        print("Implement max 5 hop traversal")

    @staticmethod
    def _sentiment_op(category: str) -> str:
        if category == "positive":
            return ">= 0"
        if category == "negative":
            return "< 0"
        raise ValueError(f"Unknown category: {category}")

    def perform_graph_benchmarks(self):
        subreddit_names = GRAPH_QUERY_SUBREDDITS

        print(f"[{self.db_name}] Performing aggregation benchmark.")
        self._save("aggregation", self.aggregate_graph())

        print(f"[{self.db_name}] Persisting aggregated edges.")
        self.persist_aggregation()

        print(f"[{self.db_name}] Performing common-neighbour match benchmark.")
        self._save(
            "common_neighbour_match", self.common_neighbour_match(subreddit_names)
        )

        for category in ("positive", "negative"):
            print(
                f"[{self.db_name}] Performing cycle detection benchmark ({category})."
            )
            self._save(
                f"cycle_{category}", self.cycle_detection(subreddit_names, category)
            )

        # print(f"[{self.db_name}] Performing friends of friends match benchmark.")
        # self._save("friends_of_friends", self.friends_of_friends(subreddit_names))

        print(f"[{self.db_name}] Graph benchmarks completed.")


class VectorBenchmarks(BaseBenchmarks):
    @cached_property
    def query_vectors_knn(self) -> dict[str, list[float]]:
        vectors = self._load_query_vectors("sample/input_vectors_1.csv")
        for name, vec in vectors.items():
            assert len(vec) == 300, f"{name}: expected 300 dims, got {len(vec)}"
            assert all(isinstance(x, float) for x in vec), (
                f"{name}: non-float element found"
            )
        return vectors

    @cached_property
    def query_vectors_ann_hnsw(self) -> dict[str, list[float]]:
        vectors = self._load_query_vectors("sample/input_vectors_2.csv")
        for name, vec in vectors.items():
            assert len(vec) == 300, f"{name}: expected 300 dims, got {len(vec)}"
            assert all(isinstance(x, float) for x in vec), (
                f"{name}: non-float element found"
            )
        return vectors

    @cached_property
    def query_vectors_ann_ivf(self) -> dict[str, list[float]]:
        vectors = self._load_query_vectors("sample/input_vectors_3.csv")
        for name, vec in vectors.items():
            assert len(vec) == 300, f"{name}: expected 300 dims, got {len(vec)}"
            assert all(isinstance(x, float) for x in vec), (
                f"{name}: non-float element found"
            )
        return vectors

    @staticmethod
    def _load_query_vectors(path: str) -> dict[str, list[float]]:
        vectors = {}
        with open(path) as f:
            reader = csv.reader(f)
            for name, *vec in reader:
                vectors[name] = [float(x) for x in vec]
        return vectors

    @abstractmethod
    def hnsw_index_build(self):
        print("Implement the hnsw build")

    @abstractmethod
    def ivf_index_build(self):
        print("Implement the ivf build")

    @abstractmethod
    def ann(self):
        print("Implement the ann search noth with IVF and HNSW indexes")

    @abstractmethod
    def knn(self):
        print("Implement the knn search")

    def perform_vector_benchmarks(self):
        for metric_name, metric, method, kwargs in [
            ("HNSW Index Build Time", "hnsw_index_build", self.hnsw_index_build, {}),
            ("IVF Index Build Time", "ivf_index_build", self.ivf_index_build, {}),
            (
                "ANN Search on HNSW Index",
                "ann_hnsw",
                self.ann,
                {"index_type": "hnsw", "query_vectors": self.query_vectors_ann_hnsw},
            ),
            (
                "ANN Search on IVF Index",
                "ann_ivf",
                self.ann,
                {"index_type": "ivf", "query_vectors": self.query_vectors_ann_ivf},
            ),
            ("KNN Search", "knn", self.knn, {"query_vectors": self.query_vectors_knn}),
        ]:
            print(f"[{self.db_name}] Benchmarking {metric_name}.")
            result = method(**kwargs)
            if result is not None:
                self._save(metric, result)

        print(f"[{self.db_name}] Vector benchamrks complete.")
