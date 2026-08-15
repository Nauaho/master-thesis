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

@dataclass
class BenchMarkResult:
    """For query-style ops where cache state matters — first rep is cold,
    remaining reps are hot (cache-warmed)."""
    cold_run: float
    hot_runs: list[float] = field(default_factory=list)

    @property
    def avg(self) -> float:
        return float(np.mean(self.hot_runs))

    @property
    def std(self) -> float:
        return float(np.std(self.hot_runs))

    @property
    def runs(self) -> list[float]:
        return [self.cold_run, *self.hot_runs]


@dataclass
class MatchResult:
    """For match_pattern: k different pattern lengths, each measured n times.
    Keyed by pattern length -> BenchMarkResult for that length."""
    by_pattern_length: dict[int, BenchMarkResult]

def _timed_repeated(func, n: int = 5, cleanup: callable = None) -> BenchMarkResult:
    """Same query run n times identically — cold_run captures cache-miss cost,
    hot_runs capture cache-warmed performance."""
    times = []
    for _ in range(n):
        start = time.time()
        func()
        times.append(time.time() - start)
        if cleanup is not None:
            cleanup()
    cold_run, *hot_runs = times
    return BenchMarkResult(cold_run=cold_run, hot_runs=hot_runs)

def _timed_per_input(func, inputs: list) -> BenchMarkResult:
    """One rep per distinct input — no cold/hot framing, since each rep is a
    genuinely different query/operation, not a repeat of the same one."""
    times = []
    for item in inputs:
        start = time.time()
        func(item)
        times.append(time.time() - start)
    cold_run, *hot_runs = times
    return BenchMarkResult(cold_run=cold_run, hot_runs=hot_runs)

def _timed_index_build(func, n: int = 5, cleanup: callable = None) -> BenchMarkResult:
    times = []
    for _ in range(n):
        start = time.time()
        func()
        times.append(time.time() - start)
        if cleanup is not None:
            cleanup()
    cold_run, *hot_runs = times
    return BenchMarkResult(cold_run=cold_run, hot_runs=hot_runs)

def _timed_match(func, pattern_lengths: range, n: int = 5) -> MatchResult:
    """func(pattern_length) run n times per pattern length (n x k total calls)."""
    by_length = {}
    for k in pattern_lengths:
        times = []
        for _ in range(n):
            start = time.time()
            func(k)
            times.append(time.time() - start)
        cold_run, *hot_runs = times
        by_length[k] = BenchMarkResult(cold_run=cold_run, hot_runs=hot_runs)
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
        if isinstance(self, VectorBenchmarks):
            self.perform_vector_benchmarks()
        if isinstance(self, GraphBenchmarks):
            self.perform_graph_benchmarks()

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb): ...


class GraphBenchmarks(BaseBenchmarks):
    @abstractmethod
    def aggregate_graph(self):
        print("Implement the graph aggregation")

    @abstractmethod
    def match_pattern(self):
        print("Implement the matching pattern")

    @abstractmethod
    def cycle_detection(self):
        print("Implement the cycle detection")

    def perform_graph_benchmarks(self):
        print(f"[{self.db_name}] Performing aggregation benchamrk.")
        self._save("aggregation", self.aggregate_graph())
        print(f"[{self.db_name}] Performing matching benchamrk.")
        self._save("match", self.match_pattern())
        print(f"[{self.db_name}] Performing cycle detection benchamrk.")
        self._save("cycle", self.cycle_detection())
        print(f"[{self.db_name}] Graph benchamrks completed.")


class VectorBenchmarks(BaseBenchmarks):

    @cached_property
    def query_vectors_knn(self) -> dict[str, list[float]]:
        vectors = self._load_query_vectors("sample/input_vectors_1.csv")
        for name, vec in vectors.items():
            assert len(vec) == 300, f"{name}: expected 300 dims, got {len(vec)}"
            assert all(isinstance(x, float) for x in vec), f"{name}: non-float element found"
        return vectors
    
    @cached_property
    def query_vectors_ann_hnsw(self) -> dict[str, list[float]]:
        vectors = self._load_query_vectors("sample/input_vectors_2.csv")
        for name, vec in vectors.items():
            assert len(vec) == 300, f"{name}: expected 300 dims, got {len(vec)}"
            assert all(isinstance(x, float) for x in vec), f"{name}: non-float element found"
        return vectors

    @cached_property
    def query_vectors_ann_ivf(self) -> dict[str, list[float]]:
        vectors = self._load_query_vectors("sample/input_vectors_3.csv")
        for name, vec in vectors.items():
            assert len(vec) == 300, f"{name}: expected 300 dims, got {len(vec)}"
            assert all(isinstance(x, float) for x in vec), f"{name}: non-float element found"
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
            ("ANN Search on HNSW Index", "ann_hnsw", self.ann, {"index_type": "hnsw", "query_vectors": self.query_vectors_ann_hnsw}),
            ("ANN Search on IVF Index", "ann_ivf", self.ann, {"index_type": "ivf", "query_vectors": self.query_vectors_ann_ivf}),
            ("KNN Search", "knn", self.knn, {"query_vectors": self.query_vectors_knn}),
        ]:
            print(f"[{self.db_name}] Benchmarking {metric_name}.")
            result = method(**kwargs)
            if result is not None:
                self._save(metric, result)

        print(f"[{self.db_name}] Vector benchamrks complete.")