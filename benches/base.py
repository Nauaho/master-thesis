import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

EXPECTED_NODE_COUNT = 87_220
EXPECTED_EDGE_COUNT = 858_488
EXPECTED_EMBEDDED_NODE_COUNT = 51_278

@dataclass
class RepeatedOpResult:
    """For ops with no cold/hot distinction — every rep is independently fresh
    (e.g. index build+drop cycles)."""
    runs: list[float]

    @property
    def avg(self) -> float:
        return float(np.mean(self.runs))

    @property
    def std(self) -> float:
        return float(np.std(self.runs))


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


@dataclass
class MatchResult:
    """For match_pattern: k different pattern lengths, each measured n times.
    Keyed by pattern length -> BenchMarkResult for that length."""
    by_pattern_length: dict[int, BenchMarkResult]


def _timed_index_build(func, n: int = 5, cleanup: callable = None) -> RepeatedOpResult:
    times = []
    for _ in range(n):
        start = time.time()
        func()
        times.append(time.time() - start)
        if cleanup is not None:
            cleanup()
    return RepeatedOpResult(runs=times)


def _timed_runs(func, n: int = 5, inputs: list = None, cleanup: callable = None) -> BenchMarkResult:
    """n identical reps, OR one rep per item in `inputs` if given."""
    reps = inputs if inputs is not None else [None] * n
    times = []
    for item in reps:
        start = time.time()
        func(item) if inputs is not None else func()
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
        np.save(f"{self.db_name}_{metric_name}_series.npy", results)

    def _run_and_save(self, metric_name: str, method, *args, **kwargs):
        result = method(*args, **kwargs)
        self._save(metric_name, result)
        return result

    @abstractmethod
    def import_data(self):
        print("Implement the graph import")

    def perform_benchamrk(self):
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
    def match_pattern(self, n: int):
        print("Implement the matching pattern")

    @abstractmethod
    def cycle_detection(self):
        print("Implement the cycle detection")

    def perform_graph_benchmarks(self):
        self._run_and_save("aggregation", self.aggregate_graph)
        match_results = [self.match_pattern(n) for n in range(50)]
        self._save("match", match_results)
        self._run_and_save("cycle", self.cycle_detection)


class VectorBenchmarks(BaseBenchmarks):
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
        for metric, method, kwargs in [
            ("hnsw_index_build", self.hnsw_index_build, {}),
            ("ivf_index_build", self.ivf_index_build, {}),
            ("ann_hnsw", self.ann, {"index_type": "hnsw"}),
            ("ann_ivf", self.ann, {"index_type": "ivf"}),
            ("knn", self.knn, {}),
        ]:
            result = method(**kwargs)
            if result is not None:
                self._save(metric, result)