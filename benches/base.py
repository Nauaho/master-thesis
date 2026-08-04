import time
import numpy as np
from abc import ABC, abstractmethod

EXPECTED_NODE_COUNT = 87_220
EXPECTED_EDGE_COUNT = 858_488


class BenchmarkImportError(Exception):
    """Raised when import_data fails to populate the database."""

def _timed_runs(func, inputs: list | None, cleanup: callable | None, n: int = 5):
    """
    Core timing loop, called explicitly by benchmark methods.
    - inputs: if given, times func(item) once per item (varying-input case).
    - n: if inputs is None, times func() n times identically (repeated case).
    - cleanup: optional callable() run after each timed rep.
    Returns (mean, std) of elapsed times.
    """
    time_list = []
    reps = inputs if inputs is not None else range(n)
    for item in reps:
        start = time.time()
        func(item) if inputs is not None else func()
        time_list.append(time.time() - start)
        if cleanup is not None:
            cleanup()
    return (np.mean(time_list), np.std(time_list))


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