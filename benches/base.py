import time
import numpy as np
from abc import ABC, abstractmethod

EXPECTED_NODE_COUNT = 87_220
EXPECTED_EDGE_COUNT = 858_488

class BenchmarkImportError(Exception):
    """Raised when import_data fails to populate the database."""
    pass

def timer(func):
    """Custom helper decorator to make possible a measurement of 5 time measurements: average and standard deviation"""
    def my_inner(*args, **kwargs):
        time_list = []
        for _ in range(5):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            time_list.append(end - start)
        return (np.mean(time_list) ,np.stdev(time_list))
    return my_inner

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
            print(f"[{self.db_name}] Unexpected error during import, skipping benchmarks: {e}")
            return
        if isinstance(self, VectorBenchmarks):
            self.perform_vector_benchmarks()
        if isinstance(self, GraphBenchmarks):
            self.perform_graph_benchmarks()

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

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
        print("Implement the ann search")

    @abstractmethod
    def knn(self):
        print("Implement the knn search")

    def perform_vector_benchmarks(self):
        for metric, method in [
            ("hnsw_index_build", self.hnsw_index_build),
            ("ivf_index_build", self.ivf_index_build),
            ("ann", self.ann),
            ("knn", self.knn),
        ]:
            result = method()
            if result is not None:
                self._save(metric, result)