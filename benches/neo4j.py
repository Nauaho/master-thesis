from neo4j import GraphDatabase

from base import GraphBenchmarks, VectorBenchmarks, BenchmarkImportError, EXPECTED_NODE_COUNT, EXPECTED_EDGE_COUNT
import time

class Neo4jBenchamrk(GraphBenchmarks, VectorBenchmarks):

    def __init__(self, port: int):
        self._driver = GraphDatabase.driver(f"bolt://localhost:{port}", auth=("neo4j", "password"))
        self.db_name = "neo4j"

    def _exec(self, query, **params):
        return self._driver.execute_query(query, **params)
        
    def import_data(self):
        try:
            self._exec("""
                LOAD CSV FROM 'file:///web-redditEmbeddings-subreddits.csv' AS row
                CALL {
                WITH row
                MERGE (s:Subreddit {name: row[0]})
                SET s.embedding = toFloatList(row[1..])
                } IN TRANSACTIONS OF 1000 ROWS
            """)
        except Exception as e:
            raise BenchmarkImportError(f"Neo4j CSV load failed: {e}") from e

        node_records, _, _ = self._exec("MATCH (n:Subreddit) RETURN count(n) AS node_count")
        edge_records, _, _ = self._exec("MATCH ()-[r:LINK_TO]->() RETURN count(r) AS edge_count")
        node_count = node_records["node_count"]
        edge_count = edge_records["edge_count"]

        if node_count != EXPECTED_NODE_COUNT or edge_count != EXPECTED_EDGE_COUNT:
            raise BenchmarkImportError(f"Neo4j CSV load failed: some data was lost.\n Imported nodes: {node_count}\n Imported edges: {edge_count}")
        else:
            return node_count, edge_count

    def ivf_index_build(self):
        print(f"[{self.db_name}] IVF index build not supported, skipping.")
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._driver.close()
        return False

def wait_neo4j_ready(port: int, timeout: int = 60):
    from neo4j import GraphDatabase
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            driver = GraphDatabase.driver(f"bolt://localhost:{port}", auth=("neo4j", "password"))
            driver.verify_connectivity()
            driver.close()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"neo4j not ready on port {port}") from last_err