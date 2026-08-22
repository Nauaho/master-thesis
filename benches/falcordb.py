# benches/falkordb_bench.py
import time
import csv
from datetime import datetime
from falkordb import FalkorDB
from .base import (
    GraphBenchmarks,
    VectorBenchmarks,
    BenchmarkImportError,
    _timed_repeated,
    _timed_per_input,
    _timed_index_build,
    EXPECTED_NODE_COUNT,
    EXPECTED_EDGE_COUNT,
    EXPECTED_EMBEDDED_NODE_COUNT,
    EXPECTED_EDGE_AGG_COUNT,
)

VECTOR_DIM = 300
INDEX_NAME = "subreddit_embeddings"
GRAPH_NAME = "subreddits"
BATCH_SIZE = 5000


class FalkorDBBenchmark(GraphBenchmarks, VectorBenchmarks):
    def __init__(self, port: int):
        self._client = FalkorDB(host="localhost", port=port)
        self._graph = self._client.select_graph(GRAPH_NAME)
        self.db_name = "falkordb"

    def _exec(self, query, **params):
        return self._graph.query(query, params=params)

    def import_data(self):
        try:
            self._graph.create_node_unique_constraint("Subreddit", "name")

            with open("data/web-redditEmbeddings-subreddits.csv") as f:
                batch = []
                reader = csv.reader(f)

                for name, *raw_vec in reader:
                    vec = [float(x) for x in raw_vec]
                    if len(vec) != VECTOR_DIM or not any(vec):
                        continue
                    batch.append({"name": name, "vec": vec})
                    
                    if len(batch) >= BATCH_SIZE:
                        self._exec(
                            """
                            UNWIND $rows AS row
                            MERGE (s:Subreddit {name: row.name})
                            SET s.raw_embedding = row.vec
                            """,
                            rows=batch,
                        )
                        batch = []

                if batch:
                    self._exec(
                        """
                        UNWIND $rows AS row
                        MERGE (s:Subreddit {name: row.name})
                        SET s.raw_embedding = row.vec
                        """,
                        rows=batch,
                    )
                    batch = []

            for filename in [
                "data/soc-redditHyperlinks-body.tsv",
                "data/soc-redditHyperlinks-title.tsv",
            ]:
                with open(filename) as f:
                    batch = []
                    reader = csv.DictReader(f, delimiter="\t")

                    for row in reader:
                        try:
                            date_string = row["TIMESTAMP"]
                            converted_timestamp = int(
                                datetime.strptime(
                                    date_string, "%Y-%m-%d %H:%M:%S"
                                ).timestamp()
                            )

                            row_data = {
                                "origin": row["SOURCE_SUBREDDIT"],
                                "target": row["TARGET_SUBREDDIT"],
                                "post_id": row["POST_ID"],
                                "sentiment": float(row["LINK_SENTIMENT"]),
                                "timestamp": converted_timestamp,
                            }
                            batch.append(row_data)

                        except (ValueError, KeyError):
                            continue

                        if len(batch) >= BATCH_SIZE:
                            self._exec(
                                """
                                UNWIND $rows AS row
                                MERGE (source:Subreddit {name: row.origin})
                                MERGE (target:Subreddit {name: row.target})
                                CREATE (source)-[r:LINK_TO {
                                    postId: row.post_id,
                                    timestamp: row.timestamp,
                                    sentimentScore: row.sentiment
                                }]->(target)
                                """,
                                rows=batch,
                            )
                            batch = []
                if batch:
                    self._exec(
                        """
                        UNWIND $rows AS row
                        MERGE (source:Subreddit {name: row.origin})
                        MERGE (target:Subreddit {name: row.target})
                        CREATE (source)-[r:LINK_TO {
                            postId: row.post_id,
                            timestamp: row.timestamp,
                            sentimentScore: row.sentiment
                        }]->(target)
                        """,
                        rows=batch,
                    )
                    batch = []

            self._exec(
                """
                MATCH (s:Subreddit) 
                WHERE s.raw_embedding IS NOT NULL
                SET s.embedding = vecf32(s.raw_embedding)
                REMOVE s.raw_embedding
                """
            )

        except Exception as e:
            raise BenchmarkImportError(f"FalkorDB import failed: {e}") from e

        # --- VALIDATION BLOCK ---
        node_count = self._exec("MATCH (n:Subreddit) RETURN count(n) AS c").result_set[0][0]
        embedded_count = self._exec(
            "MATCH (n:Subreddit) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
        ).result_set[0][0]
        edge_count = self._exec("MATCH ()-[r:LINK_TO]->() RETURN count(r) AS c").result_set[0][0]

        errors = []
        if node_count != EXPECTED_NODE_COUNT:
            errors.append(f"node count: expected {EXPECTED_NODE_COUNT}, got {node_count}")
        if embedded_count != EXPECTED_EMBEDDED_NODE_COUNT:
            errors.append(f"embedded node count: expected {EXPECTED_EMBEDDED_NODE_COUNT}, got {embedded_count}")
        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(f"edge count: expected {EXPECTED_EDGE_COUNT}, got {edge_count}")
        if errors:
            raise BenchmarkImportError("FalkorDB import validation failed:\n" + "\n".join(errors))
            
        return node_count, embedded_count, edge_count


    def ivf_index_build(self):
        print(f"[{self.db_name}] IVF index build not supported, skipping.")
        return None

    def hnsw_index_build(self):
        def build():
            self._exec(f"""
                CREATE VECTOR INDEX FOR (n:Subreddit) ON (n.embedding)
                OPTIONS {{dimension: {VECTOR_DIM}, similarityFunction: 'cosine'}}
            """)

        def drop():
            self._exec("DROP VECTOR INDEX FOR (n:Subreddit) ON (n.embedding)")

        return _timed_index_build(build, n=5, cleanup=drop)

    def knn(self, query_vectors: dict, k: int = 10):
        def run_query(vec):
            return self._exec(
                """
                MATCH (s:Subreddit) WHERE s.embedding IS NOT NULL
                WITH s, vector.similarity.cosine(s.embedding, vecf32($queryVector)) AS score
                RETURN s.name, score ORDER BY score DESC LIMIT $k
            """,
                queryVector=vec,
                k=k,
            )

        return _timed_per_input(run_query, inputs=list(query_vectors.values()))

    def ann(self, index_type: str, query_vectors: dict, k: int = 10):
        if index_type == "ivf":
            print(f"[{self.db_name}] IVF not supported, skipping ANN-IVF.")
            return None
        if index_type != "hnsw":
            raise ValueError(f"Unknown index_type: {index_type}")

        self._exec(f"""
            CREATE VECTOR INDEX FOR (n:Subreddit) ON (n.embedding)
            OPTIONS {{dimension: {VECTOR_DIM}, similarityFunction: 'cosine'}}
        """)

        def run_query(vec):
            return self._exec(
                """
                CALL db.idx.vector.queryNodes('Subreddit', 'embedding', $k, vecf32($queryVector))
                YIELD node, score
                RETURN node.name, score
            """,
                queryVector=vec,
                k=k,
            )

        try:
            return _timed_per_input(run_query, inputs=list(query_vectors.values()))
        finally:
            self._exec("DROP VECTOR INDEX FOR (n:Subreddit) ON (n.embedding)")

    def persist_aggregation(self):
        self._exec("""
            MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)
            WITH s, t, avg(r.sentimentScore) AS sentiment, count(r) AS linkCount
            MERGE (s)-[agg:LINK_TO_AGG]->(t)
            SET agg.sentiment = sentiment, agg.linkCount = linkCount
        """)

        self._exec("CREATE INDEX FOR ()-[r:LINK_TO_AGG]-() ON (r.sentiment)")
        edge_agg_count = self._exec(
            "MATCH ()-[r:LINK_TO_AGG]->() RETURN count(r) AS c"
        ).result_set[0][0]
        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:
            raise BenchmarkImportError(
                f"FalkorDB aggregation validation failed: expected {EXPECTED_EDGE_AGG_COUNT}, got {edge_agg_count}"
            )

    def aggregate_graph(self):
        def run():
            return self._exec("""
                MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)
                RETURN s.name AS source, t.name AS target, sum(r.sentimentScore) AS sentiment, count(r) AS linkCount
                ORDER BY sentiment DESC
            """)

        return _timed_repeated(run, n=5)

    def common_neighbour_match(self, subreddit_names: list[str]):
        def run(name):
            return self._exec(
                """
                MATCH (s:Subreddit {name: $name})-[r1:LINK_TO_AGG]->(common:Subreddit)<-[r2:LINK_TO_AGG]-(newFriend:Subreddit)
                WHERE r1.sentiment > 0.5 AND r2.sentiment > 0.5
                  AND s <> newFriend
                  AND NOT EXISTS { (s)-[:LINK_TO_AGG]->(newFriend) }
                  AND NOT EXISTS { (newFriend)-[:LINK_TO_AGG]->(s) }
                RETURN newFriend.name AS newFriend, r2.sentiment - r1.sentiment AS delta_interest
                ORDER BY delta_interest DESC LIMIT 100
            """,
                name=name,
            )

        return _timed_per_input(run, subreddit_names)

    def cycle_detection(self, subreddit_names: list[str], category: str = "positive"):
        op = self._sentiment_op(category)

        def run(name):
            return self._exec(
                f"""
                MATCH p = (s:Subreddit {{name: $name}})-[:LINK_TO_AGG]->(a:Subreddit)-[:LINK_TO_AGG]->(b:Subreddit)-[:LINK_TO_AGG]->(s)
                WHERE all(r IN relationships(p) WHERE r.sentiment {op}) AND a <> b
                RETURN p LIMIT 500
            """,
                name=name,
            )

        return _timed_per_input(run, subreddit_names)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()


def wait_falkordb_ready(port: int, timeout: int = 60):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            client = FalkorDB(host="localhost", port=port)
            client.select_graph("_healthcheck").query("RETURN 1")
            client.close()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"falkordb not ready on port {port}") from last_err
