# benches/falkordb_bench.py
import time
import csv
from datetime import datetime
from falkordb import FalkorDB, Graph
from .base import (
    GraphBenchmarks,
    VectorBenchmarks,
    BenchmarkImportError,
    _timed_repeated,
    _timed_per_input,
    _timed_index_build,
    _timed_match,
    EXPECTED_NODE_COUNT,
    EXPECTED_EDGE_COUNT,
    EXPECTED_EMBEDDED_NODE_COUNT,
    EXPECTED_EDGE_AGG_COUNT,
    FRIENDS_OF_FRIENDS_SENTIMENT,
    TRAVERSAL_LIMIT
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

                        except ValueError, KeyError:
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
        node_count = self._exec("MATCH (n:Subreddit) RETURN count(n) AS c").result_set[
            0
        ][0]
        embedded_count = self._exec(
            "MATCH (n:Subreddit) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
        ).result_set[0][0]
        edge_count = self._exec(
            "MATCH ()-[r:LINK_TO]->() RETURN count(r) AS c"
        ).result_set[0][0]

        errors = []
        if node_count != EXPECTED_NODE_COUNT:
            errors.append(
                f"node count: expected {EXPECTED_NODE_COUNT}, got {node_count}"
            )
        if embedded_count != EXPECTED_EMBEDDED_NODE_COUNT:
            errors.append(
                f"embedded node count: expected {EXPECTED_EMBEDDED_NODE_COUNT}, got {embedded_count}"
            )
        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(
                f"edge count: expected {EXPECTED_EDGE_COUNT}, got {edge_count}"
            )
        if errors:
            raise BenchmarkImportError(
                "FalkorDB import validation failed:\n" + "\n".join(errors)
            )

        return node_count, embedded_count, edge_count

    def ivf_index_build(self):
        print(f"[{self.db_name}] IVF index build not supported, skipping.")
        return None

    def hnsw_index_build(self):
        def build():
            self._graph.create_node_vector_index('Subreddit', 'embedding', dim=VECTOR_DIM, similarity_function='cosine')
            wait_for_hnsw_index(
                self._graph,
                label="Subreddit",
                property_name="embedding",
                interval=0.005,
            )
        def drop():
            self._graph.drop_node_vector_index(label='Subreddit', attribute='embedding')
            time.sleep(10)

        return _timed_index_build(build, n=5, cleanup=drop)

    def knn(self, query_vectors: dict, k: int = 10):
        def run_query(vec: list[float]):
            return self._exec(
                f"""
                MATCH (e:Subreddit)
                WHERE e.embedding IS NOT NULL 
                RETURN e.name, vec.cosineDistance(e.embedding, vecf32({str(vec)})) AS cos_distance
                ORDER BY cos_distance ASC
                LIMIT {k}
                """
            )

        return _timed_per_input(run_query, inputs=list(query_vectors.values()))

    def ann(self, index_type: str, query_vectors: dict, k: int = 10):
        if index_type == "ivf":
            print(f"[{self.db_name}] IVF not supported, skipping ANN-IVF.")
            return None
        if index_type != "hnsw":
            raise ValueError(f"Unknown index_type: {index_type}")

        self._exec("""
            CREATE VECTOR INDEX FOR (n:Subreddit) ON (n.embedding) OPTIONS {dimension:$dim_count, similarityFunction:'cosine'}
        """, dim_count=VECTOR_DIM)

        wait_for_hnsw_index(
            self._graph,
            label="Subreddit",
            property_name="embedding",
            interval=0.005,
        )

        time.sleep(10)
        print(
            self._exec(
                """
                CALL db.indexes()
                """
            ).result_set
        )

        def run_query(vec: list[float]):
            return self._exec(
            """
                CALL db.idx.vector.queryNodes('Subreddit', 'embedding', $k, vecf32($vec))
                YIELD node, score
                RETURN node.name, score
            """
            , k=k, vec=vec)

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

    def adamic_adar(self, subreddit_names: list[str]):
        def run(name):
            return self._exec(
                """
                MATCH (s:Subreddit {{name: $name}})-[r1:LINK_TO_AGG]->(common:Subreddit)<-[r2:LINK_TO_AGG]-(newFriend:Subreddit)
                WHERE r1.sentiment > 0.5 AND r2.sentiment > 0.5
                    AND s <> newFriend
                    AND NOT (s)-[:LINK_TO_AGG]->(newFriend)
                    AND NOT (newFriend)-[:LINK_TO_AGG]->(s)
                RETURN newFriend.name AS newFriend, r2.sentiment - r1.sentiment AS delta_interest
                ORDER BY delta_interest DESC
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
                RETURN p
            """,
                name=name,
            )

        return _timed_per_input(run, subreddit_names)

    def friends_of_friends(self, subreddit_names: list[str]):
        def run(name: str, pattern_length: int):
            return self._exec(f"""
                MATCH p = (s:Subreddit {{name: $name}})-[:LINK_TO_AGG*{pattern_length}]->(friend:Subreddit)
                WHERE all(r IN relationships(p) WHERE r.sentiment > {FRIENDS_OF_FRIENDS_SENTIMENT})
                AND all(n IN nodes(p) WHERE single(x IN nodes(p) WHERE x = n))
                RETURN p
            """, name=name)
        return _timed_match(run, subreddit_names)

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

def wait_for_hnsw_index(
    graph: Graph,
    label: str,
    property_name: str,
    interval: float = 0.010,
    timeout: float = 600.0,
):
    """
    Wait until a FalkorDB HNSW/vector index becomes operational.

    Polls graph.list_indices() every `interval` seconds.

    Raises:
        TimeoutError: if the index does not become operational within
                      `timeout` seconds.
    """
    deadline = time.perf_counter() + timeout

    while True:
        indices = graph.list_indices()

        for index in indices.result_set:
            index_label = index[0]
            fields = index[1]
            field_types = index[2]
            status = index[7]

            if (
                index_label == label
                and property_name in fields
                and "VECTOR" in field_types.get(property_name, [])
            ):
                if status == "OPERATIONAL":
                    return

                break

        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"FalkorDB HNSW index "
                f"'{label}.{property_name}' did not become operational "
                f"within {timeout:.1f}s"
            )

        time.sleep(interval)