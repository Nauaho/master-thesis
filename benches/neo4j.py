# neo4j.py
import time
from neo4j import GraphDatabase
from .base import (
    GraphBenchmarks,
    VectorBenchmarks,
    BenchmarkImportError,
    EXPECTED_NODE_COUNT,
    EXPECTED_EDGE_COUNT,
    EXPECTED_EDGE_AGG_COUNT,
    EXPECTED_EMBEDDED_NODE_COUNT,
    _timed_repeated,
    _timed_index_build,
    _timed_per_input,
)

INDEX_NAME = "subreddit_embeddings"  # single source of truth — was mismatched before


class Neo4jBenchamrk(GraphBenchmarks, VectorBenchmarks):
    def __init__(self, port: int):
        self._driver = GraphDatabase.driver(
            f"bolt://localhost:{port}", auth=("neo4j", "password")
        )
        self.db_name = "neo4j"

    def _exec(self, query, **params):
        return self._driver.execute_query(query, **params)

    def _exec_autocommit(self, query, **params):
        """Implicit/auto-commit transaction — required for CALL {...} IN TRANSACTIONS."""
        with self._driver.session() as session:
            return session.run(query, **params)

    def import_data(self):
        try:
            self._exec_autocommit(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subreddit) REQUIRE s.name IS UNIQUE"
            )

            for filename in [
                "soc-redditHyperlinks-body.tsv",
                "soc-redditHyperlinks-title.tsv",
            ]:
                self._exec_autocommit(f"""
                    LOAD CSV WITH HEADERS FROM 'file:///{filename}' AS row
                    FIELDTERMINATOR '\\t'
                    CALL (row) {{
                        WITH row
                        MERGE (source:Subreddit {{name: row.SOURCE_SUBREDDIT}})
                        MERGE (target:Subreddit {{name: row.TARGET_SUBREDDIT}})
                        CREATE (source)-[r:LINK_TO {{
                            postId: row.POST_ID,
                            timestamp: replace(row.TIMESTAMP, ' ', 'T'),
                            sentimentScore: toFloat(row.LINK_SENTIMENT),
                            properties: [p IN split(row.PROPERTIES, ',') | toFloat(p)]
                        }}]->(target)
                    }} IN TRANSACTIONS OF 5000 ROWS
                """)

            self._exec_autocommit("""
                LOAD CSV FROM 'file:///web-redditEmbeddings-subreddits.csv' AS row
                CALL (row) {
                    WITH row
                    WITH row, [x IN row[1..] | toFloat(x)] AS emb
                    WHERE size(emb) = 300
                    AND NONE(x IN emb WHERE x IS NULL)
                    AND ANY(x IN emb WHERE x <> 0.0)
                    MERGE (s:Subreddit {name: row[0]})
                    SET s.embedding = emb
                } IN TRANSACTIONS OF 1000 ROWS
            """)
        except Exception as e:
            raise BenchmarkImportError(f"Neo4j import failed: {e}") from e

        node_records, _, _ = self._exec("MATCH (n:Subreddit) RETURN count(n) AS c")
        embedded_records, _, _ = self._exec(
            "MATCH (n:Subreddit) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
        )
        edge_records, _, _ = self._exec("MATCH ()-[r:LINK_TO]->() RETURN count(r) AS c")

        node_count = node_records[0]["c"]
        embedded_count = embedded_records[0]["c"]
        edge_count = edge_records[0]["c"]

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

        # NEW: validate embedding shape/content, not just presence
        bad_length_records, _, _ = self._exec("""
            MATCH (s:Subreddit) WHERE s.embedding IS NOT NULL
            WITH s, size(s.embedding) AS len
            WHERE len <> 300
            RETURN count(s) AS c
        """)
        bad_length_count = bad_length_records[0]["c"]
        if bad_length_count != 0:
            errors.append(
                f"embedding dimension mismatch: {bad_length_count} node(s) have embedding length != 300"
            )

        null_element_records, _, _ = self._exec("""
            MATCH (s:Subreddit) WHERE s.embedding IS NOT NULL
            WITH s, [x IN s.embedding WHERE x IS NULL] AS nulls
            WHERE size(nulls) > 0
            RETURN count(s) AS c
        """)
        null_element_count = null_element_records[0]["c"]
        if null_element_count != 0:
            errors.append(
                f"embedding contains null elements: {null_element_count} node(s) affected"
            )

        if errors:
            raise BenchmarkImportError(
                "Neo4j import validation failed:\n" + "\n".join(errors)
            )

        return node_count, embedded_count, edge_count

    def _wait_index_online(self, index_name: str, timeout: int = 120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            records, _, _ = self._exec(
                "SHOW INDEXES YIELD name, state WHERE name = $name", name=index_name
            )
            if records and records[0]["state"] == "ONLINE":
                return
            time.sleep(0.5)
        raise TimeoutError(f"Index {index_name} did not come online in time")

    def ivf_index_build(self):
        print(f"[{self.db_name}] IVF index build not supported, skipping.")
        return None

    def hnsw_index_build(self):
        def build():
            self._exec(f"""
                CREATE VECTOR INDEX `{INDEX_NAME}`
                FOR (n:Subreddit) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: 300,
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
            self._wait_index_online(INDEX_NAME)

        def drop():
            self._exec(f"DROP INDEX `{INDEX_NAME}` IF EXISTS")

        return _timed_index_build(build, n=5, cleanup=drop)

    def persist_aggregation(self):
        self._exec_autocommit("""
            MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)
            WITH s, t, avg(r.sentimentScore) AS sentiment, count(r) AS linkCount
            CALL (s, t, sentiment, linkCount) {
                MERGE (s)-[agg:LINK_TO_AGG]->(t)
                SET agg.sentiment = sentiment, agg.linkCount = linkCount
            } IN TRANSACTIONS OF 5000 ROWS
        """)

        self._exec("""
            CREATE INDEX agg_link_sentiment FOR ()-[r:LINK_TO_AGG]->() ON (r.sentiment);
        """)

        edge_agg_records, _, _ = self._exec(
            "MATCH ()-[r:LINK_TO_AGG]->() RETURN count(r) AS c"
        )
        edge_agg_count = edge_agg_records[0][
            "c"
        ]  # fixed: was comparing the raw records list, not the count
        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:
            raise BenchmarkImportError(
                f"Neo4j import validation failed:\nexpected {EXPECTED_EDGE_AGG_COUNT} aggregated edges, got {edge_agg_count}"
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
        def run(name: str):
            return self._exec(
                """
                MATCH (s:Subreddit {name: $name})-[r1:LINK_TO_AGG]->(common:Subreddit)<-[r2:LINK_TO_AGG]-(newFriend:Subreddit)
                WHERE r1.sentiment > 0.33 AND r2.sentiment > 0.33 AND s <> newFriend
                    AND NOT EXISTS { (s)-[:LINK_TO_AGG]->(newFriend) }
                    AND NOT EXISTS { (newFriend)-[:LINK_TO_AGG]->(s) }
                RETURN newFriend.name AS newFriend, r2.sentiment - r1.sentiment AS delta_interest
                ORDER BY delta_interest DESC
                LIMIT 100
            """,
                name=name,
            )

        return _timed_per_input(run, subreddit_names)

    def cycle_detection(self, subreddit_names: list[str], category: str = "positive"):
        op = self._sentiment_op(category)

        def run(name: str):
            return self._exec(
                f"""
                MATCH p = (s:Subreddit {{name: $name}})-[:LINK_TO_AGG]->(a:Subreddit)-[:LINK_TO_AGG]->(b:Subreddit)-[:LINK_TO_AGG]->(s)
                WHERE all(r IN relationships(p) WHERE r.sentiment {op})
                AND a <> b
                RETURN p
                LIMIT 500
            """,
                name=name,
            )

        return _timed_per_input(run, subreddit_names)

    def knn(self, query_vectors: dict, k: int = 10):
        def run_query(vec):
            return self._exec(
                """
                MATCH (s:Subreddit)
                WHERE s.embedding IS NOT NULL
                WITH s, vector.similarity.cosine(s.embedding, $queryVector) AS score
                RETURN s.name AS name, score
                ORDER BY score DESC LIMIT $k
            """,
                queryVector=vec,
                k=k,
            )

        return _timed_per_input(run_query, inputs=list(query_vectors.values()))

    def ann(self, index_type: str, query_vectors: dict[str, list[float]], k: int = 10):
        if index_type == "ivf":
            print(f"[{self.db_name}] IVF not supported, skipping ANN-IVF.")
            return None
        if index_type != "hnsw":
            raise ValueError(f"Unknown index_type: {index_type}")

        # one-time setup: build index BEFORE the timed loop, not per-rep
        self._exec(f"""
            CREATE VECTOR INDEX `{INDEX_NAME}`
            FOR (n:Subreddit) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 300,
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
        self._wait_index_online(INDEX_NAME)

        def run_query(vec):
            return self._exec(
                f"""
                CALL db.index.vector.queryNodes('{INDEX_NAME}', $k, $queryVector)
                YIELD node, score
                RETURN node.name, score
            """,
                queryVector=vec,
                k=k,
            )

        try:
            # bug fix: dict.values() gives the vectors; original code iterated
            # keys only (`for _, x in dict` doesn't work without .items())
            return _timed_per_input(run_query, inputs=list(query_vectors.values()))
        finally:
            # one-time teardown: drop AFTER all timed queries are done
            self._exec(f"DROP INDEX `{INDEX_NAME}` IF EXISTS")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._driver.close()
        return False


def wait_neo4j_ready(port: int, timeout: int = 60):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            driver = GraphDatabase.driver(
                f"bolt://localhost:{port}", auth=("neo4j", "password")
            )
            driver.verify_connectivity()
            driver.close()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"neo4j not ready on port {port}") from last_err
