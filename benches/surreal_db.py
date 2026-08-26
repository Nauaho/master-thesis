# surrealdb.py
import time
import csv
import asyncio
from surreal_db import Surreal
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

INDEX_NAME = "subreddit_embeddings"  # single source of truth


class SurrealDBBenchmark(GraphBenchmarks, VectorBenchmarks):
    """
    Graph model:
        subreddit             -- node table
        link_to  (RELATION)   -- raw edges, subreddit ->link_to-> subreddit
        link_to_agg (RELATION)-- aggregated edges, subreddit ->link_to_agg-> subreddit

    All multi-hop reads use SurrealQL arrow traversal (`->edge->table`,
    `<-edge<-table`) with inline edge filtering (`->(edge WHERE cond)->table`)
    rather than manual JOINs, per SurrealDB's recommended graph-traversal
    patterns: https://surrealdb.com/docs/learn/data-models/graph/graph-traversal
    """

    def __init__(self, url: str = "ws://localhost:8000"):
        self.url = url
        self._db = None
        self.db_name = "surrealdb"

    async def _connect(self):
        if self._db is None:
            self._db = Surreal(self.url)
            await self._db.connect()
            await self._db.use("benchmark", "main")

    async def _exec(self, query, **params):
        await self._connect()
        return await self._db.query(query, params)

    # ------------------------------------------------------------------ #
    # Import
    # ------------------------------------------------------------------ #

    async def import_data(self):
        try:
            await self._connect()

            await self._db.query(
                """
                DEFINE TABLE subreddit SCHEMAFULL;
                DEFINE FIELD name ON TABLE subreddit TYPE string;
                DEFINE FIELD embedding ON TABLE subreddit TYPE option<array<float>>;
                DEFINE INDEX subreddit_name ON TABLE subreddit FIELDS name UNIQUE;

                DEFINE TABLE link_to TYPE RELATION IN subreddit OUT subreddit SCHEMAFULL;
                DEFINE FIELD post_id ON TABLE link_to TYPE string;
                DEFINE FIELD timestamp ON TABLE link_to TYPE datetime;
                DEFINE FIELD sentiment_score ON TABLE link_to TYPE float;
                DEFINE FIELD properties ON TABLE link_to TYPE array<float>;

                DEFINE TABLE link_to_agg TYPE RELATION IN subreddit OUT subreddit SCHEMAFULL;
                DEFINE FIELD sentiment ON TABLE link_to_agg TYPE float;
                DEFINE FIELD link_count ON TABLE link_to_agg TYPE int;
                DEFINE INDEX agg_link_sentiment ON TABLE link_to_agg FIELDS sentiment;
                """
            )

            for filename in [
                "soc-redditHyperlinks-body.tsv",
                "soc-redditHyperlinks-title.tsv",
            ]:
                await self._import_links_from_tsv(filename)

            await self._import_embeddings_from_csv(
                "web-redditEmbeddings-subreddits.csv"
            )
        except Exception as e:
            raise BenchmarkImportError(f"SurrealDB import failed: {e}") from e

        return await self._validate_import()

    async def _import_links_from_tsv(self, filename: str, batch_size: int = 5000):
        """RELATE subreddit->link_to->subreddit for every row, batched."""
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            batch = []

            for row in reader:
                batch.append(
                    {
                        "source": row["SOURCE_SUBREDDIT"],
                        "target": row["TARGET_SUBREDDIT"],
                        "post_id": row["POST_ID"],
                        "timestamp": row["TIMESTAMP"].replace(" ", "T"),
                        "sentiment_score": float(row["LINK_SENTIMENT"]),
                        "properties": [float(p) for p in row["PROPERTIES"].split(",")],
                    }
                )
                if len(batch) >= batch_size:
                    await self._relate_link_batch(batch)
                    batch = []

            if batch:
                await self._relate_link_batch(batch)

    async def _relate_link_batch(self, rows: list[dict]):
        await self._db.query(
            """
            FOR $row IN $rows {
                UPSERT type::thing('subreddit', $row.source) SET name = $row.source;
                UPSERT type::thing('subreddit', $row.target) SET name = $row.target;

                RELATE (type::thing('subreddit', $row.source))
                    ->link_to->
                    (type::thing('subreddit', $row.target))
                SET
                    post_id = $row.post_id,
                    timestamp = <datetime>$row.timestamp,
                    sentiment_score = $row.sentiment_score,
                    properties = $row.properties;
            };
            """,
            {"rows": rows},
        )

    async def _import_embeddings_from_csv(self, filename: str, batch_size: int = 1000):
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            batch = []

            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    emb = [float(x) for x in row[1:]]
                except ValueError:
                    continue

                if (
                    len(emb) != 300
                    or any(x is None for x in emb)
                    or all(x == 0.0 for x in emb)
                ):
                    continue

                batch.append({"name": row[0], "embedding": emb})
                if len(batch) >= batch_size:
                    await self._upsert_embedding_batch(batch)
                    batch = []

            if batch:
                await self._upsert_embedding_batch(batch)

    async def _upsert_embedding_batch(self, rows: list[dict]):
        await self._db.query(
            """
            FOR $row IN $rows {
                UPSERT type::thing('subreddit', $row.name)
                    SET name = $row.name, embedding = $row.embedding;
            };
            """,
            {"rows": rows},
        )

    async def _validate_import(self):
        node_count = await self._scalar_count(
            "SELECT count() AS c FROM subreddit GROUP ALL;"
        )
        embedded_count = await self._scalar_count(
            "SELECT count() AS c FROM subreddit WHERE embedding IS NOT NONE GROUP ALL;"
        )
        edge_count = await self._scalar_count(
            "SELECT count() AS c FROM link_to GROUP ALL;"
        )

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

        bad_length_count = await self._scalar_count(
            """
            SELECT count() AS c FROM subreddit
            WHERE embedding IS NOT NONE AND array::len(embedding) != 300
            GROUP ALL;
            """
        )
        if bad_length_count != 0:
            errors.append(
                f"embedding dimension mismatch: {bad_length_count} node(s) have embedding length != 300"
            )

        null_element_count = await self._scalar_count(
            """
            SELECT count() AS c FROM subreddit
            WHERE embedding IS NOT NONE AND array::any(embedding, |$v| $v IS NONE)
            GROUP ALL;
            """
        )
        if null_element_count != 0:
            errors.append(
                f"embedding contains null elements: {null_element_count} node(s) affected"
            )

        if errors:
            raise BenchmarkImportError(
                "SurrealDB import validation failed:\n" + "\n".join(errors)
            )

        return node_count, embedded_count, edge_count

    async def _scalar_count(self, query: str) -> int:
        result = await self._db.query(query)
        rows = result[0]["result"] if isinstance(result[0], dict) else result[0]
        return rows[0]["c"] if rows else 0

    # ------------------------------------------------------------------ #
    # Vector index lifecycle
    # ------------------------------------------------------------------ #

    async def ivf_index_build(self):
        print(f"[{self.db_name}] IVF index build not supported, skipping.")
        return None

    async def hnsw_index_build(self):
        async def build():
            await self._db.query(
                f"""
                DEFINE INDEX {INDEX_NAME} ON TABLE subreddit
                    FIELDS embedding
                    HNSW DIMENSION 300 DIST COSINE;
                """
            )

        async def drop():
            await self._db.query(f"REMOVE INDEX {INDEX_NAME} ON TABLE subreddit;")

        return await _timed_index_build(build, n=5, cleanup=drop)

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    async def persist_aggregation(self):
        # Aggregate raw edges grouped by (in, out), then materialize as
        # link_to_agg edges via RELATE.
        await self._db.query(
            """
            LET $pairs = SELECT
                    in, out,
                    math::mean(sentiment_score) AS sentiment,
                    count() AS link_count
                FROM link_to
                GROUP BY in, out;

            FOR $pair IN $pairs {
                RELATE ($pair.in)->link_to_agg->($pair.out)
                    SET sentiment = $pair.sentiment, link_count = $pair.link_count;
            };
            """
        )

        edge_agg_count = await self._scalar_count(
            "SELECT count() AS c FROM link_to_agg GROUP ALL;"
        )
        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:
            raise BenchmarkImportError(
                f"SurrealDB import validation failed:\nexpected {EXPECTED_EDGE_AGG_COUNT} aggregated edges, got {edge_agg_count}"
            )

    async def aggregate_graph(self):
        async def run():
            return await self._db.query(
                """
                SELECT
                    in.name AS source,
                    out.name AS target,
                    math::sum(sentiment_score) AS sentiment,
                    count() AS link_count
                FROM link_to
                GROUP BY in, out
                ORDER BY sentiment DESC;
                """
            )

        return await _timed_repeated(run, n=5)

    # ------------------------------------------------------------------ #
    # Graph traversal — arrow syntax with inline edge filtering
    # ------------------------------------------------------------------ #

    async def common_neighbour_match(self, subreddit_names: list[str]):
        """
        Mirrors Cypher's:
            (s)-[r1:LINK_TO_AGG]->(common)<-[r2:LINK_TO_AGG]-(newFriend)
            WHERE r1.sentiment > 0.33 AND r2.sentiment > 0.33 AND s <> newFriend
              AND NOT (s)-[:LINK_TO_AGG]->(newFriend)
              AND NOT (newFriend)-[:LINK_TO_AGG]->(s)

        via arrow traversal + inline edge filters. SurrealDB's traversal
        collapses each hop to a distinct node set (no path identity), so
        the per-pair (r1, r2) sentiment values are recovered with a direct
        edge lookup keyed on the now-known record ids, rather than carried
        along the path the way Cypher does natively.
        """

        async def run(name: str):
            return await self._db.query(
                """
                LET $s = type::thing('subreddit', $name);

                LET $common = array::distinct(
                    $s->(link_to_agg WHERE sentiment > 0.33)->subreddit
                );

                LET $candidates = array::complement(
                    array::distinct(
                        $common<-(link_to_agg WHERE sentiment > 0.33)<-subreddit
                    ),
                    [$s]
                );

                SELECT
                    id AS new_friend_id,
                    name AS new_friend,
                    (
                        SELECT VALUE sentiment FROM ONLY link_to_agg
                        WHERE in = $s AND out IN $common AND sentiment > 0.33
                        ORDER BY sentiment DESC LIMIT 1
                    ) AS r1_sentiment,
                    (
                        SELECT VALUE sentiment FROM ONLY link_to_agg
                        WHERE out IN $common AND in = $parent.id AND sentiment > 0.33
                        ORDER BY sentiment DESC LIMIT 1
                    ) AS r2_sentiment
                FROM $candidates
                WHERE id NOT IN ($s->link_to_agg->subreddit)
                    AND id NOT IN ($s<-link_to_agg<-subreddit)
                LIMIT 100;
                """,
                {"name": name},
            )

        return await _timed_per_input(run, subreddit_names)

    async def cycle_detection(
        self, subreddit_names: list[str], category: str = "positive"
    ):
        """
        Mirrors Cypher's 3-hop cycle:
            (s)-[:LINK_TO_AGG]->(a)-[:LINK_TO_AGG]->(b)-[:LINK_TO_AGG]->(s)
            WHERE all sentiments satisfy `op`, a <> b

        Chained arrow traversal walks three hops in one statement; the
        result is the flattened set of subreddits reachable in exactly
        three qualifying hops. A cycle back to the start closes when `$s`
        itself appears in that set.
        """
        op = self._sentiment_op(category)

        async def run(name: str):
            return await self._db.query(
                f"""
                LET $s = type::thing('subreddit', $name);

                SELECT * FROM ONLY (
                    $s
                        ->(link_to_agg WHERE sentiment {op})->subreddit
                        ->(link_to_agg WHERE sentiment {op})->subreddit
                        ->(link_to_agg WHERE sentiment {op})->subreddit
                )
                WHERE id = $s
                LIMIT 500;
                """,
                {"name": name},
            )

        return await _timed_per_input(run, subreddit_names)

    # ------------------------------------------------------------------ #
    # Vector search — SurrealDB's KNN operator
    # ------------------------------------------------------------------ #

    async def knn(self, query_vectors: dict, k: int = 10):
        """
        Brute-force exact KNN. Passing an explicit distance function inside
        `<|K, DIST|>` always forces the brute-force path, even if an index
        exists on the field — see the vector search cheat sheet:
        https://surrealdb.com/docs/learn/data-models/vector-search/vector-indexes
        """

        async def run_query(vec):
            return await self._db.query(
                """
                SELECT
                    name,
                    vector::distance::knn() AS dist
                FROM subreddit
                WHERE embedding IS NOT NONE AND embedding <|$k, COSINE|> $vec;
                """,
                {"vec": vec, "k": k},
            )

        return await _timed_per_input(run_query, inputs=list(query_vectors.values()))

    async def ann(
        self, index_type: str, query_vectors: dict[str, list[float]], k: int = 10
    ):
        if index_type == "ivf":
            print(f"[{self.db_name}] IVF not supported, skipping ANN-IVF.")
            return None
        if index_type != "hnsw":
            raise ValueError(f"Unknown index_type: {index_type}")

        # One-time setup: build index BEFORE the timed loop
        await self._db.query(
            f"""
            DEFINE INDEX {INDEX_NAME} ON TABLE subreddit
                FIELDS embedding
                HNSW DIMENSION 300 DIST COSINE;
            """
        )

        async def run_query(vec):
            # Bare `<|k|>` (no explicit distance) routes through the HNSW
            # index, using the distance function it was defined with.
            return await self._db.query(
                """
                SELECT
                    name,
                    vector::distance::knn() AS dist
                FROM subreddit
                WHERE embedding <|$k|> $vec;
                """,
                {"vec": vec, "k": k},
            )

        try:
            return await _timed_per_input(
                run_query, inputs=list(query_vectors.values())
            )
        finally:
            await self._db.query(f"REMOVE INDEX {INDEX_NAME} ON TABLE subreddit;")

    # ------------------------------------------------------------------ #

    async def __aenter__(self):
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._db:
            await self._db.close()
        return False


async def wait_surrealdb_ready(url: str = "ws://localhost:8000", timeout: int = 60):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            db = Surreal(url)
            await db.connect()
            await db.close()
            return
        except Exception as e:
            last_err = e
            await asyncio.sleep(1)
    raise TimeoutError(f"SurrealDB not ready on {url}") from last_err
