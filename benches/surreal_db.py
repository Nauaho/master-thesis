# surrealdb.py
import time
import csv
import asyncio
from surrealdb import Surreal, RecordID
from .base import (
    GraphBenchmarks,
    VectorBenchmarks,
    BenchmarkImportError,
    EXPECTED_NODE_COUNT,
    EXPECTED_EDGE_COUNT,
    EXPECTED_EDGE_AGG_COUNT,
    EXPECTED_EMBEDDED_NODE_COUNT,
    TRAVERSAL_LIMIT,
    _timed_repeated,
    _timed_index_build,
    _timed_per_input,
    _timed_match,
)

INDEX_NAME = "subreddit_embeddings"  # single source of truth
EMBEDDING_DIMENSION = 300  # derived from the source embeddings file; not hardcoded per-query


class SurrealDBBenchmark(GraphBenchmarks, VectorBenchmarks):
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

    async def _scalar_count(self, query: str) -> int:
        result = await self._db.query(query)
        rows = result[0]["result"] if isinstance(result[0], dict) else result[0]
        return rows[0]["c"] if rows else 0

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #

    async def _define_schema(self):
        await self._db.query(
            """
            DEFINE TABLE OVERWRITE subreddit TYPE NORMAL SCHEMAFULL;
            DEFINE FIELD OVERWRITE name ON TABLE subreddit TYPE string;
            DEFINE FIELD OVERWRITE embedding ON TABLE subreddit TYPE option<array<float>>;
            DEFINE INDEX OVERWRITE subreddit_name ON TABLE subreddit FIELDS name UNIQUE;

            DEFINE TABLE OVERWRITE link_to TYPE RELATION IN subreddit OUT subreddit SCHEMAFULL;
            DEFINE FIELD OVERWRITE post_id ON TABLE link_to TYPE string;
            DEFINE FIELD OVERWRITE timestamp ON TABLE link_to TYPE datetime;
            DEFINE FIELD OVERWRITE sentiment_score ON TABLE link_to TYPE float;
            DEFINE FIELD OVERWRITE properties ON TABLE link_to TYPE array<float>;
            -- composite index: aggregation groups by (in, out)
            DEFINE INDEX OVERWRITE link_to_in_out ON TABLE link_to FIELDS in, out;

            DEFINE TABLE OVERWRITE link_to_agg TYPE RELATION IN subreddit OUT subreddit SCHEMAFULL;
            DEFINE FIELD OVERWRITE sentiment ON TABLE link_to_agg TYPE float;
            DEFINE FIELD OVERWRITE link_count ON TABLE link_to_agg TYPE int;
            DEFINE INDEX OVERWRITE link_to_agg_in_out ON TABLE link_to_agg FIELDS in, out UNIQUE;
            -- queries filter on sentiment (common_neighbour_match, cycle_detection)
            DEFINE INDEX OVERWRITE agg_link_sentiment ON TABLE link_to_agg FIELDS sentiment;
            """
        )

    # ------------------------------------------------------------------ #
    # Import — native bulk INSERT / INSERT RELATION
    # ------------------------------------------------------------------ #

    async def import_data(self):
        try:
            await self._connect()
            await self._define_schema()

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
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            nodes, edges = set(), []

            for row in reader:
                source, target = row["SOURCE_SUBREDDIT"], row["TARGET_SUBREDDIT"]
                nodes.add(source)
                nodes.add(target)
                edges.append(
                    {
                        "in": RecordID("subreddit", source),
                        "out": RecordID("subreddit", target),
                        "post_id": row["POST_ID"],
                        "timestamp": row["TIMESTAMP"].replace(" ", "T"),
                        "sentiment_score": float(row["LINK_SENTIMENT"]),
                        "properties": [
                            float(p) for p in row["PROPERTIES"].split(",")
                        ],
                    }
                )
                if len(edges) >= batch_size:
                    await self._upsert_nodes(nodes)
                    await self._db.insert_relation("link_to", edges)
                    nodes, edges = set(), []

            if nodes:
                await self._upsert_nodes(nodes)
            if edges:
                await self._db.insert_relation("link_to", edges)

    async def _upsert_nodes(self, names: set[str]):
        rows = [{"id": RecordID("subreddit", n), "name": n} for n in names]
        # native bulk upsert: single INSERT statement, dedup via ON DUPLICATE KEY UPDATE
        await self._db.query(
            "INSERT INTO subreddit $rows ON DUPLICATE KEY UPDATE name = $input.name;",
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
                    len(emb) != EMBEDDING_DIMENSION
                    or any(x is None for x in emb)
                    or all(x == 0.0 for x in emb)
                ):
                    continue

                batch.append(
                    {"id": RecordID("subreddit", row[0]), "name": row[0], "embedding": emb}
                )
                if len(batch) >= batch_size:
                    await self._upsert_embedding_batch(batch)
                    batch = []

            if batch:
                await self._upsert_embedding_batch(batch)

    async def _upsert_embedding_batch(self, rows: list[dict]):
        await self._db.query(
            """
            INSERT INTO subreddit $rows
            ON DUPLICATE KEY UPDATE name = $input.name, embedding = $input.embedding;
            """,
            {"rows": rows},
        )

    async def _validate_import(self):
        node_count = await self._scalar_count("SELECT count() AS c FROM subreddit GROUP ALL;")
        embedded_count = await self._scalar_count(
            "SELECT count() AS c FROM subreddit WHERE embedding IS NOT NONE GROUP ALL;"
        )
        edge_count = await self._scalar_count("SELECT count() AS c FROM link_to GROUP ALL;")

        errors = []
        if node_count != EXPECTED_NODE_COUNT:
            errors.append(f"node count: expected {EXPECTED_NODE_COUNT}, got {node_count}")
        if embedded_count != EXPECTED_EMBEDDED_NODE_COUNT:
            errors.append(
                f"embedded node count: expected {EXPECTED_EMBEDDED_NODE_COUNT}, got {embedded_count}"
            )
        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(f"edge count: expected {EXPECTED_EDGE_COUNT}, got {edge_count}")

        bad_length_count = await self._scalar_count(
            f"""
            SELECT count() AS c FROM subreddit
            WHERE embedding IS NOT NONE AND array::len(embedding) != {EMBEDDING_DIMENSION}
            GROUP ALL;
            """
        )
        if bad_length_count != 0:
            errors.append(
                f"embedding dimension mismatch: {bad_length_count} node(s) have embedding length != {EMBEDDING_DIMENSION}"
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

    # ------------------------------------------------------------------ #
    # Vector index lifecycle — HNSW (in-memory) / DISKANN (disk-backed)
    # ------------------------------------------------------------------ #

    async def hnsw_index_build(self):
        async def build():
            await self._db.query(
                f"DEFINE INDEX OVERWRITE {INDEX_NAME} ON TABLE subreddit "
                f"FIELDS embedding HNSW DIMENSION {EMBEDDING_DIMENSION} DIST COSINE;"
            )

        async def drop():
            await self._db.query(f"REMOVE INDEX IF EXISTS {INDEX_NAME} ON TABLE subreddit;")

        return await _timed_index_build(build, n=5, cleanup=drop)

    async def ivf_index_build(self):
        """IVF has no native equivalent in SurrealDB; DISKANN fills the same
        role (disk-backed ANN index for corpora too large to keep in RAM)."""
        async def build():
            await self._db.query(
                f"DEFINE INDEX OVERWRITE {INDEX_NAME} ON TABLE subreddit "
                f"FIELDS embedding DISKANN DIMENSION {EMBEDDING_DIMENSION} DIST COSINE;"
            )

        async def drop():
            await self._db.query(f"REMOVE INDEX IF EXISTS {INDEX_NAME} ON TABLE subreddit;")

        return await _timed_index_build(build, n=5, cleanup=drop)

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    async def persist_aggregation(self):
        await self._db.query(
            """
            LET $pairs = SELECT
                    in, out,
                    math::mean(sentiment_score) AS sentiment,
                    count() AS link_count
                FROM link_to
                GROUP BY in, out;

            INSERT RELATION INTO link_to_agg $pairs;
            """
        )
     
        await self._exec("REBUILD INDEX link_to_agg_in_out ON TABLE link_to_agg;")
        await self._exec("REBUIK INDEX agg_link_sentiment ON TABLE link_to_agg;")

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
        # SurrealDB traversal collapses each hop to a distinct node set (no
        # path identity like Cypher), so r1/r2 sentiment is recovered with a
        # direct edge lookup on the resolved ids rather than carried on the path.
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

    async def cycle_detection(self, subreddit_names: list[str], category: str = "positive"):
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
                LIMIT {TRAVERSAL_LIMIT};
                """,
                {"name": name},
            )

        return await _timed_per_input(run, subreddit_names)

    # ------------------------------------------------------------------ #
    # Vector search — native KNN operator
    # ------------------------------------------------------------------ #

    async def knn(self, query_vectors: dict, k: int = 10):
        # explicit distance inside <|K, DIST|> always forces brute force,
        # even with an index defined on the field
        async def run_query(vec):
            return await self._db.query(
                """
                SELECT name, vector::distance::knn() AS dist
                FROM subreddit
                WHERE embedding IS NOT NONE AND embedding <|$k, COSINE|> $vec;
                """,
                {"vec": vec, "k": k},
            )

        return await _timed_per_input(run_query, inputs=list(query_vectors.values()))

    async def ann(self, index_type: str, query_vectors: dict[str, list[float]], k: int = 10):
        if index_type == "ivf":
            await self._db.query(
                f"DEFINE INDEX OVERWRITE {INDEX_NAME} ON TABLE subreddit "
                f"FIELDS embedding DISKANN DIMENSION {EMBEDDING_DIMENSION} DIST COSINE;"
            )
        elif index_type == "hnsw":
            await self._db.query(
                f"DEFINE INDEX OVERWRITE {INDEX_NAME} ON TABLE subreddit "
                f"FIELDS embedding HNSW DIMENSION {EMBEDDING_DIMENSION} DIST COSINE;"
            )
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        async def run_query(vec):
            # bare <|k|> routes through whichever index is defined on the field
            return await self._db.query(
                """
                SELECT name, vector::distance::knn() AS dist
                FROM subreddit
                WHERE embedding <|$k|> $vec;
                """,
                {"vec": vec, "k": k},
            )

        try:
            return await _timed_per_input(run_query, inputs=list(query_vectors.values()))
        finally:
            await self._db.query(f"REMOVE INDEX IF EXISTS {INDEX_NAME} ON TABLE subreddit;")

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