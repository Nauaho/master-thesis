# surrealdb.py

import time
import csv

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
    FRIENDS_OF_FRIENDS_SENTIMENT,
    _timed_repeated,
    _timed_index_build,
    _timed_per_input,
    _timed_match,
)

INDEX_NAME = "subreddit_embeddings"
EMBEDDING_DIMENSION = 300


class SurrealDBBenchmark(GraphBenchmarks, VectorBenchmarks):

    def __init__(self, port: int):
        self.url = f"ws://localhost:{port}/rpc"
        self.db_name = "surrealdb"
        self._db = None

    def __enter__(self):
        self._db = Surreal(self.url)
        self._db.signin({
            "username": "root",
            "password": "root",
        })
        self._db.use("benchmark", "main")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._db is not None:
            self._db.close()
            self._db = None

        return False

    def _exec(self, query, params=None):
        return self._db.query(query, params)

    def _scalar_count(self, query: str) -> int:
        result = self._exec(query)

        rows = (
            result[0]["result"]
            if isinstance(result[0], dict)
            else result[0]
        )

        return rows[0]["c"] if rows else 0

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _define_schema(self):
        self._exec("""
            DEFINE TABLE OVERWRITE subreddit TYPE NORMAL SCHEMAFULL;

            DEFINE FIELD OVERWRITE name
                ON TABLE subreddit
                TYPE string;

            DEFINE FIELD OVERWRITE embedding
                ON TABLE subreddit
                TYPE option<array<float>>;

            DEFINE INDEX OVERWRITE subreddit_name
                ON TABLE subreddit
                FIELDS name UNIQUE;

            DEFINE TABLE OVERWRITE link_to
                TYPE RELATION
                IN subreddit
                OUT subreddit
                SCHEMAFULL;

            DEFINE FIELD OVERWRITE post_id
                ON TABLE link_to
                TYPE string;

            DEFINE FIELD OVERWRITE timestamp
                ON TABLE link_to
                TYPE datetime;

            DEFINE FIELD OVERWRITE sentiment_score
                ON TABLE link_to
                TYPE float;

            DEFINE FIELD OVERWRITE properties
                ON TABLE link_to
                TYPE array<float>;

            DEFINE INDEX OVERWRITE link_to_in_out
                ON TABLE link_to
                FIELDS in, out;

            DEFINE TABLE OVERWRITE link_to_agg
                TYPE RELATION
                IN subreddit
                OUT subreddit
                SCHEMAFULL;

            DEFINE FIELD OVERWRITE sentiment
                ON TABLE link_to_agg
                TYPE float;

            DEFINE FIELD OVERWRITE link_count
                ON TABLE link_to_agg
                TYPE int;

            DEFINE INDEX OVERWRITE link_to_agg_in_out
                ON TABLE link_to_agg
                FIELDS in, out UNIQUE;

            DEFINE INDEX OVERWRITE agg_link_sentiment
                ON TABLE link_to_agg
                FIELDS sentiment;
        """)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_data(self):
        try:
            self._define_schema()

            self._import_subreddits()
            self._import_links()
            self._import_embeddings()

        except Exception as e:
            raise BenchmarkImportError(
                f"SurrealDB import failed: {e}"
            ) from e

        return self._validate_import()


    def _import_subreddits(self, filename="data/reddit_agefreighter/subreddits.csv", batch_size=5000):
        """
        Import the complete, deduplicated subreddit list.

        CSV format:
            id,name
            32024,AskReddit
            57106,gaming
            ...
        """
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            batch = []

            for row in reader:
                batch.append({
                    "id": RecordID("subreddit", row["id"]),
                    "name": row["name"],
                })

                if len(batch) >= batch_size:
                    self._insert_subreddit_batch(batch)
                    batch.clear()

            if batch:
                self._insert_subreddit_batch(batch)


    def _insert_subreddit_batch(self, rows):
        self._exec(
            """
            INSERT INTO subreddit $rows
            ON DUPLICATE KEY UPDATE
                name = $input.name;
            """,
            {"rows": rows},
        )


    def _import_links(self, filename="data/reddit_agefreighter/links.csv", batch_size=5000):
        """
        Import all graph edges.

        CSV format:
            id,start_id,start_vertex_type,end_id,end_vertex_type,sentimentScore

        Example:
            1,32024,Subreddit,57106,Subreddit,1.0

        The vertex type columns are currently ignored because this
        benchmark uses only Subreddit -> Subreddit relations.
        """
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            batch = []

            for row in reader:
                batch.append({
                    "id": RecordID("link_to", row["id"]),
                    "in": RecordID("subreddit", row["start_id"]),
                    "out": RecordID("subreddit", row["end_id"]),
                    "sentimentScore": float(row["sentimentScore"]),
                })

                if len(batch) >= batch_size:
                    self._insert_link_batch(batch)
                    batch.clear()

            if batch:
                self._insert_link_batch(batch)


    def _insert_link_batch(self, rows):
        self._run(
            self._db.insert_relation("link_to", rows)
        )


    def _import_embeddings(
        self,
        filename="data/web-redditEmbeddings-subreddits.tsv",
        batch_size=1000,
    ):
        """
        Import subreddit embeddings.

        CSV format:
            subreddit_name,embedding_1,...,embedding_300
        """
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter='\t')
            batch = []

            for row in reader:
                if len(row) < 2:
                    continue

                try:
                    embedding = [float(x) for x in row[1:]]
                except ValueError:
                    continue

                if len(embedding) != EMBEDDING_DIMENSION:
                    continue

                if not any(x != 0.0 for x in embedding):
                    continue

                batch.append({
                    "id": RecordID("subreddit", row[0]),
                    "name": row[0],
                    "embedding": embedding,
                })

                if len(batch) >= batch_size:
                    self._insert_embedding_batch(batch)
                    batch.clear()

            if batch:
                self._insert_embedding_batch(batch)


    def _insert_embedding_batch(self, rows):
        self._exec(
            """
            INSERT INTO subreddit $rows
            ON DUPLICATE KEY UPDATE
                name = $input.name,
                embedding = $input.embedding;
            """,
            {"rows": rows},
        )

    def _validate_import(self):
        node_count = self._scalar_count("SELECT count() AS c FROM subreddit GROUP ALL;")
        embedded_count = self._scalar_count(
            "SELECT count() AS c FROM subreddit WHERE embedding IS NOT NONE GROUP ALL;"
        )
        edge_count = self._scalar_count("SELECT count() AS c FROM link_to GROUP ALL;")

        errors = []
        if node_count != EXPECTED_NODE_COUNT:
            errors.append(f"node count: expected {EXPECTED_NODE_COUNT}, got {node_count}")
        if embedded_count != EXPECTED_EMBEDDED_NODE_COUNT:
            errors.append(f"embedded node count: expected {EXPECTED_EMBEDDED_NODE_COUNT}, got {embedded_count}")
        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(f"edge count: expected {EXPECTED_EDGE_COUNT}, got {edge_count}")

        bad_length_count = self._scalar_count(f"""
            SELECT count() AS c FROM subreddit
            WHERE embedding IS NOT NONE AND array::len(embedding) != {EMBEDDING_DIMENSION}
            GROUP ALL;
        """)
        if bad_length_count != 0:
            errors.append(f"embedding dimension mismatch: {bad_length_count} node(s) affected")

        null_element_count = self._scalar_count("""
            SELECT count() AS c FROM subreddit
            WHERE embedding IS NOT NONE AND array::any(embedding, |$v| $v IS NONE)
            GROUP ALL;
        """)
        if null_element_count != 0:
            errors.append(f"embedding contains null elements: {null_element_count} node(s) affected")

        if errors:
            raise BenchmarkImportError("SurrealDB import validation failed:\n" + "\n".join(errors))
        return node_count, embedded_count, edge_count


    # ------------------------------------------------------------------
    # Vector index lifecycle
    # ------------------------------------------------------------------

    def hnsw_index_build(self):
        def build():
            self._exec(
                f"""
                DEFINE INDEX OVERWRITE {INDEX_NAME}
                ON TABLE subreddit
                FIELDS embedding
                HNSW
                DIMENSION {EMBEDDING_DIMENSION}
                DIST COSINE;
                """
            )

        def drop():
            self._exec(
                f"""
                REMOVE INDEX IF EXISTS {INDEX_NAME}
                ON TABLE subreddit;
                """
            )

        return _timed_index_build(
            build,
            n=5,
            cleanup=drop,
        )

    def ivf_index_build(self):
        """No native IVF; DISKANN fills the equivalent disk-backed ANN role."""

        def build():
            self._exec(
                f"""
                DEFINE INDEX OVERWRITE {INDEX_NAME}
                ON TABLE subreddit
                FIELDS embedding
                DISKANN
                DIMENSION {EMBEDDING_DIMENSION}
                DIST COSINE;
                """
            )

        def drop():
            self._exec(
                f"""
                REMOVE INDEX IF EXISTS {INDEX_NAME}
                ON TABLE subreddit;
                """
            )

        return _timed_index_build(
            build,
            n=5,
            cleanup=drop,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def persist_aggregation(self):
        self._exec("""
            LET $pairs = SELECT
                in,
                out,
                math::mean(sentiment_score) AS sentiment,
                count() AS link_count
            FROM link_to
            GROUP BY in, out;

            INSERT RELATION INTO link_to_agg $pairs;
        """)

        self._exec(
            """
            REBUILD INDEX link_to_agg_in_out
            ON TABLE link_to_agg;
            """
        )

        self._exec(
            """
            REBUILD INDEX agg_link_sentiment
            ON TABLE link_to_agg;
            """
        )

        edge_agg_count = self._scalar_count(
            """
            SELECT count() AS c
            FROM link_to_agg
            GROUP ALL;
            """
        )

        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:
            raise BenchmarkImportError(
                f"SurrealDB aggregation validation failed: "
                f"expected {EXPECTED_EDGE_AGG_COUNT}, "
                f"got {edge_agg_count}"
            )

    def aggregate_graph(self):
        def run():
            return self._exec("""
                SELECT
                    in.name AS source,
                    out.name AS target,
                    math::sum(sentiment_score) AS sentiment,
                    count() AS link_count
                FROM link_to
                GROUP BY in, out
                ORDER BY sentiment DESC;
            """)

        return _timed_repeated(
            run,
            n=5,
        )

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def common_neighbour_match(
        self,
        subreddit_names: list[str],
    ):
        def run(name: str):
            return self._exec(
                """
                LET $s = type::thing(
                    'subreddit',
                    $name
                );

                LET $common = array::distinct(
                    $s
                    ->(link_to_agg
                        WHERE sentiment > 0.33
                    )
                    ->subreddit
                );

                LET $candidates = array::complement(
                    array::distinct(
                        $common
                        <-(link_to_agg
                            WHERE sentiment > 0.33
                        )
                        <-subreddit
                    ),
                    [$s]
                );

                SELECT
                    id AS new_friend_id,
                    name AS new_friend,

                    (
                        SELECT VALUE sentiment
                        FROM ONLY link_to_agg
                        WHERE in = $s
                          AND out IN $common
                          AND sentiment > 0.33
                        ORDER BY sentiment DESC
                        LIMIT 1
                    ) AS r1_sentiment,

                    (
                        SELECT VALUE sentiment
                        FROM ONLY link_to_agg
                        WHERE out IN $common
                          AND in = $parent.id
                          AND sentiment > 0.33
                        ORDER BY sentiment DESC
                        LIMIT 1
                    ) AS r2_sentiment

                FROM $candidates

                WHERE id NOT IN (
                    $s->link_to_agg->subreddit
                )

                AND id NOT IN (
                    $s<-link_to_agg<-subreddit
                )

                LIMIT 100;
                """,
                {"name": name},
            )

        return _timed_per_input(
            run,
            subreddit_names,
        )

    def cycle_detection(
        self,
        subreddit_names: list[str],
        category: str = "positive",
    ):
        op = self._sentiment_op(category)

        def run(name: str):
            return self._exec(
                f"""
                LET $s = type::thing(
                    'subreddit',
                    $name
                );

                SELECT * FROM ONLY (
                    $s
                        ->(link_to_agg
                            WHERE sentiment {op}
                        )->subreddit

                        ->(link_to_agg
                            WHERE sentiment {op}
                        )->subreddit

                        ->(link_to_agg
                            WHERE sentiment {op}
                        )->subreddit
                )

                WHERE id = $s

                LIMIT {TRAVERSAL_LIMIT};
                """,
                {"name": name},
            )

        return _timed_per_input(
            run,
            subreddit_names,
        )

    def friends_of_friends(
        self,
        subreddit_names: list[str],
        pattern_lengths: range,
    ):
        def run(
            pattern_length: int,
            name: str,
        ):
            return self._exec(
                f"""
                LET $s = type::thing(
                    'subreddit',
                    $name
                );

                LET $raw = $s.{{{pattern_length}+path}}(
                    ->(
                        link_to_agg
                        WHERE sentiment >
                            {FRIENDS_OF_FRIENDS_SENTIMENT}
                    )->subreddit
                );

                RETURN array::filter(
                    $raw,
                    |$p|
                        $p.len() =
                        array::distinct($p).len()
                )[0..{TRAVERSAL_LIMIT}];
                """,
                {"name": name},
            )

        return _timed_match(
            run,
            pattern_lengths,
            subreddit_names,
        )

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def knn(
        self,
        query_vectors: dict,
        k: int = 10,
    ):
        def run_query(vec):
            return self._exec(
                """
                SELECT
                    name,
                    vector::distance::knn() AS dist
                FROM subreddit
                WHERE embedding IS NOT NONE
                  AND embedding <|$k, COSINE|> $vec;
                """,
                {
                    "vec": vec,
                    "k": k,
                },
            )

        return _timed_per_input(
            run_query,
            inputs=list(query_vectors.values()),
        )

    def ann(
        self,
        index_type: str,
        query_vectors: dict[str, list[float]],
        k: int = 10,
    ):
        if index_type == "ivf":
            self._exec(
                f"""
                DEFINE INDEX OVERWRITE {INDEX_NAME}
                ON TABLE subreddit
                FIELDS embedding
                DISKANN
                DIMENSION {EMBEDDING_DIMENSION}
                DIST COSINE;
                """
            )

        elif index_type == "hnsw":
            self._exec(
                f"""
                DEFINE INDEX OVERWRITE {INDEX_NAME}
                ON TABLE subreddit
                FIELDS embedding
                HNSW
                DIMENSION {EMBEDDING_DIMENSION}
                DIST COSINE;
                """
            )

        else:
            raise ValueError(
                f"Unknown index_type: {index_type}"
            )

        def run_query(vec):
            return self._exec(
                """
                SELECT
                    name,
                    vector::distance::knn() AS dist
                FROM subreddit
                WHERE embedding <|$k|> $vec;
                """,
                {
                    "vec": vec,
                    "k": k,
                },
            )

        try:
            return _timed_per_input(
                run_query,
                inputs=list(query_vectors.values()),
            )
        finally:
            self._exec(
                f"""
                REMOVE INDEX IF EXISTS {INDEX_NAME}
                ON TABLE subreddit;
                """
            )

def wait_surrealdb_ready(
    port: int,
    timeout: int = 60,
):
    url = f"http://localhost:{port}"

    deadline = time.time() + timeout
    last_err = None

    while time.time() < deadline:
        try:
            db = Surreal(url)
            db.signin({
                "username": "root",
                "password": "root",
            })

            # Actually verify that the connection can select
            # the namespace/database used by the benchmark.
            db.use("benchmark", "main")

            return

        except Exception as e:
            last_err = e
            time.sleep(1)

    raise TimeoutError(
        f"SurrealDB not ready on {url}"
    ) from last_err