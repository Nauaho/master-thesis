# surrealdb.py

import time
import csv
import json
from datetime import datetime, timezone
import traceback

from surrealdb import Surreal, RecordID, Datetime

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

    def _exec(self, query, **params):
        return self._db.query(query, params)

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

    def import_data(self):
        try:
            self._define_schema()
            self._import_subreddits()
            self._import_links()
        except Exception as e:
            traceback.print_exc()
            raise BenchmarkImportError(f"SurrealDB import failed: {e}") from e

        return self._validate_import()

    def _import_subreddits(
        self,
        filename="data/normalised_csvs/subreddits_full.csv",
        batch_size=5000,
    ):
        """
        Import the complete subreddit dataset.

        subreddits_full.csv contains the union of:
        1. every subreddit appearing in the link datasets;
        2. every subreddit appearing in the embedding dataset.

        CSV format:

            id,name,embedding

        Example:

            1,AskReddit,"[0.0123,-0.0456,...]"
            2,gaming,""
            3,programming,"[0.0012,0.0345,...]"

        Every subreddit is imported exactly once.

        The embedding column is optional because not every subreddit has
        an embedding. When present, it is converted from its CSV string
        representation into a list of floats before insertion.
        """
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            batch = []

            for row in reader:
                record = {
                    "id": RecordID("subreddit", row["id"]),
                    "name": row["name"],
                }

                embedding_raw = row.get("embedding", "").strip()

                if embedding_raw:
                    try:
                        embedding = json.loads(embedding_raw)

                        if (
                            not isinstance(embedding, list)
                            or len(embedding) != EMBEDDING_DIMENSION
                        ):
                            raise ValueError(
                                f"Expected {EMBEDDING_DIMENSION}-dimensional embedding"
                            )
                        
                        vector = [float(value) for value in embedding]
                        if(any(vector)):
                            record["embedding"] = vector
                        else:
                            continue
                    except (ValueError, SyntaxError) as exc:
                        raise BenchmarkImportError(
                            f"Invalid embedding for subreddit "
                            f"{row['name']!r}: {embedding_raw!r}"
                        ) from exc

                batch.append(record)

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
                name = $input.name,
                embedding = $input.embedding;
            """,
            rows=rows
        )

    def _import_links(
        self,
        filename="data/normalised_csvs/links.csv",
        batch_size=5000,
    ):
        """
        Import all graph edges.

        CSV format:

            id,start_id,start_vertex_type,end_id,end_vertex_type,
            sentimentScore,post_id,timestamp,properties

        The vertex-type columns are ignored because this benchmark
        contains only Subreddit -> Subreddit relations.

        The properties column is converted from its CSV string
        representation into a list of floats.
        """
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            batch = []

            for row in reader:
                properties_raw = row["properties"].strip()

                try:
                    properties = json.loads(properties_raw)
                    properties = [float(value) for value in properties]
                except (ValueError, SyntaxError, TypeError) as exc:
                    raise BenchmarkImportError(
                        f"Invalid properties for edge {row['id']}: "
                        f"{properties_raw!r}"
                    ) from exc

                date = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).isoformat()

                batch.append(
                    {
                        "id": RecordID("link_to", row["id"]),
                        "in": RecordID("subreddit", row["start_id"]),
                        "out": RecordID("subreddit", row["end_id"]),
                        "sentiment_score": float(row["sentimentScore"]),
                        "post_id": row["post_id"],
                        "timestamp": Datetime(date),
                        "properties": properties,
                    }
                )

                if len(batch) >= batch_size:
                    self._insert_link_batch(batch)
                    batch.clear()

            if batch:
                self._insert_link_batch(batch)

    def _insert_link_batch(self, rows):
        self._exec(
            """
            INSERT RELATION INTO link_to $rows;
            """,
            rows=rows
        )

    def _validate_import(self):
        result_nodes = self._exec("SELECT count() AS c FROM subreddit GROUP ALL;")
        node_count = result_nodes[0]["c"] if result_nodes else 0

        result_embedddings = self._exec("SELECT count() AS c FROM subreddit WHERE embedding IS NOT NONE GROUP ALL;")
        embedded_count = result_embedddings[0]["c"] if result_embedddings else 0

        result_edges = self._exec("SELECT count() AS c FROM link_to GROUP ALL;")
        edge_count = result_edges[0]["c"] if result_edges else 0
        
        errors = []
        if node_count != EXPECTED_NODE_COUNT:
            errors.append(f"node count: expected {EXPECTED_NODE_COUNT}, got {node_count}")
        if embedded_count != EXPECTED_EMBEDDED_NODE_COUNT:
            errors.append(f"embedded node count: expected {EXPECTED_EMBEDDED_NODE_COUNT}, got {embedded_count}")
        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(f"edge count: expected {EXPECTED_EDGE_COUNT}, got {edge_count}")

        # it is unsafe, idgaf
        bad_length_count = self._exec(f"""
            SELECT count() AS c FROM subreddit
            WHERE embedding IS NOT NONE AND array::len(embedding) != {EMBEDDING_DIMENSION}
            GROUP ALL;
        """)[0]["c"]
        if bad_length_count != 0:
            errors.append(f"embedding dimension mismatch: {bad_length_count} node(s) affected")

        null_element_count = self._exec("""
            SELECT count() AS c FROM subreddit
            WHERE embedding IS NOT NONE AND array::any(embedding, |$v| $v IS NONE)
            GROUP ALL;
        """)[0]["c"]
        if null_element_count != 0:
            errors.append(f"embedding contains null elements: {null_element_count} node(s) affected")

        if errors:
            raise BenchmarkImportError("SurrealDB import validation failed:\n" + "\n".join(errors))
        return node_count, embedded_count, edge_count

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

        edge_agg_count = self._exec(
            """
            SELECT count() AS c
            FROM link_to_agg
            GROUP ALL;
            """
        )[0]["c"]

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
                f"""
                SELECT
                    name,
                    vector::distance::knn() AS dist
                FROM subreddit
                WHERE embedding IS NOT NONE
                  AND embedding <|{k}, COSINE|> $vec;
                """,
                vec=vec
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
                ON subreddit
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
                f"""
                SELECT
                    name,
                    vector::distance::knn() AS dist
                FROM subreddit
                WHERE embedding <|{k}, 50|> $vec;
                """,
                vec=vec
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