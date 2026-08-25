import csv
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import psycopg

from .base import (
    GraphBenchmarks,
    BenchmarkImportError,
    _timed_repeated,
    _timed_per_input,
    EXPECTED_NODES_WITHOUT_EMBEDDED_DATASET,
    EXPECTED_EDGE_COUNT,
    EXPECTED_EDGE_AGG_COUNT,
)


GRAPH_NAME = "reddit"

INPUT_FILES = [
    Path("data/soc-redditHyperlinks-body.tsv"),
    Path("data/soc-redditHyperlinks-title.tsv"),
]


class AGEBenchmark(GraphBenchmarks):

    def __init__(self, port: int):
        self._port = port

        self._conn = psycopg.connect(
            host="localhost",
            port=port,
            dbname="postgres",
            user="postgres",
            password="password",
            autocommit=True,
        )

        self._conn.execute("CREATE EXTENSION IF NOT EXISTS age")
        self._conn.execute("LOAD 'age'")
        self._conn.execute(
            'SET search_path = ag_catalog, "$user", public'
        )

        try:
            self._conn.execute(
                f"SELECT create_graph('{GRAPH_NAME}')"
            )
        except psycopg.errors.UniqueViolation:
            # Graph already exists.
            pass

        self.db_name = "age"

    def _exec(
        self,
        cypher: str,
        out_cols: str = "result agtype",
    ):
        """
        Execute raw Cypher through AGE's cypher() SQL function.
        """
        return self._conn.execute(
            f"""
            SELECT *
            FROM cypher(
                '{GRAPH_NAME}',
                $$ {cypher} $$
            ) AS ({out_cols})
            """
        ).fetchall()

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "\\'")

    # ------------------------------------------------------------------
    # AGEFreighter import
    # ------------------------------------------------------------------

    def _prepare_agefreighter_data(
        self,
        output_dir: Path,
    ) -> tuple[Path, int, int]:
        """
        Convert the two Reddit TSV files into the CSV format expected
        by AGEFreighter.

        Produces:
            subreddits.csv
            links.csv
            config.json

        Returns:
            (config_path, node_count, edge_count)
        """

        output_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------
        # Pass 1:
        # Collect unique subreddit names.
        # --------------------------------------------------------------

        subreddits: set[str] = set()

        for input_path in INPUT_FILES:
            print(f"Scanning {input_path}...")

            with input_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as f:
                reader = csv.DictReader(f, delimiter="\t")

                required_columns = {
                    "SOURCE_SUBREDDIT",
                    "TARGET_SUBREDDIT",
                    "LINK_SENTIMENT",
                }

                if not required_columns.issubset(
                    reader.fieldnames or []
                ):
                    raise BenchmarkImportError(
                        f"{input_path} is missing required columns. "
                        f"Found: {reader.fieldnames}"
                    )

                for row in reader:
                    source = row["SOURCE_SUBREDDIT"]
                    target = row["TARGET_SUBREDDIT"]

                    if not source or not target:
                        raise BenchmarkImportError(
                            f"Empty subreddit name in {input_path}"
                        )

                    subreddits.add(source)
                    subreddits.add(target)

        print(
            f"Found {len(subreddits):,} unique subreddits"
        )

        # --------------------------------------------------------------
        # Assign deterministic numeric IDs.
        #
        # IMPORTANT:
        # AGEFreighter's graph input IDs are logical IDs from the CSV.
        # We use integers here and keep the subreddit name as a property.
        # --------------------------------------------------------------

        ordered_subreddits = sorted(subreddits)

        subreddit_to_id = {
            name: idx
            for idx, name in enumerate(
                ordered_subreddits,
                start=1,
            )
        }

        # --------------------------------------------------------------
        # Write vertex CSV.
        # --------------------------------------------------------------

        nodes_path = output_dir / "subreddits.csv"

        with nodes_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "id",
                "name",
            ])

            for name in ordered_subreddits:
                writer.writerow([
                    subreddit_to_id[name],
                    name,
                ])

        # --------------------------------------------------------------
        # Write edge CSV.
        #
        # We deliberately combine body + title into ONE LINK_TO edge
        # type, exactly as the original importer did.
        #
        # Every original row becomes one edge.
        # Therefore duplicate source/target pairs are preserved.
        # --------------------------------------------------------------

        edges_path = output_dir / "links.csv"

        edge_id = 1

        with edges_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "id",
                "start_id",
                "start_vertex_type",
                "end_id",
                "end_vertex_type",
                "sentimentScore",
            ])

            for input_path in INPUT_FILES:
                print(f"Converting {input_path}...")

                with input_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as source:
                    reader = csv.DictReader(
                        source,
                        delimiter="\t",
                    )

                    for row in reader:
                        source_name = row["SOURCE_SUBREDDIT"]
                        target_name = row["TARGET_SUBREDDIT"]

                        sentiment = float(
                            row["LINK_SENTIMENT"]
                        )

                        writer.writerow([
                            edge_id,
                            subreddit_to_id[source_name],
                            "Subreddit",
                            subreddit_to_id[target_name],
                            "Subreddit",
                            sentiment,
                        ])

                        edge_id += 1

        edge_count = edge_id - 1

        # --------------------------------------------------------------
        # AGEFreighter config.
        #
        # One node type + one edge type.
        # --------------------------------------------------------------

        config_path = output_dir / "config.json"

        config = {
            "edge": {
                "csv_path": str(edges_path.resolve()),
                "type": "LINK_TO",
                "props": [
                    "sentimentScore",
                ],
                "start_vertex": {
                    "csv_path": str(nodes_path.resolve()),
                    "id": "id",
                    "label": "Subreddit",
                    "props": [
                        "name",
                    ],
                },
                "end_vertex": {
                    "csv_path": str(nodes_path.resolve()),
                    "id": "id",
                    "label": "Subreddit",
                    "props": [
                        "name",
                    ],
                },
            }
        }

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                config,
                f,
                indent=2,
            )

        print(f"Prepared nodes: {nodes_path}")
        print(f"Prepared edges: {edges_path}")
        print(f"Prepared config: {config_path}")

        return (
            config_path,
            len(subreddits),
            edge_count,
        )

    def _run_agefreighter(
        self,
        config_path: Path,
    ) -> None:
        """
        Invoke AGEFreighter's CLI from the current Python process.
        """

        agefreighter = shutil.which("agefreighter")

        if agefreighter is None:
            raise BenchmarkImportError(
                "AGEFreighter executable was not found in PATH. "
                "Install it with: pip install agefreighter"
            )

        pg_connection_string = (
            f"host=localhost "
            f"port={self._port} "
            f"dbname=postgres "
            f"user=postgres "
            f"password=password"
        )

        command = [
            agefreighter,
            "--graphname",
            GRAPH_NAME,
            "--pg-con-str",
            pg_connection_string,
            "load",
            "--source-type",
            "csv",
            "--config",
            str(config_path),
            "--progress",
        ]

        print()
        print("Running AGEFreighter:")
        print(" ".join(command))
        print()

        result = subprocess.run(
            command,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise BenchmarkImportError(
                "AGEFreighter failed with exit code "
                f"{result.returncode}"
            )

    def _create_indexes(self) -> None:
        """
        Create indexes after bulk loading.
        """
        print("Converting imported values to the proper types")
        self._exec(
        """
        MATCH ()-[r:LINK_TO]->()
        SET r.sentimentScore = toFloat(r.sentimentScore)        
        """)

        print("Creating indexes...")

        # Index for looking up Subreddit by the "name" property.
        self._conn.execute(
            f"""
            CREATE UNIQUE INDEX idx_subreddit_name_unique
            ON {GRAPH_NAME}."Subreddit"
            USING btree (
                agtype_access_operator(
                    VARIADIC ARRAY[
                        properties,
                        '"name"'::agtype
                    ]
                )
            );
            """
        )

        # AGE's internal vertex id.
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
                idx_subreddit_id
            ON {GRAPH_NAME}."Subreddit"
            USING BTREE (id);
            """
        )

        # Properties.
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
                idx_subreddit_properties
            ON {GRAPH_NAME}."Subreddit"
            USING GIN (properties);
            """
        )

        # Relationship indexes.
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
                idx_link_to_id
            ON {GRAPH_NAME}."LINK_TO"
            USING BTREE (id);
            """
        )

        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
                idx_link_to_properties
            ON {GRAPH_NAME}."LINK_TO"
            USING GIN (properties);
            """
        )

        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
                idx_link_to_start_id
            ON {GRAPH_NAME}."LINK_TO"
            USING BTREE (start_id);
            """
        )

        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
                idx_link_to_end_id
            ON {GRAPH_NAME}."LINK_TO"
            USING BTREE (end_id);
            """
        )

    def _analyze(self) -> None:
        print("Running ANALYZE...")

        self._conn.execute(
            f'ANALYZE {GRAPH_NAME}."Subreddit";'
        )

        self._conn.execute(
            f'ANALYZE {GRAPH_NAME}."LINK_TO";'
        )

    def _validate_import(self) -> tuple[int, int]:
        """
        Perform validation only once, after the entire import.
        """

        print("Validating import...")

        node_count = int(self._exec(
            """
            MATCH (n:Subreddit)
            RETURN count(n)
            """
        )[0][0])

        edge_count = int(self._exec(
            """
            MATCH ()-[r:LINK_TO]->()
            RETURN count(r)
            """
        )[0][0])

        print(f"{edge_count} : {type(edge_count)}")
        print(f"{node_count} : {type(node_count)}")
        errors = []

        if node_count != EXPECTED_NODES_WITHOUT_EMBEDDED_DATASET:
            errors.append(
                "node count benchmark mismatch: "
                f"expected "
                f"{EXPECTED_NODES_WITHOUT_EMBEDDED_DATASET}, "
                f"got {node_count}"
            )

        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(
                "edge count benchmark mismatch: "
                f"expected {EXPECTED_EDGE_COUNT}, "
                f"got {edge_count}"
            )

        if errors:
            raise BenchmarkImportError(
                "AGE import validation failed:\n"
                + "\n".join(errors)
            )

        print(
            f"Validation successful: "
            f"{node_count:,} nodes, "
            f"{edge_count:,} edges"
        )

        return node_count, edge_count

    # ------------------------------------------------------------------
    # Public import
    # ------------------------------------------------------------------

    def import_data(self):
        """
        Import the Reddit dataset using AGEFreighter.

        Pipeline:

            Reddit TSV files
                ↓
            preprocess to CSV
                ↓
            AGEFreighter COPY import
                ↓
            indexes
                ↓
            ANALYZE
                ↓
            final validation
        """

        try:
            print("=== AGE Reddit import ===")

            with tempfile.TemporaryDirectory(
                prefix="age_reddit_"
            ) as temp_dir:

                temp_path = Path(temp_dir)

                print(
                    f"Preparing AGEFreighter data in "
                    f"{temp_path}"
                )

                (
                    config_path,
                    expected_nodes,
                    expected_edges,
                ) = self._prepare_agefreighter_data(
                    temp_path
                )

                print()
                print(
                    f"Expected nodes: "
                    f"{expected_nodes:,}"
                )
                print(
                    f"Expected edges: "
                    f"{expected_edges:,}"
                )
                print()

                # Bulk import.
                self._run_agefreighter(
                    config_path
                )

            # CSV files are no longer needed after AGEFreighter exits.
            print()
            print("AGEFreighter import complete.")

            # Index only after bulk loading.
            self._create_indexes()

            # Update planner statistics.
            self._analyze()

            # One validation pass.
            return self._validate_import()

        except BenchmarkImportError:
            raise

        except Exception as e:
            raise BenchmarkImportError(
                f"AGE import failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def persist_aggregation(self):
        self._exec(
            """
            MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)
            WITH s, t, avg(r.sentimentScore) AS sentiment, count(r) AS linkCount
            MERGE (s)-[agg:LINK_TO_AGG]->(t)
            SET
                agg.sentiment = sentiment,
                agg.linkCount = linkCount
            """
        )

        self._exec(
            f"""
            CREATE IF NOT EXISTS INDEX idx_link_agg_sentimentScore
            ON {GRAPH_NAME}."LINK_TO"
            USING btree (
                agtype_access_operator(
                    VARIADIC ARRAY[
                        properties,
                        '"sentimentScore"'::agtype
                    ]
                )
            );
            """
        )

        edge_agg_count = self._exec(
            """
            MATCH ()-[r:LINK_TO_AGG]->()
            RETURN count(r)
            """
        )[0][0]

        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:
            raise BenchmarkImportError(
                "AGE aggregation validation failed: "
                f"expected {EXPECTED_EDGE_AGG_COUNT}, "
                f"got {edge_agg_count}"
            )

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def aggregate_graph(self):

        def run():
            return self._exec(
                """
                MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)
                RETURN
                    s.name,
                    t.name,
                    sum(r.sentimentScore),
                    count(r)
                ORDER BY sum(r.sentimentScore) DESC
                """,
                out_cols=(
                    "source agtype, "
                    "target agtype, "
                    "sentiment agtype, "
                    "linkCount agtype"
                ),
            )

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
                f"""
                MATCH (s:Subreddit {{name: '{self._escape(name)}'}})-[r1:LINK_TO_AGG]->(common:Subreddit)<-[r2:LINK_TO_AGG]-(newFriend:Subreddit)
                WHERE r1.sentiment > 0.33 AND r2.sentiment > 0.33 AND s <> newFriend
                    AND NOT (s)-[:LINK_TO_AGG]->(newFriend)
                    AND NOT (newFriend)-[:LINK_TO_AGG]->(s)
                RETURN newFriend.name, r2.sentiment - r1.sentiment AS delta_interest
                ORDER BY
                    delta_interest DESC

                LIMIT 100
                """,
                out_cols=(
                    "newFriend agtype, "
                    "delta_interest agtype"
                ),
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
                MATCH p = (s:Subreddit {{name: '{self._escape(name)}'}})-[:LINK_TO_AGG]->(a:Subreddit)-[:LINK_TO_AGG]->(b:Subreddit)-[:LINK_TO_AGG]->(s)
                WHERE all(r IN relationships(p) WHERE r.sentiment {op}) AND a <> b
                RETURN p
                LIMIT 500
                """,
                out_cols="p agtype",
            )

        return _timed_per_input(
            run,
            subreddit_names,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        self._conn.close()
        return False


def wait_age_ready(
    port: int,
    timeout: int = 60,
):
    deadline = time.time() + timeout
    last_err = None

    while time.time() < deadline:
        try:
            conn = psycopg.connect(
                host="localhost",
                port=port,
                dbname="postgres",
                user="postgres",
                password="password",
                connect_timeout=3,
            )

            conn.close()
            return

        except Exception as e:
            last_err = e
            time.sleep(1)

    raise TimeoutError(
        f"AGE (postgres) not ready on port {port}"
    ) from last_err