import time

from pathlib import Path

import psycopg

from .base import (

    GraphBenchmarks,

    BenchmarkImportError,

    _timed_repeated,

    _timed_per_input,

    _timed_match,

    CYCLE_DETECTION_SENTIMENT,

    ADAMIC_AGAR_MIN_SENTIMENT,

    MIN_LINKS_AGGREGATED,

    EXPECTED_EDGE_COUNT,

    EXPECTED_EDGE_AGG_COUNT,

    EXPECTED_NODE_COUNT,

    FRIENDS_OF_FRIENDS_SENTIMENT,

    P99_DEGREE

)



GRAPH_NAME = "reddit"

INPUT_FILES = [

    Path("data/soc-redditHyperlinks-body.tsv"),

    Path("data/soc-redditHyperlinks-title.tsv"),

]



class AGEBenchmark(GraphBenchmarks):

    def __init__(self, port: int):

        self._port = 5432

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

        self._conn.execute('SET search_path = ag_catalog, "$user", public')

        try:

            self._conn.execute(f"SELECT create_graph('{GRAPH_NAME}')")

        except psycopg.errors.UniqueViolation:

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

        return value.replace("'", "\\\\'")

    def _create_labels(self) -> None:

        print("Creating vertex/edge labels...")

        try:

            self._conn.execute(

                f"SELECT create_vlabel('{GRAPH_NAME}', 'Subreddit')"

            )

        except psycopg.errors.UniqueViolation:

            pass

        try:

            self._conn.execute(

                f"SELECT create_elabel('{GRAPH_NAME}', 'LINK_TO')"

            )

        except psycopg.errors.UniqueViolation:

            pass

    def _create_indexes(self) -> None:

        """

        Create indexes after bulk loading.

        """

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

        self._conn.execute(

            f"""

            CREATE INDEX idf_float_mapping_of_sentiment 

            ON {GRAPH_NAME}."LINK_TO"

            USING btree (

                (ag_catalog.agtype_to_float8(

                    ag_catalog.agtype_access_operator(

                        VARIADIC ARRAY[properties, '"sentimentScore"'::ag_catalog.agtype]

                        )

                    )

                )

            );

            """

        )

    def _analyze(self) -> None:

        print("Running ANALYZE...")

        self._conn.execute(f'ANALYZE {GRAPH_NAME}."Subreddit";')

        self._conn.execute(f'ANALYZE {GRAPH_NAME}."LINK_TO";')

    def _validate_import(self) -> tuple[int, int]:

        """

        Perform validation only once, after the entire import.

        """

        print("Validating import...")

        node_count = int(

            self._exec(

                """

            MATCH (n:Subreddit)

            RETURN count(n)

            """

            )[0][0]

        )

        edge_count = int(

            self._exec(

                """

            MATCH ()-[r:LINK_TO]->()

            RETURN count(r)

            """

            )[0][0]

        )

        errors = []

        if node_count != EXPECTED_NODE_COUNT + 9:

            errors.append(

                "node count benchmark mismatch: "

                f"expected "

                f"{EXPECTED_NODE_COUNT + 9}, "

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

                "AGE import validation failed:\n" + "\n".join(errors)

            )

        print(f"Validation successful: {node_count:,} nodes, {edge_count:,} edges")

        return node_count, edge_count

    def import_data(self):

        try:

            print("=== AGE Reddit import ===")

            self._create_labels()

            self._conn.execute(

                f"""

                SELECT load_labels_from_file(

                    '{GRAPH_NAME}',

                    'Subreddit',

                    'data/normalised_csvs/subreddits_light.csv'

                )

                """

            )

            self._conn.execute(

                f"""

                SELECT load_edges_from_file(

                    '{GRAPH_NAME}',

                    'LINK_TO',

                    'data/normalised_csvs/links_age.csv'

                )

                """

            )

            self._create_indexes()

            self._analyze()

            return self._validate_import()

        except BenchmarkImportError:

            raise

        except Exception as e:

            raise BenchmarkImportError(f"AGE import failed: {e}") from e

    def persist_aggregation(self):

        print("Aggregating LINK_TO edges...")

        self._exec(

            """

            MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)

            WITH s, t, avg(r.sentimentScore) AS avgSentiment, count(r) AS linkCount

            CREATE (s)-[agg:LINK_TO_AGG]->(t)

            SET

                agg.sentiment = avgSentiment,

                agg.linkCount = linkCount

            """

        )

        # LINK_TO_AGG indexes created immediately — every materialization

        # step below filters/traverses on agg.sentiment, so they need to

        # exist before those queries run, not just before benchmarking.

        self._conn.execute(

            f"""

            CREATE INDEX IF NOT EXISTS

            idx_link_to_agg_id

            ON {GRAPH_NAME}."LINK_TO_AGG"

            USING BTREE (id);

            """

        )

        self._conn.execute(

            f"""

            CREATE INDEX IF NOT EXISTS

            idx_link_to_agg_properties

            ON {GRAPH_NAME}."LINK_TO_AGG"

            USING GIN (properties);

            """

        )

        self._conn.execute(

            f"""

            CREATE INDEX IF NOT EXISTS

            idx_link_to_agg_start_id

            ON {GRAPH_NAME}."LINK_TO_AGG"

            USING BTREE (start_id);

            """

        )

        self._conn.execute(

            f"""

            CREATE INDEX IF NOT EXISTS

            idx_link_to_agg_end_id

            ON {GRAPH_NAME}."LINK_TO_AGG"

            USING BTREE (end_id);

            """

        )

        self._conn.execute(

            f"""

            CREATE INDEX idx_link_agg_sentimentScore

            ON {GRAPH_NAME}."LINK_TO_AGG"

            USING btree (

                agtype_access_operator(

                    VARIADIC ARRAY[

                        properties,

                        '"sentiment"'::agtype

                    ]

                )

            );

            """

        )

        self._conn.execute(f'ANALYZE {GRAPH_NAME}."LINK_TO_AGG";')

        edge_agg_count = int(

            self._exec(

                """

                MATCH ()-[r:LINK_TO_AGG]->()

                RETURN count(r)

                """

            )[0][0]

        )

        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:

            raise BenchmarkImportError(

                "AGE aggregation validation failed: "

                f"expected {EXPECTED_EDGE_AGG_COUNT}, "

                f"got {edge_agg_count}"

            )


    def aggregate_graph(self):

        def run():

            return self._exec(

                """

                MATCH (s:Subreddit)-[r:LINK_TO]->(t:Subreddit)

                RETURN

                    s.name,

                    t.name,

                    avg(r.sentimentScore) AS sentiment,

                    count(r)

                ORDER BY sentiment DESC

                """,

                out_cols=(

                    "source agtype, target agtype, sentiment agtype, link_count agtype"

                ),

            )

        return _timed_repeated(

            run,

            n=5,

        )

    def adamic_adar(self, subreddit_names: list[str]):

        def run(name: str):

            return self._exec(

                f"""

                MATCH (s:Subreddit {{name: '{self._escape(name)}'}})-[r1:LINK_TO_AGG]->(common:Subreddit)

                WHERE r1.sentiment >= {ADAMIC_AGAR_MIN_SENTIMENT}

                AND common.degree <= {P99_DEGREE}

                MATCH (common)<-[r2:LINK_TO_AGG]-(newFriend:Subreddit)

                WHERE r2.sentiment >= {ADAMIC_AGAR_MIN_SENTIMENT}

                AND newFriend.degree <= {P99_DEGREE}

                AND s <> newFriend

                OPTIONAL MATCH (s)-[existing:LINK_TO_AGG]-(newFriend)

                WITH newFriend, common, r1, r2, existing

                WHERE existing IS NULL

                WITH newFriend,

                    count(DISTINCT common) AS commonNeighborsCount,

                    avg(abs(r1.sentiment - r2.sentiment)) AS avgDeltaSentiment,

                    sum(1.0 / log(common.degree + 2)) AS adamicAdarScore

                WHERE commonNeighborsCount >= 3

                RETURN newFriend.name,

                    commonNeighborsCount,

                    avgDeltaSentiment,

                    adamicAdarScore,

                    adamicAdarScore * (1 - avgDeltaSentiment) AS combinedScore

                ORDER BY combinedScore DESC

                """,

                out_cols=(

                    "suggested agtype, commonNeighborsCount agtype, "

                    "avgDeltaSentiment agtype, adamicAdarScore agtype, combinedScore agtype"

                ),

            )

        return _timed_per_input(run, subreddit_names)

    def cycle_detection(self, subreddit_names: list[str]):

        def run(name: str):

            return self._exec(

                f"""

                MATCH p = (s:Subreddit {{name: '{self._escape(name)}'}})-[r1:LINK_TO_AGG]->(a:Subreddit)-[r2:LINK_TO_AGG]->(b:Subreddit)-[r3:LINK_TO_AGG]->(s)

                WHERE r1.sentiment >= {CYCLE_DETECTION_SENTIMENT}

                AND r2.sentiment >= {CYCLE_DETECTION_SENTIMENT}

                AND r3.sentiment >= {CYCLE_DETECTION_SENTIMENT}

                AND a <> b AND a.name < b.name

                AND a.degree <= {P99_DEGREE} AND b.degree <= {P99_DEGREE}

                MATCH (a)-[rev1:LINK_TO_AGG]->(s)

                WHERE rev1.sentiment >= {CYCLE_DETECTION_SENTIMENT}

                MATCH (b)-[rev2:LINK_TO_AGG]->(a)

                WHERE rev2.sentiment >= {CYCLE_DETECTION_SENTIMENT}

                MATCH (s)-[rev3:LINK_TO_AGG]->(b)

                WHERE rev3.sentiment >= {CYCLE_DETECTION_SENTIMENT}

                RETURN p

                """,

                out_cols="p agtype",

            )

        return _timed_per_input(run, subreddit_names)

    def friends_of_friends(self, subreddit_names: list[str]):

        def run(name: str, pattern_length: int):

            return self._exec(

                f"""

                MATCH p = (s:Subreddit {{name: '{name}'}})-[:LINK_TO_AGG*1..{pattern_length}]->(friend:Subreddit)

                WITH p, nodes(p) AS pathNodes, relationships(p) AS pathRelationships

                UNWIND pathRelationships AS r

                WITH p, pathNodes, pathRelationships, r

                WHERE r.sentiment >= {FRIENDS_OF_FRIENDS_SENTIMENT}

                AND r.linkCount >= {MIN_LINKS_AGGREGATED}

                WITH p, pathNodes, pathRelationships, count(r) AS valid_count

                WHERE valid_count = length(p)

                UNWIND pathNodes AS n

                WITH p, count(n) AS total_count, count(DISTINCT id(n)) AS unique_count

                WHERE total_count = unique_count

                RETURN p

                """,

                out_cols="p agtype",

            )

        return _timed_match(run, subreddit_names)

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

    raise TimeoutError(f"AGE (postgres) not ready on port {port}") from last_err