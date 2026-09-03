import time
import csv
import psycopg
from .base import (
    GraphBenchmarks,
    BenchmarkImportError,
    _timed_repeated,
    _timed_per_input,
    _timed_match,
    EXPECTED_NODE_COUNT,
    EXPECTED_EDGE_COUNT,
    EXPECTED_EDGE_AGG_COUNT,
    ADAMIC_AGAR_MIN_SENTIMENT,
    CYCLE_DETECTION_SENTIMENT,
    MIN_LINKS_AGGREGATED,
    TRAVERSAL_LIMIT,
    P99_DEGREE,
    FRIENDS_OF_FRIENDS_SENTIMENT
)


class PostgresGraphBenchmark(GraphBenchmarks):
    def __init__(self, port: int):
        self._conn = psycopg.connect(
            host="localhost",
            port=port,
            dbname="postgres",
            user="postgres",
            password="password",
            autocommit=True,
        )
        self.db_name = "postgres_sql"

    def import_data(self):
        try:
            self._conn.execute(
                """
                CREATE TABLE subreddits (
                    id       BIGINT PRIMARY KEY,
                    subreddit TEXT NOT NULL UNIQUE,
                    degree   BIGINT NOT NULL DEFAULT 0
                );

                CREATE TABLE links_to (
                    id                BIGINT PRIMARY KEY,
                    start             BIGINT NOT NULL REFERENCES subreddits(id),
                    start_vertex_type TEXT,
                    finish            BIGINT NOT NULL REFERENCES subreddits(id),
                    end_vertex_type   TEXT,
                    sentiment         DOUBLE PRECISION NOT NULL,
                    ts                TIMESTAMP NOT NULL,
                    properties        JSONB NOT NULL,
                    post_id           TEXT NOT NULL
                );

                CREATE TABLE links_to_agg (
                    id          BIGINT PRIMARY KEY,
                    start       BIGINT NOT NULL REFERENCES subreddits(id),
                    finish      BIGINT NOT NULL REFERENCES subreddits(id),
                    sentiment   DOUBLE PRECISION NOT NULL,
                    link_count  BIGINT NOT NULL,
                    CONSTRAINT links_to_agg_start_finish_unique UNIQUE (start, finish)
                );
                """
            )

            with self._conn.cursor() as cur:
                with open("data/normalised_csvs/subreddits_light.csv") as f:
                    with cur.copy(
                        "COPY subreddits (id, subreddit) FROM STDIN WITH (FORMAT csv, HEADER true)"
                    ) as copy:
                        while data := f.read(1024 * 1024):
                            copy.write(data)

                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_subreddit_lookup ON subreddits (id)"
                )
                self._conn.execute("ANALYZE subreddits;")

                with open("data/normalised_csvs/links.csv") as f:
                    with cur.copy(
                        """COPY links_to
                        (id, start, start_vertex_type, finish, end_vertex_type,
                            sentiment, ts, properties, post_id)
                        FROM STDIN WITH (FORMAT csv, HEADER true)"""
                    ) as copy:
                        while data := f.read(1024 * 1024):
                            copy.write(data)

            # Drop the vertex-type columns now that they've served their purpose
            # as COPY landing spots. Comment this out if you'd rather keep them
            # (e.g. the dataset ever has non-"Subreddit" vertex types).

            self._conn.execute(
                """
                ALTER TABLE links_to DROP COLUMN start_vertex_type;
                ALTER TABLE links_to DROP COLUMN end_vertex_type;
                """
            )

            self._conn.execute(
                """
                CREATE INDEX links_to_start_idx ON links_to (start);
                CREATE INDEX links_to_finish_idx ON links_to (finish);
                CREATE INDEX links_to_start_finish_idx ON links_to (start, finish);
                """
            )
            self._conn.execute("ANALYZE links_to;")
        except Exception as e:
            raise BenchmarkImportError(f"Postgres (SQL) import failed: {e}") from e

        node_count = self._conn.execute("SELECT count(*) FROM subreddits").fetchone()[0]
        edge_count = self._conn.execute("SELECT count(*) FROM links_to").fetchone()[0]

        errors = []
        if node_count != EXPECTED_NODE_COUNT + 9:
            errors.append(f"node count: expected {EXPECTED_NODE_COUNT}, got {node_count}")
        if edge_count != EXPECTED_EDGE_COUNT:
            errors.append(f"edge count: expected {EXPECTED_EDGE_COUNT}, got {edge_count}")
        if errors:
            raise BenchmarkImportError("Postgres (SQL) import validation failed:\n" + "\n".join(errors))
        return node_count, edge_count

    def persist_aggregation(self):
        self._conn.execute(
            """
            INSERT INTO links_to_agg (
                id,
                start,
                finish,
                sentiment,
                link_count
            )
            SELECT
                ROW_NUMBER() OVER (ORDER BY start, finish),
                start,
                finish,
                AVG(sentiment),
                COUNT(*)
            FROM links_to
            GROUP BY start, finish
            ORDER BY start, finish;
            """)

        # indexes on id
        self._conn.execute(
        """
        CREATE INDEX links_to_agg_start_idx
            ON links_to_agg (start);

        CREATE INDEX links_to_agg_finish_idx
            ON links_to_agg (finish);

        CREATE INDEX links_to_agg_start_finish_idx
            ON links_to_agg (start, finish);
        """)

        self._conn.execute(
        """
        WITH degrees AS (
            SELECT
                s.id,
                COALESCE(out_deg.degree, 0)
                + COALESCE(in_deg.degree, 0) AS degree
            FROM subreddits s
            LEFT JOIN (
                SELECT start, COUNT(*) AS degree
                FROM links_to_agg
                GROUP BY start
            ) out_deg ON out_deg.start = s.id
            LEFT JOIN (
                SELECT finish, COUNT(*) AS degree
                FROM links_to_agg
                GROUP BY finish
            ) in_deg ON in_deg.finish = s.id
        )
        UPDATE subreddits s
        SET degree = d.degree
        FROM degrees d
        WHERE s.id = d.id;
        """
        )

        # indexes on properties used in filtering
        self._conn.execute(
        """
        CREATE INDEX links_to_agg_sentiment_idx
            ON links_to_agg (sentiment);

        CREATE INDEX links_to_agg_link_count_idx
            ON links_to_agg (link_count);

        CREATE INDEX subreddits_degree_idx
            ON subreddits (degree);
        """)

        edge_agg_count = self._conn.execute(
            "SELECT count(*) FROM links_to_agg"
        ).fetchone()[0]
        if edge_agg_count != EXPECTED_EDGE_AGG_COUNT:
            raise BenchmarkImportError(
                f"Postgres (SQL) aggregation validation failed: expected {EXPECTED_EDGE_AGG_COUNT}, got {edge_agg_count}"
            )

    def aggregate_graph(self):
        def run():
            return self._conn.execute(
                """
                SELECT
                    s.subreddit AS source,
                    t.subreddit AS target,
                    AVG(l.sentiment) AS sentiment,
                    COUNT(*) AS link_count
                FROM links_to l
                JOIN subreddits s ON s.id = l.start
                JOIN subreddits t ON t.id = l.finish
                GROUP BY
                    s.id,
                    s.subreddit,
                    t.id,
                    t.subreddit
                ORDER BY sentiment DESC;
            """).fetchall()

        return _timed_repeated(run, n=5)

    def adamic_adar(self, subreddit_names: list[str]):
        # First, get the IDs for the provided subreddit names
        def run(name: str):
            return self._conn.execute(
                """
                SELECT
                    nf.subreddit AS suggested_subreddit,
                    COUNT(DISTINCT common.id) AS common_neighbors_count,
                    AVG(ABS(r1.sentiment - r2.sentiment)) AS avg_delta_sentiment,
                    SUM(
                        1.0 / LN(common.degree + 2)
                    ) AS adamic_adar_score
                FROM subreddits s

                JOIN links_to_agg r1
                    ON r1.start = s.id
                JOIN subreddits common
                    ON common.id = r1.finish

                JOIN links_to_agg r2
                    ON common.id = r2.finish
                JOIN subreddits nf
                    ON r2.start = nf.id

                WHERE s.subreddit = %s
                AND r1.sentiment > %s
                AND r2.sentiment > %s
                AND common.degree < %s
                AND nf.id <> s.id

                AND NOT EXISTS (
                    SELECT 1
                    FROM links_to_agg existing
                    WHERE (existing.start = s.id AND existing.finish = nf.id)
                        OR (existing.finish = s.id AND existing.start = nf.id)
                )

                GROUP BY nf.id, nf.subreddit

                HAVING COUNT(DISTINCT common.id) >= 3

                ORDER BY
                    adamic_adar_score DESC;
                """,
                (name, ADAMIC_AGAR_MIN_SENTIMENT, ADAMIC_AGAR_MIN_SENTIMENT, P99_DEGREE),
            ).fetchall()

        return _timed_per_input(run, subreddit_names)

    def cycle_detection(self, subreddit_names: list[str]):
        sentiment = CYCLE_DETECTION_SENTIMENT

        def run(name: str):
            return self._conn.execute(
                """
                SELECT
                    s.subreddit AS source,
                    a.subreddit AS node_a,
                    b.subreddit AS node_b,
                    r1.sentiment + r2.sentiment + r3.sentiment AS total_sentiment
                FROM subreddits s

                JOIN links_to_agg r1
                    ON r1.start = s.id
                JOIN subreddits a
                    ON a.id = r1.finish

                JOIN links_to_agg r2
                    ON r2.start = a.id
                JOIN subreddits b
                    ON b.id = r2.finish

                JOIN links_to_agg r3
                    ON r3.start = b.id
                AND r3.finish = s.id

                -- reverse s ← a
                JOIN links_to_agg rev1
                    ON rev1.start = a.id
                AND rev1.finish = s.id

                -- reverse a ← b
                JOIN links_to_agg rev2
                    ON rev2.start = b.id
                AND rev2.finish = a.id

                -- reverse s ← b
                JOIN links_to_agg rev3
                    ON rev3.start = s.id
                AND rev3.finish = b.id

                WHERE s.subreddit = %s
                AND r1.sentiment > %s
                AND r2.sentiment > %s
                AND r3.sentiment > %s
                AND rev1.sentiment > %s
                AND rev2.sentiment > %s
                AND rev3.sentiment > %s
                AND a.id <> b.id
                AND a.subreddit < b.subreddit

                ORDER BY total_sentiment DESC;
            """,
                (name, sentiment, sentiment, sentiment, sentiment, sentiment, sentiment,),
            ).fetchall()

        return _timed_per_input(run, subreddit_names)

    def friends_of_friends(self, subreddit_names: list[str]):
        def run(name: str):
            return self._conn.execute("""
                WITH RECURSIVE traversal (origin_id, node, hop_distance, sentiments) AS (
                    SELECT s.id, l.finish, 1, ARRAY[l.sentiment]
                        FROM subreddits s
                        JOIN links_to_agg l ON l.start = s.id
                        JOIN subreddits target ON target.id = l.finish
                        WHERE s.subreddit = %s
                        AND l.sentiment > %s
                        AND l.link_count > %s
                        AND target.degree < %s

                    UNION ALL

                    SELECT t.origin_id, l.finish, t.hop_distance + 1, t.sentiments || l.sentiment
                        FROM traversal t
                        JOIN links_to_agg l ON l.start = t.node
                        JOIN subreddits target ON target.id = l.finish
                        WHERE t.hop_distance < %s
                        AND target.id != t.origin_id 
                        AND l.sentiment > %s
                        AND l.link_count > %s
                        AND target.degree < %s
                )  CYCLE node SET is_cycle USING path
                SELECT
                    subs.subreddit AS reached_subreddit,
                    t.hop_distance,
                    (
                        SELECT AVG(x)
                        FROM unnest(t.sentiments) AS x
                    ) AS avg_path_sentiment
                FROM traversal t
                JOIN subreddits subs
                    ON subs.id = t.node
                WHERE NOT t.is_cycle
                ORDER BY
                    t.hop_distance ASC,
                    avg_path_sentiment DESC;
            """, (name, FRIENDS_OF_FRIENDS_SENTIMENT, MIN_LINKS_AGGREGATED, P99_DEGREE, TRAVERSAL_LIMIT, FRIENDS_OF_FRIENDS_SENTIMENT, MIN_LINKS_AGGREGATED, P99_DEGREE)).fetchall()
        return _timed_per_input(run, subreddit_names)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        return False


def wait_postgres_sql_ready(port: int, timeout: int = 60):
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
    raise TimeoutError(f"postgres (sql) not ready on port {port}") from last_err
