import csv
from pathlib import Path


DATA_DIR = Path("data")

INPUT_FILES = [
    DATA_DIR / "soc-redditHyperlinks-body.tsv",
    DATA_DIR / "soc-redditHyperlinks-title.tsv",
]

EMBEDDINGS_FILE = DATA_DIR / "web-redditEmbeddings-subreddits.csv"

OUTPUT_DIR = DATA_DIR / "normalised_csvs"

NODES_LIGHT_FILE = OUTPUT_DIR / "subreddits_light.csv"
NODES_FULL_FILE = OUTPUT_DIR / "subreddits_full.csv"
EDGES_FILE = OUTPUT_DIR / "links.csv"

EMBEDDING_DIMENSION = 300


def collect_link_subreddits() -> set[str]:
    """
    Collect every unique subreddit appearing in the hyperlink datasets.

    A subreddit is included when it appears as either the source or
    target of at least one link in any of the input TSV files.

    The resulting set represents the complete set of subreddits
    referenced by the graph edges.
    """
    subreddits: set[str] = set()

    for path in INPUT_FILES:
        print(f"Scanning {path}...")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            required_columns = {
                "SOURCE_SUBREDDIT",
                "TARGET_SUBREDDIT",
            }

            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{path} is missing required columns. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                source = row["SOURCE_SUBREDDIT"]
                target = row["TARGET_SUBREDDIT"]

                if not source or not target:
                    raise ValueError(
                        f"Empty subreddit name in {path}: {row}"
                    )

                subreddits.add(source)
                subreddits.add(target)

    return subreddits


def read_embeddings() -> dict[str, list[float]]:
    """
    Read subreddit embeddings from the embedding CSV.

    The embedding file is expected to contain one subreddit per row,
    with the subreddit name in the first column followed by exactly
    300 floating-point values.

    The embedding dataset is independent from the hyperlink dataset:
    it may contain subreddits that never occur in the hyperlink files.

    Rows with invalid embeddings are skipped.

    Returns:
        A mapping from subreddit name to its 300-dimensional embedding.
    """
    embeddings: dict[str, list[float]] = {}

    print(f"Reading embeddings from {EMBEDDINGS_FILE}...")

    with EMBEDDINGS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.reader(f)

        for line_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue

            name = row[0].strip()

            if not name:
                continue

            try:
                embedding = [float(value) for value in row[1:]]
            except ValueError:
                # This also naturally skips a header row, if present.
                continue

            if len(embedding) != EMBEDDING_DIMENSION:
                print(
                    f"Warning: skipping embedding for {name!r} "
                    f"on line {line_number}: "
                    f"expected {EMBEDDING_DIMENSION} dimensions, "
                    f"got {len(embedding)}"
                )
                continue

            # NaN is not a valid embedding component.
            if any(value != value for value in embedding):
                print(
                    f"Warning: skipping embedding for {name!r}: "
                    f"contains NaN"
                )
                continue

            embeddings[name] = embedding

    return embeddings


def build_subreddit_ids(
    link_subreddits: set[str],
    embedding_subreddits: set[str],
) -> dict[str, int]:
    """
    Build deterministic numeric IDs for the complete subreddit set.

    The complete set is the UNION of:

        link_subreddits ∪ embedding_subreddits

    Therefore, a subreddit is included even if it appears only in the
    embedding dataset and has no corresponding graph edges.

    Subreddit names are sorted before assigning IDs so that the same
    input datasets always produce the same IDs.
    """
    all_subreddits = link_subreddits | embedding_subreddits

    ordered = sorted(all_subreddits)

    return {
        name: index
        for index, name in enumerate(ordered, start=1)
    }


def write_nodes(
    subreddit_to_id: dict[str, int],
    embeddings: dict[str, list[float]],
) -> None:
    """
    Write the two normalized subreddit datasets.

    subreddits_light.csv contains the complete UNION of all subreddits
    from both datasets and contains only:

        id,name

    subreddits_full.csv contains the same complete UNION and contains:

        id,name,embedding

    The embedding field contains a 300-dimensional array of floats
    when an embedding exists for that subreddit. If no embedding is
    available, the field is left empty.

    Consequently, subreddits_full is not a subset of
    subreddits_light based on embedding coverage. Both files contain
    exactly the same set of subreddit IDs and names.
    """

    print(f"Writing {NODES_LIGHT_FILE}...")

    with NODES_LIGHT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "name",
        ])

        for name, subreddit_id in subreddit_to_id.items():
            writer.writerow([
                subreddit_id,
                name,
            ])

    print(f"Writing {NODES_FULL_FILE}...")

    with NODES_FULL_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "name",
            "embedding",
        ])

        for name, subreddit_id in subreddit_to_id.items():
            embedding = embeddings.get(name)

            if embedding is None:
                embedding_value = ""
            else:
                embedding_value = (
                    "["
                    + ",".join(
                        str(value)
                        for value in embedding
                    )
                    + "]"
                )

            writer.writerow([
                subreddit_id,
                name,
                embedding_value,
            ])


def write_edges(
    subreddit_to_id: dict[str, int],
) -> int:
    """
    Combine the body and title hyperlink TSV files into links.csv.

    Every row in the original hyperlink datasets becomes exactly one
    normalized graph edge.

    The resulting CSV contains:

        id
        start_id
        start_vertex_type
        end_id
        end_vertex_type
        sentimentScore
        timestamp
        properties
        post_id

    The source and target subreddit names are replaced by their
    deterministic numeric IDs.

    LINK_SENTIMENT is converted to a floating-point value.

    TIMESTAMP is preserved in its original format:

        YYYY-MM-DD HH:MM:SS

    PROPERTIES is converted from the original comma-separated string
    of numbers into an array representation:

        "0.1,0.2,0.3"

    becomes:

        "[0.1,0.2,0.3]"

    No aggregation is performed here. Every original hyperlink
    remains a separate edge.
    """
    edge_id = 1

    print(f"Writing {EDGES_FILE}...")

    with EDGES_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as out:
        writer = csv.writer(out)

        writer.writerow([
            "id",
            "start_id",
            "start_vertex_type",
            "end_id",
            "end_vertex_type",
            "sentimentScore",
            "timestamp",
            "properties",
            "post_id",
        ])

        for path in INPUT_FILES:
            print(f"Converting {path}...")

            with path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                )

                required_columns = {
                    "SOURCE_SUBREDDIT",
                    "TARGET_SUBREDDIT",
                    "POST_ID",
                    "TIMESTAMP",
                    "LINK_SENTIMENT",
                    "PROPERTIES",
                }

                if not required_columns.issubset(
                    reader.fieldnames or []
                ):
                    raise ValueError(
                        f"{path} is missing required columns. "
                        f"Found: {reader.fieldnames}"
                    )

                for row in reader:
                    source = row["SOURCE_SUBREDDIT"]
                    target = row["TARGET_SUBREDDIT"]
                    post_id = row["POST_ID"]

                    if source not in subreddit_to_id:
                        raise ValueError(
                            f"Source subreddit {source!r} "
                            f"was not found in subreddit ID mapping"
                        )

                    if target not in subreddit_to_id:
                        raise ValueError(
                            f"Target subreddit {target!r} "
                            f"was not found in subreddit ID mapping"
                        )

                    # --------------------------------------------------
                    # Sentiment
                    # --------------------------------------------------

                    try:
                        sentiment = float(
                            row["LINK_SENTIMENT"]
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid LINK_SENTIMENT in "
                            f"{path}: {row}"
                        ) from exc

                    # --------------------------------------------------
                    # Timestamp
                    # --------------------------------------------------

                    timestamp = row["TIMESTAMP"].strip()

                    if not timestamp:
                        raise ValueError(
                            f"Empty TIMESTAMP in {path}: {row}"
                        )

                    # Keep the original:
                    #
                    # 2013-12-31 16:39:58
                    #
                    # rather than converting it to ISO format here.
                    # The SurrealDB import layer can interpret it as
                    # a datetime.
                    # --------------------------------------------------

                    # --------------------------------------------------
                    # Properties
                    # --------------------------------------------------

                    properties_raw = row["PROPERTIES"].strip()

                    if properties_raw:
                        try:
                            properties = [
                                float(value.strip())
                                for value in properties_raw.split(",")
                            ]
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid PROPERTIES in "
                                f"{path}: {row}"
                            ) from exc

                        properties_value = (
                            "["
                            + ",".join(
                                str(value)
                                for value in properties
                            )
                            + "]"
                        )
                    else:
                        properties_value = "[]"

                    # --------------------------------------------------
                    # Write normalized edge
                    # --------------------------------------------------

                    writer.writerow([
                        edge_id,
                        subreddit_to_id[source],
                        "Subreddit",
                        subreddit_to_id[target],
                        "Subreddit",
                        sentiment,
                        timestamp,
                        properties_value,
                        post_id,
                    ])

                    edge_id += 1

    return edge_id - 1


def prepare_csvs() -> None:
    """
    Preprocess the Reddit hyperlink and embedding datasets.

    The preprocessing consists of five stages:

    1. Collect all unique subreddits from the hyperlink datasets.

    2. Read all valid subreddit embeddings from the embedding dataset.

    3. Construct the complete subreddit set as:

           LINK_SUBREDDITS ∪ EMBEDDING_SUBREDDITS

       This ensures that embedding-only subreddits are not lost.

    4. Assign deterministic numeric IDs and write:

           subreddits_light.csv
           subreddits_full.csv

       Both files contain exactly the same UNION of subreddits.

    5. Convert the original hyperlink TSV files into a single
       normalized links.csv containing numeric vertex IDs,
       sentiment, timestamp, properties arrays, and post IDs.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== Reddit preprocessing ===")
    print()

    # --------------------------------------------------------------
    # 1. Collect subreddits appearing in links.
    # --------------------------------------------------------------

    link_subreddits = collect_link_subreddits()

    print()
    print(
        f"Subreddits from links: "
        f"{len(link_subreddits):,}"
    )

    # --------------------------------------------------------------
    # 2. Read embeddings.
    # --------------------------------------------------------------

    embeddings = read_embeddings()

    embedding_subreddits = set(embeddings)

    print(
        f"Subreddits with embeddings: "
        f"{len(embedding_subreddits):,}"
    )

    # --------------------------------------------------------------
    # 3. Build UNION.
    #
    #     FULL = LINKS ∪ EMBEDDINGS
    # --------------------------------------------------------------

    subreddit_to_id = build_subreddit_ids(
        link_subreddits,
        embedding_subreddits,
    )

    print(
        f"Total unique subreddits (UNION): "
        f"{len(subreddit_to_id):,}"
    )

    overlap = (
        link_subreddits
        & embedding_subreddits
    )

    embedding_only = (
        embedding_subreddits
        - link_subreddits
    )

    link_only = (
        link_subreddits
        - embedding_subreddits
    )

    print(
        f"  Present in both: "
        f"{len(overlap):,}"
    )

    print(
        f"  Link-only: "
        f"{len(link_only):,}"
    )

    print(
        f"  Embedding-only: "
        f"{len(embedding_only):,}"
    )

    # --------------------------------------------------------------
    # 4. Write node files.
    # --------------------------------------------------------------

    write_nodes(
        subreddit_to_id,
        embeddings,
    )

    # --------------------------------------------------------------
    # 5. Write edge file.
    # --------------------------------------------------------------

    edge_count = write_edges(
        subreddit_to_id,
    )

    print()
    print("=== Done ===")
    print(
        f"Nodes in UNION: "
        f"{len(subreddit_to_id):,}"
    )

    print(
        f"Nodes with embeddings: "
        f"{len(embedding_subreddits):,}"
    )

    print(
        f"Edges: "
        f"{edge_count:,}"
    )

    print()
    print(f"Wrote: {NODES_LIGHT_FILE}")
    print(f"Wrote: {NODES_FULL_FILE}")
    print(f"Wrote: {EDGES_FILE}")


if __name__ == "__main__":
    prepare_csvs()