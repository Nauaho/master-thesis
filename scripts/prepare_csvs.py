import csv
from pathlib import Path


INPUT_FILES = [
    Path("data/soc-redditHyperlinks-body.tsv"),
    Path("data/soc-redditHyperlinks-title.tsv"),
]

OUTPUT_DIR = Path("data/reddit_agefreighter")

NODES_FILE = OUTPUT_DIR / "subreddits.csv"
EDGES_FILE = OUTPUT_DIR / "links.csv"


def collect_subreddits() -> set[str]:
    """First pass: collect every unique subreddit name."""
    subreddits: set[str] = set()

    for path in INPUT_FILES:
        print(f"Scanning {path}...")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            required_columns = {
                "SOURCE_SUBREDDIT",
                "TARGET_SUBREDDIT",
                "LINK_SENTIMENT",
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


def write_nodes(subreddits: set[str]) -> dict[str, int]:
    """
    Assign deterministic numeric IDs to subreddits and write nodes.csv.

    Example:
        1,leagueoflegends
        2,teamredditteams
        3,gamedev
    """
    # Sorting gives us deterministic IDs across runs.
    ordered = sorted(subreddits)

    subreddit_to_id = {
        name: index
        for index, name in enumerate(ordered, start=1)
    }

    with NODES_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["id", "name"])

        for name in ordered:
            writer.writerow([
                subreddit_to_id[name],
                name,
            ])

    return subreddit_to_id


def write_edges(subreddit_to_id: dict[str, int]) -> int:
    """
    Combine body + title TSVs into one AGEFreighter edge CSV.

    Every source row becomes exactly one graph edge.
    """
    edge_id = 1

    with EDGES_FILE.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)

        writer.writerow([
            "id",
            "start_id",
            "start_vertex_type",
            "end_id",
            "end_vertex_type",
            "sentimentScore",
        ])

        for path in INPUT_FILES:
            print(f"Converting {path}...")

            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")

                for row in reader:
                    source = row["SOURCE_SUBREDDIT"]
                    target = row["TARGET_SUBREDDIT"]

                    try:
                        sentiment = float(row["LINK_SENTIMENT"])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid LINK_SENTIMENT in {path}: {row}"
                        ) from exc

                    writer.writerow([
                        edge_id,
                        subreddit_to_id[source],
                        "Subreddit",
                        subreddit_to_id[target],
                        "Subreddit",
                        sentiment,
                    ])

                    edge_id += 1

    return edge_id - 1


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Reddit → AGEFreighter preprocessing ===")
    print()

    # Pass 1: find all vertices.
    subreddits = collect_subreddits()

    print()
    print(f"Unique subreddits: {len(subreddits):,}")

    # Create numeric IDs and nodes.csv.
    subreddit_to_id = write_nodes(subreddits)

    print(f"Wrote: {NODES_FILE}")

    # Pass 2: convert both edge files.
    edge_count = write_edges(subreddit_to_id)

    print(f"Wrote: {EDGES_FILE}")
    print()
    print("=== Done ===")
    print(f"Nodes: {len(subreddits):,}")
    print(f"Edges: {edge_count:,}")


if __name__ == "__main__":
    main()