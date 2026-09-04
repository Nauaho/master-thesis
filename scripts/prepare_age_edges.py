import csv
from pathlib import Path


DATA_DIR = Path("data") / "normalised_csvs"

SOURCE_EDGES_FILE = DATA_DIR / "links.csv"
AGE_EDGES_FILE = DATA_DIR / "links_age.csv"


def prepare_age_edges() -> int:
    """
    Derive an AGE-compatible edge CSV from the canonical links.csv.

    AGE's load_edges_from_file() expects:

        start_id,start_vertex_type,end_id,end_vertex_type,<properties...>

    with no leading id column, unlike the canonical links.csv (which
    keeps an id column for the other engines). This step strips that
    column without touching the shared preprocessing output.
    """
    edge_count = 0

    print(f"Reading {SOURCE_EDGES_FILE}...")

    with SOURCE_EDGES_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as src, AGE_EDGES_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)

        required_columns = {
            "start_id",
            "start_vertex_type",
            "end_id",
            "end_vertex_type",
            "sentimentScore",
            "timestamp",
            "properties",
            "post_id",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{SOURCE_EDGES_FILE} is missing required columns. "
                f"Found: {reader.fieldnames}"
            )

        writer = csv.writer(dst)

        writer.writerow([
            "start_id",
            "start_vertex_type",
            "end_id",
            "end_vertex_type",
            "sentimentScore",
            "timestamp",
            "properties",
            "post_id",
        ])

        for row in reader:
            writer.writerow([
                row["start_id"],
                row["start_vertex_type"],
                row["end_id"],
                row["end_vertex_type"],
                row["sentimentScore"],
                row["timestamp"],
                row["properties"],
                row["post_id"],
            ])
            edge_count += 1

    print(f"Wrote {AGE_EDGES_FILE} ({edge_count:,} edges)")

    return edge_count


if __name__ == "__main__":
    prepare_age_edges()