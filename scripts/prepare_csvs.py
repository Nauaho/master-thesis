# scripts/prepare_age_csvs.py
import csv
from datetime import datetime
def convert(input_tsv: str, output_csv: str):
    with open(input_tsv) as fin, open(output_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=[
            "id", "start_id", "end_id",
            "start_vertex_type", "end_vertex_type",
            "postId", "timestamp", "sentimentScore", "properties"
        ])
        writer.writeheader()
        for i, row in enumerate(reader, start=1):
            writer.writerow({
                "id": i,
                "start_id": row["SOURCE_SUBREDDIT"],
                "end_id": row["TARGET_SUBREDDIT"],
                "start_vertex_type": "Subreddit",
                "end_vertex_type": "Subreddit",
                "postId": row["POST_ID"],
                "timestamp": int(datetime.strptime(row["TIMESTAMP"], "%Y-%m-%d %H:%M:%S").timestamp()),
                "sentimentScore": row["LINK_SENTIMENT"],
                "properties": [float(num) for num in row["PROPERTIES"].split(",")]
            })

if __name__ == "__main__":
    convert("data/soc-redditHyperlinks-body.tsv", "data/age_body.csv")
    convert("data/soc-redditHyperlinks-title.tsv", "data/age_title.csv")