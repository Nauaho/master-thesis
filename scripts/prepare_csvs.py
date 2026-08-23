import csv
from datetime import datetime

def convert(input_tsv: str, output_csv: str):
    with open(input_tsv) as fin, open(output_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=[
            "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP", "LINK_SENTIMENT", "PROPERTIES"
        ])
        writer.writeheader()
        for row in reader:
            writer.writerow({
                "SOURCE_SUBREDDIT": row["SOURCE_SUBREDDIT"],
                "TARGET_SUBREDDIT": row["TARGET_SUBREDDIT"],
                "POST_ID": row["POST_ID"],
                "TIMESTAMP": int(datetime.strptime(row["TIMESTAMP"], "%Y-%m-%d %H:%M:%S").timestamp()),
                "LINK_SENTIMENT": row["LINK_SENTIMENT"],
                "PROPERTIES": [float(num) for num in row["PROPERTIES"].split(",")]
            })

if __name__ == "__main__":
    convert("data/soc-redditHyperlinks-body.tsv", "data/age_body.csv")
    convert("data/soc-redditHyperlinks-title.tsv", "data/age_title.csv")