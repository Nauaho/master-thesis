# Code used for Master Thesis
This is my small benchmarking platform that uses Reddit SNAP dataset and test different databases on how well they perform certain set of graph and vector queries.

## Requirements
- Docker
- uv
- 16 GB of RAM
- 4 CPU cores
 Last 2 can be increases/decreased, just got to the `setup.py` and regulate them.

## How to run
All commands are run from the repository root
1) Download all flies from [here](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) and only `web-redditEmbeddings-subreddits.csv` from [here](https://snap.stanford.edu/data/web-RedditEmbeddings.html) 
2) Create folder `data` and unpack those files and put them there
3) `uv run main.py`
4) Wait until complete
5) To see formatted answer -- `uv run scripts/process_benchmarks.py ./results/<insert_db_name>`

## Known issues
- Sometimes the python Docker client refuses to see and connect with Docker Host on local computer. Usually restart, cleaning of the disk helps
- SurrealDB, Apache AGE and `friends of friends` test in FalcorDB crushed my computer repeatedly. Dangerous parts of code were commented out, but be aware. If the problem occurs, you can go and comment out corresponding DBSpec in the `setup.py`.


## License 
It is a part of my Master Thesis at PJAIT, this code does not have any commercial use, only educational and research purposes. Use it on your own risk, you were informed, I am not responsible for any damages.
