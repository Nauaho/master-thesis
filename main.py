from setup import run_all_benchmarks, setup
from scripts.prepare_csvs import prepare_csvs
from scripts.prepare_age_edges import prepare_age_edges


def main():
    print("Hello from master-thesis!")
    prepare_csvs()
    prepare_age_edges()
    setup()
    run_all_benchmarks()


if __name__ == "__main__":
    main()
