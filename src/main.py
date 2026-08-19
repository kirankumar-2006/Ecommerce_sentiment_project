import argparse
import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE)


def run(script):
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print(f"{'='*60}")

    result = subprocess.run([sys.executable, os.path.join(BASE, script)])

    if result.returncode != 0:
        sys.exit(f"Step failed: {script}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip regenerating dataset"
    )

    args = parser.parse_args()

    if not args.skip_data:
        run("generate_data.py")
    else:
        print("Skipping data generation...")

    run("train_model.py")
    run("generate_report.py")

    print("\n==============================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("==============================")

    print(f"Dataset      : {os.path.join(PROJECT_ROOT,'data','reviews.csv')}")
    print(f"Model        : {os.path.join(PROJECT_ROOT,'models','sentiment_model.joblib')}")
    print(f"Report       : {os.path.join(PROJECT_ROOT,'reports','sentiment_report.html')}")
    print(f"Predictions  : {os.path.join(PROJECT_ROOT,'reports','predictions.csv')}")
    print(f"Metrics      : {os.path.join(PROJECT_ROOT,'reports','metrics.txt')}")
    print(f"Confusion Mat: {os.path.join(PROJECT_ROOT,'reports','confusion_matrix.png')}")


if __name__ == "__main__":
    main()