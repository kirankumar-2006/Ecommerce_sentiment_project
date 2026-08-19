

import argparse
import sys
import joblib
import pandas as pd

MODEL_PATH = "/home/claude/ecommerce_sentiment_project/models/sentiment_model.joblib"


def score_texts(texts):
    pipeline = joblib.load(MODEL_PATH)
    preds = pipeline.predict(texts)
    proba = pipeline.predict_proba(texts)
    confidences = proba.max(axis=1)
    return list(zip(texts, preds, confidences))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="A single review text to score")
    parser.add_argument("--file", help="CSV file with reviews to batch-score")
    parser.add_argument("--column", default="review_text", help="Column name containing review text")
    args = parser.parse_args()

    if args.file:
        df = pd.read_csv(args.file)
        results = score_texts(df[args.column].astype(str).tolist())
        for text, pred, conf in results:
            print(f"[{pred:8s} {conf:.2f}] {text[:80]}")
    elif args.text:
        results = score_texts([args.text])
        text, pred, conf = results[0]
        print(f"Sentiment: {pred}  (confidence: {conf:.2%})")
    else:
        print("Provide either a review text or --file path/to.csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
