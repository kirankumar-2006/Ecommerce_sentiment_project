# Customer Sentiment Analysis & Reporting for E-commerce

An end-to-end machine learning project that classifies e-commerce customer
reviews as **positive**, **neutral**, or **negative**, and turns the
results into an executive-ready HTML analytics report.

## What it does

1. **Data** — synthetic (but realistic) dataset of 3,000 e-commerce
   reviews across 8 product categories, with dates, star ratings, and
   ground-truth sentiment labels. Swap `data/reviews.csv` with your own
   real review export (same columns) to use real data instead.
2. **Model** — a `TF-IDF (unigrams + bigrams) → Logistic Regression`
   scikit-learn pipeline, trained with class balancing to handle the
   natural skew of review data (most reviews are positive). ~98% held-out
   accuracy on this dataset.
3. **Reporting** — scores every review, then generates:
   - `reports/predictions.csv` — every review + predicted sentiment + confidence
   - `reports/confusion_matrix.png` — model evaluation
   - `reports/sentiment_report.html` — a single-file dashboard with KPIs,
     sentiment trend over time, category breakdowns, rating-vs-sentiment
     analysis, top complaint keywords, and a flagged-reviews triage table

## Project structure

```
ecommerce_sentiment_project/
├── main.py                     # orchestrates the full pipeline
├── requirements.txt
├── data/
│   └── reviews.csv             # generated review dataset
├── models/
│   └── sentiment_model.joblib  # trained ML pipeline
├── reports/
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── metrics.txt
│   └── sentiment_report.html   # <- open this in a browser
└── src/
    ├── generate_data.py        # synthetic data generator
    ├── train_model.py          # trains + evaluates the classifier
    ├── generate_report.py      # builds charts + HTML dashboard
    └── predict_review.py       # score new/unseen review text
```

## Running it

```bash
pip install -r requirements.txt

# Full pipeline (regenerates data, trains, reports)
python3 main.py

# Reuse existing data, just retrain + re-report
python3 main.py --skip-data

# Score a brand-new review with the trained model
python3 src/predict_review.py "Arrived broken and support never replied."

# Batch-score a new CSV of reviews
python3 src/predict_review.py --file new_reviews.csv --column review_text
```

## Swapping in real data

Replace `data/reviews.csv` with your own export containing these columns:

| column      | description                              |
|-------------|-------------------------------------------|
| review_id   | unique identifier                          |
| date        | `YYYY-MM-DD`                               |
| category    | product category                           |
| product     | product name                               |
| review_text | the raw customer review text               |
| rating      | 1–5 star rating                            |
| sentiment   | ground truth label (only needed for training) |

Then run `python3 src/train_model.py` followed by
`python3 src/generate_report.py`. If you don't have labeled data, you can
bootstrap labels from star ratings (1–2 = negative, 3 = neutral, 4–5 =
positive) before training.

## Extending this project

- Swap `LogisticRegression` for `LinearSVC`, `RandomForest`, or a
  transformer-based model (e.g. `distilbert-base-uncased-finetuned-sst-2`
  via HuggingFace) inside `train_model.py` — the rest of the pipeline is
  model-agnostic.
- Add aspect-based sentiment (e.g. sentiment toward "delivery" vs
  "product quality" specifically) by extracting noun phrases before
  classification.
- Schedule `main.py` to run nightly against a live reviews database and
  email/Slack the HTML report automatically.
