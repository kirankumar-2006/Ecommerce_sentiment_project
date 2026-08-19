

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import re
import base64
from io import BytesIO

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "reviews.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model.joblib")

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(REPORTS_DIR, "sentiment_report.html")
PREDICTIONS_PATH = os.path.join(REPORTS_DIR, "predictions.csv")

COLORS = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
STOPWORDS = set("""a an the this that it is was were be been being to of for and or but in on at
with as by from not no very so much just really so""".split())


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def chart_sentiment_distribution(df):
    counts = df["predicted_sentiment"].value_counts()
    order = [s for s in ["positive", "neutral", "negative"] if s in counts.index]
    counts = counts.reindex(order)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[s] for s in counts.index])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                str(int(b.get_height())), ha="center", fontsize=10)
    ax.set_title("Overall Sentiment Distribution")
    ax.set_ylabel("Number of Reviews")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_trend_over_time(df):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    grouped = d.groupby(["month", "predicted_sentiment"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(grouped_pct.index, grouped_pct["positive"], grouped_pct["neutral"], grouped_pct["negative"],
                 colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]],
                 labels=["Positive", "Neutral", "Negative"])
    ax.set_title("Sentiment Mix Over Time (% of monthly reviews)")
    ax.set_ylabel("% of reviews")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_category_breakdown(df):
    grouped = df.groupby(["category", "predicted_sentiment"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    pct = pct.sort_values("negative", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(pct))
    for sentiment in ["positive", "neutral", "negative"]:
        ax.barh(pct.index, pct[sentiment], left=bottom, color=COLORS[sentiment], label=sentiment.capitalize())
        bottom += pct[sentiment].values
    ax.set_title("Sentiment Breakdown by Product Category (sorted by negative %)")
    ax.set_xlabel("% of reviews")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_rating_vs_sentiment(df):
    grouped = df.groupby(["rating", "predicted_sentiment"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    grouped.plot(kind="bar", stacked=True, ax=ax,
                 color=[COLORS[c] for c in grouped.columns])
    ax.set_title("Star Rating vs Predicted Sentiment")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.legend(title=None, loc="upper left")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_top_keywords(df, sentiment="negative", top_n=12):
    texts = df.loc[df["predicted_sentiment"] == sentiment, "review_text"].str.lower()
    words = Counter()
    for t in texts:
        for w in re.findall(r"[a-z']+", t):
            if w not in STOPWORDS and len(w) > 2:
                words[w] += 1
    top = words.most_common(top_n)
    if not top:
        return None
    labels, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(labels[::-1], counts[::-1], color=COLORS[sentiment])
    ax.set_title(f"Top Keywords in {sentiment.capitalize()} Reviews")
    ax.set_xlabel("Frequency")
    fig.tight_layout()
    return fig_to_base64(fig)


def build_html_report(df, charts, metrics_text):
    total = len(df)
    counts = df["predicted_sentiment"].value_counts()
    pos_pct = round(100 * counts.get("positive", 0) / total, 1)
    neg_pct = round(100 * counts.get("negative", 0) / total, 1)
    neu_pct = round(100 * counts.get("neutral", 0) / total, 1)
    avg_rating = round(df["rating"].mean(), 2)

    worst_category = (
        df.groupby("category")
        .apply(lambda g: (g["predicted_sentiment"] == "negative").mean(), include_groups=False)
        .sort_values(ascending=False)
        .index[0]
    )

    # A sample of the most negative flagged reviews for quick triage
    flagged = df[df["predicted_sentiment"] == "negative"].sample(
        min(8, (df["predicted_sentiment"] == "negative").sum()), random_state=1
    )[["date", "category", "product", "rating", "review_text"]]

    flagged_rows = "".join(
        f"""<tr>
                <td>{r.date}</td>
                <td>{r.category}</td>
                <td>{r.product}</td>
                <td>{'★' * int(r.rating)}{'☆' * (5 - int(r.rating))}</td>
                <td>{r.review_text}</td>
            </tr>"""
        for r in flagged.itertuples()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Customer Sentiment Analysis Report</title>
<style>
  :root {{
    --pos: {COLORS['positive']}; --neg: {COLORS['negative']}; --neu: {COLORS['neutral']};
    --ink: #1f2937; --muted: #6b7280; --bg: #f7f8fa; --card: #ffffff; --border: #e5e7eb;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--ink); margin: 0; padding: 32px;
  }}
  .container {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-top: 0; margin-bottom: 28px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 32px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; text-align: center; }}
  .kpi .value {{ font-size: 1.6rem; font-weight: 700; }}
  .kpi .label {{ color: var(--muted); font-size: 0.8rem; margin-top: 4px; }}
  .pos {{ color: var(--pos); }} .neg {{ color: var(--neg); }} .neu {{ color: var(--neu); }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 28px; }}
  .card h2 {{ font-size: 1.1rem; margin-top: 0; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  img {{ max-width: 100%; border-radius: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; }}
  pre {{ background: #f3f4f6; padding: 14px; border-radius: 8px; font-size: 0.8rem; overflow-x: auto; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
  .badge.neg {{ background: #fdecea; color: var(--neg); }}
  footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 32px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Customer Sentiment Analysis &amp; Reporting</h1>
  <p class="subtitle">E-commerce reviews · {total:,} reviews analyzed · ML-based classification (TF-IDF + Logistic Regression)</p>

  <div class="kpi-grid">
    <div class="kpi"><div class="value">{total:,}</div><div class="label">Total Reviews</div></div>
    <div class="kpi"><div class="value pos">{pos_pct}%</div><div class="label">Positive</div></div>
    <div class="kpi"><div class="value neu">{neu_pct}%</div><div class="label">Neutral</div></div>
    <div class="kpi"><div class="value neg">{neg_pct}%</div><div class="label">Negative</div></div>
    <div class="kpi"><div class="value">{avg_rating}★</div><div class="label">Avg. Star Rating</div></div>
  </div>

  <div class="card">
    <h2>Overall Sentiment Distribution</h2>
    <img src="data:image/png;base64,{charts['dist']}" alt="Sentiment distribution">
  </div>

  <div class="card">
    <h2>Sentiment Trend Over Time</h2>
    <img src="data:image/png;base64,{charts['trend']}" alt="Sentiment trend">
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>By Product Category</h2>
      <p style="color:var(--muted); font-size:0.85rem;">Highest negative share: <b>{worst_category}</b></p>
      <img src="data:image/png;base64,{charts['category']}" alt="Category breakdown">
    </div>
    <div class="card">
      <h2>Star Rating vs. Predicted Sentiment</h2>
      <img src="data:image/png;base64,{charts['rating']}" alt="Rating vs sentiment">
    </div>
  </div>

  <div class="card">
    <h2>What Customers Complain About Most</h2>
    <img src="data:image/png;base64,{charts['keywords']}" alt="Top negative keywords">
  </div>

  <div class="card">
    <h2>Flagged Reviews for Triage <span class="badge neg">Negative</span></h2>
    <table>
      <tr><th>Date</th><th>Category</th><th>Product</th><th>Rating</th><th>Review</th></tr>
      {flagged_rows}
    </table>
  </div>

  <div class="card">
    <h2>Model Performance (held-out test set)</h2>
    <pre>{metrics_text}</pre>
  </div>

  <footer>Generated automatically by the Customer Sentiment Analysis pipeline.</footer>
</div>
</body>
</html>"""
    return html



def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Load trained model
    pipeline = joblib.load(MODEL_PATH)

    # Predict sentiments
    df["predicted_sentiment"] = pipeline.predict(df["review_text"])
    proba = pipeline.predict_proba(df["review_text"])
    df["confidence"] = proba.max(axis=1).round(3)

    # Ensure reports folder exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Save predictions
    pred_path = os.path.join(REPORTS_DIR, "predictions.csv")
    df.to_csv(pred_path, index=False)
    print(f"Predictions saved -> {pred_path}")

    # Generate charts
    charts = {
        "dist": chart_sentiment_distribution(df),
        "trend": chart_trend_over_time(df),
        "category": chart_category_breakdown(df),
        "rating": chart_rating_vs_sentiment(df),
        "keywords": chart_top_keywords(df, "negative"),
    }

    # Read model metrics
    metrics_path = os.path.join(REPORTS_DIR, "metrics.txt")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_text = f.read()
    else:
        metrics_text = "Metrics file not found."

    # Build HTML report
    html = build_html_report(df, charts, metrics_text)

    # Save HTML report using UTF-8
    html_path = os.path.join(REPORTS_DIR, "sentiment_report.html")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report saved -> {html_path}")

if __name__ == "__main__":
    main()
