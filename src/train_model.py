

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "reviews.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model.joblib")

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

CM_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.png")


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # guards against class imbalance
            C=3.0,
        )),
    ])


def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Sentiment Classifier")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df["review_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"Test Accuracy: {acc:.3f}\n")
    print(report)

    labels = sorted(y.unique())
    plot_confusion_matrix(y_test, y_pred, labels, CM_PATH)
    print(f"Confusion matrix saved -> {CM_PATH}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    # Save a small metrics summary for the report generator to reuse
    metrics_path = os.path.join(REPORTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test Accuracy: {acc:.3f}\n\n")
        f.write(report)
    print(f"Metrics summary saved -> {metrics_path}")


if __name__ == "__main__":
    main()
