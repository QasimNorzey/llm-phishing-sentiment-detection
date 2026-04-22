from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.common import evaluate_predictions, plot_confusion, save_json
from src.data_utils import load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TF-IDF + Logistic Regression phishing baseline.")
    parser.add_argument("--data", default="data/demo_emails.csv", help="Path to CSV with columns text,label")
    parser.add_argument("--outdir", default="results/baseline", help="Directory for outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=10000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data["label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=args.max_features,
        strip_accents="unicode",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    metrics = evaluate_predictions(y_test, predictions)
    save_json(metrics, outdir / "metrics.json")
    (outdir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")
    plot_confusion(metrics["confusion_matrix"], outdir / "confusion_matrix.png", "Baseline confusion matrix")

    vocab = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    coeffs = pd.DataFrame({"feature": vocab, "weight": weights}).sort_values("weight", ascending=False)
    coeffs.head(20).to_csv(outdir / "top_phishing_terms.csv", index=False)
    coeffs.tail(20).sort_values("weight").to_csv(outdir / "top_benign_terms.csv", index=False)

    joblib.dump({"vectorizer": vectorizer, "model": model}, outdir / "baseline_model.joblib")

    print("Baseline metrics")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
