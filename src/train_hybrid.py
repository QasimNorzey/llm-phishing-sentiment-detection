from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.common import evaluate_predictions, plot_confusion, save_json
from src.data_utils import load_dataset
from src.features import build_feature_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TF-IDF + persuasion/sentiment hybrid phishing model.")
    parser.add_argument("--data", default="data/demo_emails.csv", help="Path to CSV with columns text,label")
    parser.add_argument("--outdir", default="results/hybrid", help="Directory for outputs")
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
    X_train_text = vectorizer.fit_transform(X_train)
    X_test_text = vectorizer.transform(X_test)

    train_features = build_feature_frame(X_train)
    test_features = build_feature_frame(X_test)
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_features)
    X_test_num = scaler.transform(test_features)

    X_train_full = hstack([X_train_text, csr_matrix(X_train_num)], format="csr")
    X_test_full = hstack([X_test_text, csr_matrix(X_test_num)], format="csr")

    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(X_train_full, y_train)
    predictions = model.predict(X_test_full)

    metrics = evaluate_predictions(y_test, predictions)
    save_json(metrics, outdir / "metrics.json")
    (outdir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")
    plot_confusion(metrics["confusion_matrix"], outdir / "confusion_matrix.png", "Hybrid confusion matrix")

    vocab = vectorizer.get_feature_names_out().tolist()
    coeffs = model.coef_[0]
    text_weights = pd.DataFrame({"feature": vocab, "weight": coeffs[: len(vocab)]}).sort_values("weight", ascending=False)
    text_weights.head(20).to_csv(outdir / "top_phishing_terms.csv", index=False)
    text_weights.tail(20).sort_values("weight").to_csv(outdir / "top_benign_terms.csv", index=False)

    numeric_columns = train_features.columns.tolist()
    numeric_weights = pd.DataFrame(
        {"feature": numeric_columns, "weight": coeffs[len(vocab): len(vocab) + len(numeric_columns)]}
    ).sort_values("weight", ascending=False)
    numeric_weights.to_csv(outdir / "numeric_feature_weights.csv", index=False)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "scaler": scaler,
            "model": model,
            "numeric_columns": numeric_columns,
        },
        outdir / "hybrid_model.joblib",
    )

    print("Hybrid metrics")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
