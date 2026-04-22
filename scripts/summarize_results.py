from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    baseline = load_metrics(RESULTS / "baseline" / "metrics.json")
    hybrid = load_metrics(RESULTS / "hybrid" / "metrics.json")

    df = pd.DataFrame(
        [
            {"model": "Baseline TF-IDF + LR", **{k: baseline[k] for k in ("accuracy", "precision", "recall", "f1")}},
            {"model": "Hybrid TF-IDF + Sentiment", **{k: hybrid[k] for k in ("accuracy", "precision", "recall", "f1")}},
        ]
    )
    df.to_csv(RESULTS / "demo_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        ax.plot(df["model"], df[metric], marker="o", label=metric)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Teaching demo benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "demo_metrics.png", dpi=160)
    plt.close(fig)

    summary = (
        "# Demo summary\n\n"
        f"- Baseline F1: {baseline['f1']:.3f}\n"
        f"- Hybrid F1: {hybrid['f1']:.3f}\n"
        f"- Baseline Recall: {baseline['recall']:.3f}\n"
        f"- Hybrid Recall: {hybrid['recall']:.3f}\n"
    )
    (RESULTS / "demo_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
