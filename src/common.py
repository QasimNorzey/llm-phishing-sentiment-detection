from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


RESULT_LABELS = ["Benign", "Phishing"]


def evaluate_predictions(y_true, y_pred) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=RESULT_LABELS, zero_division=0),
    }
    return metrics


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def plot_confusion(cm: list[list[int]], path: str | Path, title: str) -> None:
    array = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(array)
    ax.set_xticks([0, 1], labels=RESULT_LABELS)
    ax.set_yticks([0, 1], labels=RESULT_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j, i, str(array[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
