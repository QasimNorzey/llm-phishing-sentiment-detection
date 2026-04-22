from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

POSITIVE_LABELS = {1, "1", "spam", "phishing", "phish", "malicious", "fraud"}
NEGATIVE_LABELS = {0, "0", "ham", "legitimate", "legit", "benign", "safe"}


def normalize_label(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Label contains NaN")
    text = str(value).strip().lower()
    if value in POSITIVE_LABELS or text in POSITIVE_LABELS:
        return 1
    if value in NEGATIVE_LABELS or text in NEGATIVE_LABELS:
        return 0
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {value!r}") from exc
    if number in (0, 1):
        return number
    raise ValueError(f"Unsupported numeric label: {value!r}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {sorted(required)}; missing: {sorted(missing)}")
    data = df[["text", "label"]].copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() > 0].reset_index(drop=True)
    data["label"] = data["label"].apply(normalize_label)
    return data
