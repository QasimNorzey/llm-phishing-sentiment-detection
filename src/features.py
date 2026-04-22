from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.lexicons import (
    ACTION_VERBS,
    AUTHORITY_WORDS,
    CREDENTIAL_WORDS,
    FINANCIAL_WORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    REWARD_WORDS,
    THREAT_WORDS,
    TRUST_WORDS,
    URGENCY_WORDS,
)

URL_RE = re.compile(r"(https?://\S+|www\.\S+|[a-z0-9.-]+\.(?:com|net|org|edu|io|co|sa|ai|info)\S*)", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _count_overlap(tokens: list[str], vocabulary: set[str]) -> int:
    return sum(1 for token in tokens if token in vocabulary)


def _uppercase_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


def extract_feature_dict(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    token_count = max(len(tokens), 1)
    num_urls = len(URL_RE.findall(text))
    num_emails = len(EMAIL_RE.findall(text))
    num_digits = sum(ch.isdigit() for ch in text)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = _uppercase_ratio(text)

    urgency_count = _count_overlap(tokens, URGENCY_WORDS)
    threat_count = _count_overlap(tokens, THREAT_WORDS)
    authority_count = _count_overlap(tokens, AUTHORITY_WORDS)
    reward_count = _count_overlap(tokens, REWARD_WORDS)
    credential_count = _count_overlap(tokens, CREDENTIAL_WORDS)
    action_count = _count_overlap(tokens, ACTION_VERBS)
    positive_count = _count_overlap(tokens, POSITIVE_WORDS)
    negative_count = _count_overlap(tokens, NEGATIVE_WORDS)
    trust_count = _count_overlap(tokens, TRUST_WORDS)
    financial_count = _count_overlap(tokens, FINANCIAL_WORDS)

    return {
        "token_count": float(token_count),
        "char_count": float(len(text)),
        "avg_token_length": float(np.mean([len(t) for t in tokens]) if tokens else 0.0),
        "num_urls": float(num_urls),
        "num_emails": float(num_emails),
        "num_digits": float(num_digits),
        "exclamation_count": float(exclamation_count),
        "question_count": float(question_count),
        "uppercase_ratio": float(uppercase_ratio),
        "urgency_count": float(urgency_count),
        "threat_count": float(threat_count),
        "authority_count": float(authority_count),
        "reward_count": float(reward_count),
        "credential_count": float(credential_count),
        "action_count": float(action_count),
        "positive_count": float(positive_count),
        "negative_count": float(negative_count),
        "trust_count": float(trust_count),
        "financial_count": float(financial_count),
        "sentiment_balance": float(positive_count - negative_count),
        "emotion_intensity": float(exclamation_count + question_count + urgency_count + threat_count),
    }


def build_feature_frame(texts: Iterable[str]) -> pd.DataFrame:
    rows = [extract_feature_dict(text) for text in texts]
    return pd.DataFrame(rows)
