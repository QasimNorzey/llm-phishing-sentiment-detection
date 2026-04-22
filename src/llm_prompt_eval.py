from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

from src.common import evaluate_predictions, save_json
from src.data_utils import load_dataset

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

PROMPT_TEMPLATE = """You are a cybersecurity analyst. Classify the following message as phishing or benign.
Return only valid JSON with these keys:
label: one of [phishing, benign]
risk_score: integer from 0 to 100
urgency: integer from 0 to 5
authority: integer from 0 to 5
reward: integer from 0 to 5
reason: short explanation

Message:
{text}
"""


def parse_json(text: str) -> dict:
    match = JSON_RE.search(text)
    if not match:
        raise ValueError(f"Could not find JSON in model output: {text[:200]}")
    return json.loads(match.group(0))


def query_ollama(model: str, prompt: str, host: str) -> dict:
    response = requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return parse_json(payload.get("response", ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a local LLM via Ollama on phishing classification prompts.")
    parser.add_argument("--data", default="data/demo_emails.csv")
    parser.add_argument("--outdir", default="results/ollama_eval")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    parser.add_argument("--limit", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.data).head(args.limit)

    records = []
    y_true = []
    y_pred = []
    for _, row in data.iterrows():
        prompt = PROMPT_TEMPLATE.format(text=row["text"])
        result = query_ollama(args.model, prompt, args.host)
        pred = 1 if str(result.get("label", "")).strip().lower() == "phishing" else 0
        y_true.append(int(row["label"]))
        y_pred.append(pred)
        records.append({
            "text": row["text"],
            "true_label": int(row["label"]),
            "predicted_label": pred,
            "risk_score": result.get("risk_score"),
            "urgency": result.get("urgency"),
            "authority": result.get("authority"),
            "reward": result.get("reward"),
            "reason": result.get("reason"),
        })

    pd.DataFrame(records).to_csv(outdir / "predictions.csv", index=False)
    metrics = evaluate_predictions(y_true, y_pred)
    save_json(metrics, outdir / "metrics.json")
    (outdir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
