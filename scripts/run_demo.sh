#!/usr/bin/env bash
set -euo pipefail

python -m src.build_demo_dataset
python -m src.train_baseline --data data/demo_emails.csv --outdir results/baseline
python -m src.train_hybrid --data data/demo_emails.csv --outdir results/hybrid
python scripts/summarize_results.py
