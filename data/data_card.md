# Demo data card

This repository includes a **synthetic teaching dataset** in `data/demo_emails.csv`.
It is intended for smoke tests, classroom demonstrations, and GitHub CI validation.
It is **not** a substitute for real-world benchmarking.

## Expected CSV schema

- `text`: email or message body as plain text
- `label`: `1` for phishing, `0` for benign

## Recommended real datasets for full experiments

- CEAS-08 phishing email corpus
- Enron email datasets for benign corporate mail
- SpamAssassin public corpus
- Nazario phishing corpus

For a stronger master's-level experiment, combine a legitimate corpus such as Enron with a phishing corpus and report train/test separation carefully.
