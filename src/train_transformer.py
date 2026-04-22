from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.common import evaluate_predictions, save_json
from src.data_utils import load_dataset
from src.features import build_feature_frame


@dataclass
class BatchEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    numeric_features: torch.Tensor
    labels: torch.Tensor


class PhishingHybridDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, numeric_features, max_length: int = 256):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> BatchEncoding:
        return BatchEncoding(
            input_ids=self.encodings["input_ids"][idx],
            attention_mask=self.encodings["attention_mask"][idx],
            numeric_features=self.numeric_features[idx],
            labels=self.labels[idx],
        )


class HybridTransformerClassifier(nn.Module):
    def __init__(self, encoder_name: str, extra_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask, numeric_features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        combined = torch.cat([pooled, numeric_features], dim=1)
        return self.classifier(combined)


def collate_fn(batch: list[BatchEncoding]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item.input_ids for item in batch]),
        "attention_mask": torch.stack([item.attention_mask for item in batch]),
        "numeric_features": torch.stack([item.numeric_features for item in batch]),
        "labels": torch.stack([item.labels for item in batch]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a hybrid transformer + sentiment feature model.")
    parser.add_argument("--data", default="data/demo_emails.csv")
    parser.add_argument("--outdir", default="results/transformer_hybrid")
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser


def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                numeric_features=batch["numeric_features"].to(device),
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            labels = batch["labels"].cpu().numpy().tolist()
            y_true.extend(labels)
            y_pred.extend(preds)
    return evaluate_predictions(y_true, y_pred)


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(args.data)
    train_df, test_df = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_numeric = build_feature_frame(train_df["text"])
    test_numeric = build_feature_frame(test_df["text"])
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_numeric)
    X_test_num = scaler.transform(test_numeric)

    train_dataset = PhishingHybridDataset(train_df["text"], train_df["label"], tokenizer, X_train_num, max_length=args.max_length)
    test_dataset = PhishingHybridDataset(test_df["text"], test_df["label"], tokenizer, X_test_num, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = HybridTransformerClassifier(args.model_name, extra_dim=X_train_num.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -np.inf
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                numeric_features=batch["numeric_features"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

        metrics = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}: f1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}, precision={metrics['precision']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            save_json(metrics, outdir / "metrics.json")
            (outdir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")

    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "numeric_columns": train_numeric.columns.tolist(),
                "model_name": args.model_name,
            },
            outdir / "hybrid_transformer.pt",
        )


if __name__ == "__main__":
    main()
