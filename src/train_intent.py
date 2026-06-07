"""Fine-tune DistilBERT for campus FAQ intent classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .utils import FAQ_CSV, FIGURES_DIR, INTENTS, MODELS_DIR, RESULTS_DIR, bootstrap_project_assets


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_training_curves(log_history: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    train_steps = [row["step"] for row in log_history if "loss" in row]
    train_loss = [row["loss"] for row in log_history if "loss" in row]
    eval_steps = [row["step"] for row in log_history if "eval_loss" in row]
    eval_loss = [row["eval_loss"] for row in log_history if "eval_loss" in row]
    eval_acc = [row["eval_accuracy"] for row in log_history if "eval_accuracy" in row]

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_loss:
        ax.plot(train_steps, train_loss, label="Training loss", color="#2f6f9f")
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="Validation loss", color="#b14d4d")
    ax.set_title("DistilBERT Training and Validation Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_curves.png", dpi=180)
    plt.close(fig)

    if eval_acc:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eval_steps, eval_acc, color="#3f7f5f", marker="o")
        ax.set_title("Validation Accuracy Curve")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(output_dir / "validation_accuracy_curve.png", dpi=180)
        plt.close(fig)


def train(
    dataset_path: Path = FAQ_CSV,
    output_dir: Path = MODELS_DIR / "nlp" / "distilbert-intent",
    epochs: int = 3,
    batch_size: int = 16,
) -> dict:
    bootstrap_project_assets()
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    df = pd.read_csv(dataset_path)
    label2id = {label: idx for idx, label in enumerate(INTENTS)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df["intent"].map(label2id)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["intent"], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(INTENTS),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    plot_training_curves(trainer.state.log_history)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "distilbert_training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT intent classifier.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    metrics = train(epochs=args.epochs, batch_size=args.batch_size)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
