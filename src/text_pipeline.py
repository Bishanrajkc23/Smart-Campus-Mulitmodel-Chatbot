"""Text understanding pipeline for FAQ intent classification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from .utils import FAQ_CSV, FIGURES_DIR, INTENTS, MODELS_DIR, RESULTS_DIR, normalize_text, write_faq_dataset


class IntentClassifier:
    """DistilBERT intent classifier with an interpretable fallback."""

    def __init__(
        self,
        model_dir: Path = MODELS_DIR / "nlp" / "distilbert-intent",
        base_model: str = "distilbert-base-uncased",
    ):
        self.model_dir = Path(model_dir)
        self.base_model = base_model
        self.id2label = {i: label for i, label in enumerate(INTENTS)}
        self.label2id = {label: i for i, label in self.id2label.items()}
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            source = str(self.model_dir) if self.model_dir.exists() else self.base_model
            self.tokenizer = AutoTokenizer.from_pretrained(source)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                source,
                num_labels=len(INTENTS),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
            self.model.eval()
        except Exception:
            self.tokenizer = None
            self.model = None

    @staticmethod
    def preprocess(text: str) -> str:
        return normalize_text(text)

    def _heuristic_predict(self, text: str) -> tuple[str, float]:
        q = self.preprocess(text)
        rules = [
            ("greeting", r"\b(hello|hi|hey|morning|afternoon|start)\b"),
            ("goodbye", r"\b(bye|goodbye|thanks|cheers|done|end)\b"),
            ("opening_hours", r"\b(open|close|hours|time|weekend|today)\b"),
            ("event_query", r"\b(event|events|workshop|activity|happening|on at|fair|seminar)\b"),
            ("food_services", r"\b(food|lunch|coffee|breakfast|halal|vegetarian|snack|cafeteria)\b"),
            ("study_space", r"\b(study|revise|quiet|silent|booth|room|coursework)\b"),
            ("accessibility", r"\b(accessible|accessibility|disability|assistive|step free|adjustment|neurodiversity)\b"),
            ("find_location", r"\b(where|directions|find|get to|take me|which way|nearest)\b"),
            ("facility_information", r"\b(facilities|describe|information|what can|have|support)\b"),
        ]
        for intent, pattern in rules:
            if re.search(pattern, q):
                return intent, 0.74
        return "other", 0.55

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = self.preprocess(text)
        if self.model is None or self.tokenizer is None:
            intent, confidence = self._heuristic_predict(cleaned)
            return {
                "intent": intent,
                "confidence": confidence,
                "probabilities": {intent: confidence},
                "preprocessed_text": cleaned,
                "model": "heuristic-fallback",
            }
        import torch

        encoded = self.tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        label_id = int(np.argmax(probs))
        return {
            "intent": self.id2label[label_id],
            "confidence": float(probs[label_id]),
            "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(probs)},
            "preprocessed_text": cleaned,
            "model": "distilbert-base-uncased",
        }

    def batch_predict(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.predict(text) for text in texts]

    def evaluate(self, dataset_path: Path = FAQ_CSV, output_dir: Path = FIGURES_DIR) -> dict[str, Any]:
        if not dataset_path.exists():
            write_faq_dataset(300)
        df = pd.read_csv(dataset_path)
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["intent"], random_state=42)
        y_true = val_df["intent"].tolist()
        y_pred = [self.predict(text)["intent"] for text in val_df["text"].tolist()]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=INTENTS, average="weighted", zero_division=0
        )
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_train": len(train_df),
            "n_validation": len(val_df),
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "intent_metrics.csv", index=False)
        with (RESULTS_DIR / "intent_classification_report.json").open("w", encoding="utf-8") as handle:
            json.dump(classification_report(y_true, y_pred, output_dict=True, zero_division=0), handle, indent=2)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = INTENTS
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(11, 9))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=60, colorbar=False)
        ax.set_title("Intent Classification Confusion Matrix")
        fig.tight_layout()
        fig.savefig(output_dir / "intent_confusion_matrix.png", dpi=180)
        plt.close(fig)
        return metrics


def main() -> None:
    classifier = IntentClassifier()
    print(classifier.predict("What time does the library open?"))
    print(classifier.evaluate())


if __name__ == "__main__":
    main()
