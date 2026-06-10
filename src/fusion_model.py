from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None
try:
    from sklearn.metrics import accuracy_score
except Exception:
    def accuracy_score(y_true, y_pred):
        if not y_true:
            return 0.0
        return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

from .knowledge_base import CampusKnowledgeBase
from .utils import FIGURES_DIR, RESULTS_DIR


if nn is not None:
    class MLPFusionNetwork(nn.Module):
        def __init__(self, input_dim: int, n_locations: int, hidden_dim: int = 512, dropout: float = 0.2):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_locations),
            )

        def forward(self, fused):
            return self.network(fused)
else:
    class MLPFusionNetwork:
        def __init__(self, *args, **kwargs):
            self.available = False


@dataclass
class FusionInput:
    clip_embedding: np.ndarray | None = None
    text_embedding: np.ndarray | None = None
    transcript_embedding: np.ndarray | None = None
    text_query: str = ""
    transcript: str = ""


class MultiModalFusionEngine:
    def __init__(
        self,
        knowledge_base: CampusKnowledgeBase | None = None,
        embedding_dim: int = 512,
        checkpoint_path: Path = Path("models/fusion/fusion_mlp.pt"),
    ):
        self.kb = knowledge_base or CampusKnowledgeBase()
        self.embedding_dim = embedding_dim
        self.records = self.kb.all_locations()
        self.checkpoint_path = Path(checkpoint_path)
        self.network = MLPFusionNetwork(input_dim=embedding_dim * 3 + 3, n_locations=len(self.records))
        self._sentence_model = None
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if torch is not None and hasattr(self.network, "load_state_dict") and self.checkpoint_path.exists():
            try:
                self.network.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
                self.network.eval()
            except Exception:
                pass

    def _load_sentence_model(self):
        if self._sentence_model is not None:
            return self._sentence_model
        try:
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            self._sentence_model = False
        return self._sentence_model

    def encode_text(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        model = self._load_sentence_model()
        if model:
            vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            vector = np.asarray(vector, dtype=np.float32)
        else:
            vector = self.kb.encode_query(text).astype(np.float32)
        return self._resize_embedding(vector)

    def _resize_embedding(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape[0] == self.embedding_dim:
            return vector.astype(np.float32)
        if vector.shape[0] > self.embedding_dim:
            resized = vector[: self.embedding_dim]
        else:
            resized = np.pad(vector, (0, self.embedding_dim - vector.shape[0]), mode="constant")
        norm = np.linalg.norm(resized) + 1e-8
        return (resized / norm).astype(np.float32)

    def fuse(self, fusion_input: FusionInput) -> tuple[np.ndarray, np.ndarray]:
        vectors = []
        mask = []
        for vector in [
            fusion_input.clip_embedding,
            fusion_input.text_embedding,
            fusion_input.transcript_embedding,
        ]:
            if vector is None:
                vectors.append(np.zeros(self.embedding_dim, dtype=np.float32))
                mask.append(0.0)
            else:
                vectors.append(self._resize_embedding(vector))
                mask.append(1.0)
        fused = np.concatenate(vectors + [np.asarray(mask, dtype=np.float32)])
        return fused.astype(np.float32), np.asarray(mask, dtype=np.float32)

    def retrieve(self, fusion_input: FusionInput, top_k: int = 3) -> dict[str, Any]:
        if fusion_input.text_embedding is None and fusion_input.text_query:
            fusion_input.text_embedding = self.encode_text(fusion_input.text_query)
        if fusion_input.transcript_embedding is None and fusion_input.transcript:
            fusion_input.transcript_embedding = self.encode_text(fusion_input.transcript)

        fused, mask = self.fuse(fusion_input)
        query_text = " ".join([fusion_input.text_query, fusion_input.transcript]).strip()
        expanded_query = self.kb.expand_query(query_text)
        semantic_results = self.kb.semantic_search(query_text, top_k=top_k) if query_text else []

        combined: dict[int, float] = {}

        for idx, record in enumerate(self.records):
            if record["name"].lower() in expanded_query:
                combined[idx] = combined.get(idx, 0.0) + 1.0

        if torch is not None and hasattr(self.network, "eval"):
            with torch.no_grad():
                logits = self.network(torch.from_numpy(fused).unsqueeze(0))
                probs = torch.softmax(logits, dim=-1).numpy()[0]
            mlp_order = np.argsort(probs)[::-1]
            for idx in mlp_order[:top_k]:
                combined[int(idx)] = combined.get(int(idx), 0.0) + float(probs[idx])

        for result in semantic_results:
            idx = next(i for i, rec in enumerate(self.records) if rec["id"] == result.record["id"])
            combined[idx] = combined.get(idx, 0.0) + 0.5 * max(result.score, 0.0)

        if not combined and query_text:
            for result in self.kb.lexical_search(query_text, top_k=top_k):
                idx = next(i for i, rec in enumerate(self.records) if rec["id"] == result.record["id"])
                combined[idx] = max(result.score, 0.05)

        if not combined:
            combined = {idx: 0.05 for idx in range(min(top_k, len(self.records)))}

        order = sorted(combined, key=combined.get, reverse=True)[:top_k]
        max_score = max([combined[idx] for idx in order], default=1.0)
        normalizer = max(max_score, 1.0)
        top3 = [{"location": self.records[idx], "score": float(combined[idx] / normalizer)} for idx in order]
        confidence = top3[0]["score"] if top3 else 0.0
        return {
            "top_prediction": top3[0]["location"] if top3 else None,
            "confidence": confidence,
            "top_3": top3,
            "mask": mask.tolist(),
            "fused_embedding": fused,
            "explanation": "Missing modalities are zero-padded and masked. If PyTorch is available, an MLP score is blended with semantic KB retrieval; otherwise the app uses semantic retrieval only.",
        }

    def evaluate(self, scenarios: pd.DataFrame | None = None, output_dir: Path = FIGURES_DIR) -> dict[str, float]:
        if scenarios is None:
            scenarios = pd.DataFrame(
                [
                    {"query": "Where is the Library?", "label": "Library"},
                    {"query": "What food is available at the Cafeteria?", "label": "Cafeteria"},
                    {"query": "I need disability support", "label": "Accessibility Office"},
                    {"query": "Show me start-up events", "label": "Innovation Hub"},
                    {"query": "Where can I exercise?", "label": "Sports Centre"},
                ]
            )
        y_true, y_pred, top3_hits = [], [], []
        for _, row in scenarios.iterrows():
            result = self.retrieve(FusionInput(text_query=row["query"]), top_k=3)
            candidates = [item["location"]["name"] for item in result["top_3"]]
            y_true.append(row["label"])
            y_pred.append(candidates[0])
            top3_hits.append(row["label"] in candidates)
        metrics = {
            "end_to_end_retrieval_accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
            "top3_retrieval_accuracy": float(np.mean(top3_hits)) if top3_hits else 0.0,
            "n_samples": float(len(y_true)),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "fusion_metrics.csv", index=False)

        if y_true:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(["Top-1", "Top-3"], [metrics["end_to_end_retrieval_accuracy"], metrics["top3_retrieval_accuracy"]])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy")
            ax.set_title("Fusion Retrieval Accuracy")
            fig.tight_layout()
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / "fusion_accuracy.png", dpi=180)
            plt.close(fig)
        return metrics


def main() -> None:
    engine = MultiModalFusionEngine()
    print(engine.retrieve(FusionInput(text_query="Where can I find quiet study rooms?")))


if __name__ == "__main__":
    main()
