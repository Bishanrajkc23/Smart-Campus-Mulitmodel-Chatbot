from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from .knowledge_base import CampusKnowledgeBase
from .utils import DATA_DIR, FIGURES_DIR


class CLIPCampusRetriever:
    def __init__(
        self,
        knowledge_base: CampusKnowledgeBase | None = None,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        self.kb = knowledge_base or CampusKnowledgeBase()
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.index = None
        self.text_embeddings: np.ndarray | None = None
        self._load_clip()
        self.build_index()

    def _load_clip(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception:
            self.model = None
            self.processor = None
            self.device = "cpu"

    @staticmethod
    def preprocess_image(image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
        image = image.convert("RGB")
        image.thumbnail(size)
        canvas = Image.new("RGB", size, (245, 247, 249))
        left = (size[0] - image.width) // 2
        top = (size[1] - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas

    @staticmethod
    def preprocess_with_opencv(image_path: str | Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        return (image - mean) / std

    @staticmethod
    def augment_image(image: Image.Image) -> list[Image.Image]:
        from PIL import ImageEnhance

        image = image.convert("RGB")
        return [
            image,
            image.rotate(5),
            image.rotate(-5),
            ImageEnhance.Color(image).enhance(1.2),
            ImageEnhance.Contrast(image).enhance(1.15),
        ]

    def _fallback_embedding(self, text_or_bytes: str | bytes, dim: int = 512) -> np.ndarray:
        if isinstance(text_or_bytes, str):
            payload = text_or_bytes.encode("utf-8")
        else:
            payload = text_or_bytes
        digest = hashlib.sha256(payload).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vector = rng.normal(size=dim).astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-8
        return vector

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.model is None or self.processor is None:
            return np.vstack([self._fallback_embedding(text) for text in texts])
        import torch

        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            embeddings = self.model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)

    def encode_image(self, image: Image.Image | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = self.preprocess_image(image)
        if self.model is None or self.processor is None:
            return self._fallback_embedding(image.tobytes())
        import torch

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()[0].astype(np.float32)

    def build_index(self) -> None:
        records = self.kb.all_locations()
        texts = [
            f"Photo of {r['name']}, a {r['category']} campus building. {r['description']}"
            for r in records
        ]
        self.text_embeddings = self.encode_texts(texts)
        try:
            import faiss

            index = faiss.IndexFlatIP(self.text_embeddings.shape[1])
            index.add(self.text_embeddings)
            self.index = index
        except Exception:
            self.index = None

    def retrieve(self, image: Image.Image | str | Path, top_k: int = 3) -> dict[str, Any]:
        query = self.encode_image(image).reshape(1, -1).astype(np.float32)
        records = self.kb.all_locations()
        if self.index is not None:
            scores, indices = self.index.search(query, top_k)
            order = indices[0]
            sims = scores[0]
        else:
            sims_all = self.text_embeddings @ query.ravel()
            order = np.argsort(sims_all)[::-1][:top_k]
            sims = sims_all[order]
        candidates = [
            {
                "location": records[int(idx)],
                "score": float(max(min(score, 1.0), -1.0)),
            }
            for idx, score in zip(order, sims)
        ]
        return {
            "top_prediction": candidates[0]["location"],
            "similarity_score": candidates[0]["score"],
            "top_3": candidates,
            "embedding": query.ravel(),
            "explanation": "Image embedding compared with CLIP text embeddings of campus descriptions using cosine similarity.",
        }

    def evaluate(self, manifest: pd.DataFrame | None = None, output_dir: Path = FIGURES_DIR) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if manifest is None:
            rows = []
            for image_path in sorted((DATA_DIR / "campus_images").glob("*.png")):
                label = image_path.stem.replace("_", " ").title()
                rows.append({"image_path": str(image_path), "label": label})
            manifest = pd.DataFrame(rows)

        y_true, y_pred, top3_hits = [], [], []
        for _, row in manifest.iterrows():
            result = self.retrieve(row["image_path"], top_k=3)
            candidates = [item["location"]["name"] for item in result["top_3"]]
            y_true.append(row["label"])
            y_pred.append(candidates[0])
            top3_hits.append(row["label"] in candidates)

        top1 = accuracy_score(y_true, y_pred) if y_true else 0.0
        top3 = float(np.mean(top3_hits)) if top3_hits else 0.0
        labels = sorted(set(y_true) | set(y_pred))
        if labels:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(12, 10))
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=80, colorbar=False)
            ax.set_title("Vision Retrieval Confusion Matrix")
            fig.tight_layout()
            fig.savefig(output_dir / "vision_confusion_matrix.png", dpi=180)
            plt.close(fig)

        metrics = {"top1_accuracy": top1, "top3_accuracy": top3, "n_samples": len(y_true)}
        pd.DataFrame([metrics]).to_csv(output_dir.parent / "results" / "vision_metrics.csv", index=False)
        return metrics


def main() -> None:
    retriever = CLIPCampusRetriever()
    print(retriever.evaluate())


if __name__ == "__main__":
    main()
