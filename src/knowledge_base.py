"""SQLite knowledge base and semantic retrieval for campus information."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

from .utils import DB_PATH, LOCATION_JSON, bootstrap_project_assets, load_locations


@dataclass
class SearchResult:
    record: dict[str, Any]
    score: float


class CampusKnowledgeBase:
    """Campus records stored in SQLite with optional SentenceTransformer search."""

    def __init__(self, db_path: Path = DB_PATH, json_path: Path = LOCATION_JSON):
        self.db_path = Path(db_path)
        self.json_path = Path(json_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._sentence_model = None
        self._semantic_embeddings: np.ndarray | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        bootstrap_project_assets(generate_figures=False)
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS campus_locations (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    opening_hours TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    events TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    modalities TEXT,
                    predicted_location TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        self.populate_from_json()

    def populate_from_json(self) -> None:
        locations = load_locations()
        with self.connect() as conn:
            for row in locations:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO campus_locations
                    (id, name, category, description, opening_hours, latitude, longitude, events)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["name"],
                        row["category"],
                        row["description"],
                        row["opening_hours"],
                        row["coordinates"]["lat"],
                        row["coordinates"]["lon"],
                        json.dumps(row["events"]),
                    ),
                )
            conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "category": row["category"],
            "description": row["description"],
            "opening_hours": row["opening_hours"],
            "coordinates": {"lat": row["latitude"], "lon": row["longitude"]},
            "events": json.loads(row["events"]),
        }

    def all_locations(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM campus_locations ORDER BY id").fetchall()
        return [self._row_to_dict(row) for row in rows]

    def as_dataframe(self) -> pd.DataFrame:
        records = []
        for location in self.all_locations():
            record = dict(location)
            record["lat"] = record["coordinates"]["lat"]
            record["lon"] = record["coordinates"]["lon"]
            record["events"] = "; ".join(record["events"])
            del record["coordinates"]
            records.append(record)
        return pd.DataFrame(records)

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM campus_locations WHERE lower(name)=lower(?)",
                (name,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_id(self, location_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM campus_locations WHERE id=?", (location_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    @staticmethod
    def expand_query(query: str) -> str:
        """Add lightweight spelling and alias normalization for campus queries."""

        replacements = {
            "heppening": "happening",
            "hapenning": "happening",
            "eventhub": "innovation hub",
            "event hub": "innovation hub",
            "it lab": "computer lab",
            "student union building": "student union",
            "gym": "sports centre",
        }
        expanded = query.lower()
        for source, target in replacements.items():
            expanded = expanded.replace(source, target)
        return expanded

    def lexical_search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query = self.expand_query(query)
        locations = self.all_locations()
        corpus = [
            f"{item['name']} {item['category']} {item['description']} {item['opening_hours']} {' '.join(item['events'])}"
            for item in locations
        ]
        if TfidfVectorizer is None or cosine_similarity is None:
            query_terms = set(query.lower().split())
            scores = []
            for doc in corpus:
                doc_terms = set(doc.lower().split())
                scores.append(len(query_terms & doc_terms) / max(len(query_terms), 1))
            order = np.argsort(scores)[::-1][:top_k]
            return [SearchResult(locations[i], float(scores[i])) for i in order]
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        query_vec = self._tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(locations[i], float(scores[i])) for i in order]

    def _load_sentence_model(self):
        if self._sentence_model is not None:
            return self._sentence_model
        try:
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            self._sentence_model = False
        return self._sentence_model

    def semantic_texts(self) -> list[str]:
        texts = []
        for item in self.all_locations():
            texts.append(
                f"{item['name']}. Category: {item['category']}. {item['description']} "
                f"Opening hours: {item['opening_hours']}. Events: {'; '.join(item['events'])}."
            )
        return texts

    def semantic_embeddings(self) -> np.ndarray:
        if self._semantic_embeddings is not None:
            return self._semantic_embeddings
        model = self._load_sentence_model()
        texts = self.semantic_texts()
        if model:
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self._semantic_embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            if TfidfVectorizer is None:
                embeddings = []
                for text in texts:
                    vector = np.zeros(384, dtype=np.float32)
                    for token in text.lower().split():
                        vector[hash(token) % 384] += 1.0
                    vector /= np.linalg.norm(vector) + 1e-8
                    embeddings.append(vector)
                self._semantic_embeddings = np.vstack(embeddings)
                return self._semantic_embeddings
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=384)
            matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
            self._semantic_embeddings = matrix / norms
            self._tfidf_vectorizer = vectorizer
            self._tfidf_matrix = matrix
        return self._semantic_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        query = self.expand_query(query)
        model = self._load_sentence_model()
        if model:
            vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            return np.asarray(vector, dtype=np.float32)
        if self._tfidf_vectorizer is None:
            self.semantic_embeddings()
        if self._tfidf_vectorizer is None:
            vector = np.zeros(384, dtype=np.float32)
            for token in query.lower().split():
                vector[hash(token) % 384] += 1.0
            return vector / (np.linalg.norm(vector) + 1e-8)
        vector = self._tfidf_vectorizer.transform([query]).toarray().astype(np.float32)[0]
        denom = np.linalg.norm(vector) + 1e-8
        return vector / denom

    def semantic_search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query = self.expand_query(query)
        locations = self.all_locations()
        embeddings = self.semantic_embeddings()
        query_vector = self.encode_query(query)
        if embeddings.shape[1] != query_vector.shape[0]:
            return self.lexical_search(query, top_k=top_k)
        scores = embeddings @ query_vector
        order = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(locations[i], float(scores[i])) for i in order]

    def direction_hint(self, location: dict[str, Any]) -> str:
        """Return a simple route-style hint from the central quad."""

        lat = location["coordinates"]["lat"]
        lon = location["coordinates"]["lon"]
        centre_lat, centre_lon = 51.7540, -1.2540
        north_south = "north" if lat > centre_lat else "south"
        east_west = "east" if lon > centre_lon else "west"
        minutes = max(2, int((abs(lat - centre_lat) + abs(lon - centre_lon)) * 9000))
        return (
            f"From the central quad, walk {north_south}-{east_west} for about {minutes} minutes. "
            f"Use the main signed entrance for {location['name']}."
        )

    def log_interaction(
        self,
        query: str,
        modalities: list[str],
        predicted_location: str | None,
        confidence: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO interactions (query, modalities, predicted_location, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (query, ",".join(modalities), predicted_location or "", float(confidence)),
            )
            conn.commit()

    def export_json(self, output_path: Path | None = None) -> Path:
        output_path = output_path or self.db_path.with_name("campus_knowledge_export.json")
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.all_locations(), handle, indent=2)
        return output_path


def main() -> None:
    kb = CampusKnowledgeBase()
    kb.export_json()
    print(f"Knowledge base ready: {kb.db_path}")
    print(kb.as_dataframe()[["id", "name", "category"]].to_string(index=False))


if __name__ == "__main__":
    main()
