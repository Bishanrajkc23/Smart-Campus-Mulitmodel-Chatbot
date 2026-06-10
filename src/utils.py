from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"

LOCATION_JSON = DATA_DIR / "campus_locations.json"
FAQ_CSV = DATA_DIR / "faq_dataset.csv"
DB_PATH = DATA_DIR / "knowledge_base.db"

INTENTS = [
    "find_location",
    "opening_hours",
    "event_query",
    "facility_information",
    "study_space",
    "food_services",
    "accessibility",
    "greeting",
    "goodbye",
    "other",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "can",
    "could",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "please",
    "show",
    "the",
    "to",
    "what",
    "where",
    "with",
    "you",
}


def ensure_directories() -> None:
    paths = [
        DATA_DIR / "campus_images",
        DATA_DIR / "audio_queries",
        MODELS_DIR / "vision",
        MODELS_DIR / "nlp",
        MODELS_DIR / "fusion",
        MODELS_DIR / "checkpoints",
        PROJECT_ROOT / "notebooks",
        FIGURES_DIR,
        RESULTS_DIR,
        PROJECT_ROOT / "tests",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_locations() -> list[dict]:
    with LOCATION_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_export(locations: Sequence[dict] | None = None) -> Path:
    ensure_directories()
    locations = list(locations or load_locations())
    export_path = DATA_DIR / "campus_knowledge_export.json"
    with export_path.open("w", encoding="utf-8") as handle:
        json.dump(locations, handle, indent=2)
    return export_path


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def simple_lemmatize(token: str) -> str:
    for suffix in ("ing", "ies", "ied", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            if suffix == "ies":
                return token[: -len(suffix)] + "y"
            return token[: -len(suffix)]
    return token


def preprocess_text(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = normalize_text(text).split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return [simple_lemmatize(token) for token in tokens]


def _location_templates(location: dict) -> dict[str, list[str]]:
    name = location["name"]
    category = location["category"].lower()
    return {
        "find_location": [
            f"Where is the {name}?",
            f"How do I get to {name}?",
            f"Take me to the {name}",
            f"Give me directions to {name}",
            f"Which way is {name} from here?",
        ],
        "opening_hours": [
            f"What time does {name} open?",
            f"When does the {name} close today?",
            f"Tell me the opening hours for {name}",
            f"Is {name} open this weekend?",
            f"What are the hours at {name}?",
        ],
        "event_query": [
            f"What events are happening at {name}?",
            f"Are there any activities in the {name}?",
            f"Show upcoming events for {name}",
            f"What is on at {name} this week?",
            f"Tell me about workshops at {name}",
        ],
        "facility_information": [
            f"What facilities does {name} have?",
            f"Describe the {name}",
            f"What can I do in the {name}?",
            f"Does {name} have useful student facilities?",
            f"Give me information about the {category} building",
        ],
    }


def build_synthetic_faq_dataset(min_examples: int = 300) -> pd.DataFrame:
    locations = load_locations()
    rows: list[dict[str, str]] = []

    for location in locations:
        for intent, questions in _location_templates(location).items():
            rows.extend({"text": q, "intent": intent, "location": location["name"]} for q in questions)

    study_questions = [
        "Where can I find a quiet study space?",
        "I need a group study room",
        "Which building is best for late night studying?",
        "Is there somewhere silent to revise?",
        "Can I book a study booth?",
        "Find me a place to study with a laptop",
        "Where are postgraduate study rooms?",
        "I need study space near the library",
        "Show study facilities on campus",
        "Where can my team work on coursework?",
    ]
    food_questions = [
        "Where can I buy lunch?",
        "Is there halal food on campus?",
        "Where is the nearest coffee?",
        "What food options are open now?",
        "I need vegetarian food",
        "Where can I get breakfast?",
        "Show me campus dining places",
        "Can I buy snacks nearby?",
        "Where is the cafeteria?",
        "Which building has food services?",
    ]
    accessibility_questions = [
        "Which buildings have accessible entrances?",
        "I need disability support",
        "Where can I ask about exam adjustments?",
        "Does this place have assistive technology?",
        "How do I get accessibility help?",
        "Where is the accessibility office?",
        "I need step free campus guidance",
        "Can someone help with a learning plan?",
        "Where can I get support for neurodiversity?",
        "Does the lecture hall have an induction loop?",
    ]
    greetings = [
        "hello",
        "hi there",
        "good morning",
        "hey campus assistant",
        "can you help me",
        "hello chatbot",
        "good afternoon",
        "hi, I need help",
        "are you there",
        "start campus tour",
    ]
    goodbyes = [
        "bye",
        "goodbye",
        "thanks that is all",
        "see you later",
        "end chat",
        "thanks for your help",
        "that answers my question",
        "I am done now",
        "close the assistant",
        "cheers goodbye",
    ]
    other = [
        "What is the weather today?",
        "Who won the football match?",
        "Tell me a joke",
        "Can you solve my maths homework?",
        "What is the meaning of life?",
        "Play music",
        "Order a taxi",
        "How much is tuition next year?",
        "Translate this sentence",
        "What is the university ranking?",
    ]

    extras = {
        "study_space": study_questions,
        "food_services": food_questions,
        "accessibility": accessibility_questions,
        "greeting": greetings,
        "goodbye": goodbyes,
        "other": other,
    }
    for intent, questions in extras.items():
        for question in questions:
            rows.append({"text": question, "intent": intent, "location": ""})

    base = pd.DataFrame(rows)
    augmented_rows = []
    polite_prefixes = ["", "Please ", "Could you ", "Can you ", "I want to know "]
    suffixes = ["", " today", " on campus", " please", " for a new student"]

    i = 0
    while len(base) + len(augmented_rows) < min_examples:
        row = base.iloc[i % len(base)].to_dict()
        prefix = polite_prefixes[(i // len(base)) % len(polite_prefixes)]
        suffix = suffixes[i % len(suffixes)]
        text = f"{prefix}{row['text']}{suffix}".strip()
        augmented_rows.append({"text": text, "intent": row["intent"], "location": row.get("location", "")})
        i += 1

    dataset = pd.concat([base, pd.DataFrame(augmented_rows)], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return dataset


def write_faq_dataset(min_examples: int = 300) -> Path:
    ensure_directories()
    df = build_synthetic_faq_dataset(min_examples=min_examples)
    df.to_csv(FAQ_CSV, index=False)
    return FAQ_CSV


def create_placeholder_campus_images(force: bool = False) -> list[Path]:
    ensure_directories()
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return []

    image_dir = DATA_DIR / "campus_images"
    created: list[Path] = []
    colors = [
        (33, 92, 147),
        (42, 126, 98),
        (184, 104, 48),
        (116, 78, 145),
        (169, 69, 79),
    ]
    locations = load_locations()
    for idx, location in enumerate(locations):
        output = image_dir / f"{location['name'].lower().replace(' ', '_')}.png"
        if output.exists() and not force:
            created.append(output)
            continue
        img = Image.new("RGB", (640, 420), color=(242, 245, 247))
        draw = ImageDraw.Draw(img)
        accent = colors[idx % len(colors)]
        draw.rectangle((0, 0, 640, 95), fill=accent)
        draw.rectangle((40, 135, 600, 375), outline=accent, width=6)
        draw.rectangle((65, 165, 575, 220), fill=(255, 255, 255))
        draw.text((80, 178), location["name"], fill=(20, 31, 43))
        draw.text((80, 240), location["category"], fill=accent)
        draw.text((80, 285), "Smart Campus Synthetic Image", fill=(64, 75, 87))
        img.save(output)
        created.append(output)
    return created


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    dp[:, 0] = np.arange(len(ref) + 1)
    dp[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[len(ref), len(hyp)] / len(ref))


def save_class_distribution(df: pd.DataFrame, label_col: str, output_path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    counts = df[label_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    counts.plot(kind="bar", ax=ax, color="#2f6f9f")
    ax.set_title(title)
    ax.set_xlabel(label_col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def bootstrap_project_assets(force: bool = False, generate_figures: bool = True) -> None:
    ensure_directories()
    save_json_export()
    if force or not FAQ_CSV.exists() or len(pd.read_csv(FAQ_CSV)) < 300:
        write_faq_dataset(min_examples=300)
    create_placeholder_campus_images(force=force)
    if generate_figures:
        faq_df = pd.read_csv(FAQ_CSV)
        try:
            save_class_distribution(
                faq_df,
                "intent",
                FIGURES_DIR / "intent_distribution.png",
                "Synthetic FAQ Intent Distribution",
            )
        except Exception:
            pass


def batched(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Smart Campus project assets.")
    parser.add_argument("--generate", action="store_true", help="Generate data, JSON export, images, and charts.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing derived assets.")
    args = parser.parse_args()
    if args.generate:
        bootstrap_project_assets(force=args.force)
        print(f"Generated assets under {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
