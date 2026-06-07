"""End-to-end evaluation runner and visualization generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fusion_model import FusionInput, MultiModalFusionEngine
from .image_pipeline import CLIPCampusRetriever
from .knowledge_base import CampusKnowledgeBase
from .text_pipeline import IntentClassifier
from .utils import FAQ_CSV, FIGURES_DIR, RESULTS_DIR, bootstrap_project_assets


def plot_precision_recall(metrics_path: Path = RESULTS_DIR / "intent_classification_report.json") -> None:
    if not metrics_path.exists():
        return
    with metrics_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    labels = [k for k in report.keys() if isinstance(report[k], dict) and "precision" in report[k]]
    labels = [label for label in labels if label not in {"accuracy", "macro avg", "weighted avg"}]
    if not labels:
        return
    precision = [report[label]["precision"] for label in labels]
    recall = [report[label]["recall"] for label in labels]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.18, precision, width=0.36, label="Precision", color="#2f6f9f")
    ax.bar(x + 0.18, recall, width=0.36, label="Recall", color="#9f3f5f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Intent Precision and Recall by Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_by_intent.png", dpi=180)
    plt.close(fig)


def plot_word_cloud() -> None:
    try:
        from wordcloud import WordCloud
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not FAQ_CSV.exists():
        return
    df = pd.read_csv(FAQ_CSV)
    text = " ".join(df["text"].astype(str).tolist())
    wc = WordCloud(width=1200, height=700, background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("FAQ Dataset Word Cloud")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "faq_word_cloud.png", dpi=180)
    plt.close(fig)


def plot_embedding_similarity(kb: CampusKnowledgeBase) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    embeddings = kb.semantic_embeddings()
    sim = embeddings @ embeddings.T
    labels = [record["name"] for record in kb.all_locations()]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=80, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Campus Knowledge Embedding Similarity")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "embedding_similarity_plot.png", dpi=180)
    plt.close(fig)


def create_testing_scenarios() -> pd.DataFrame:
    scenarios = pd.DataFrame(
        [
            {
                "scenario": "Library image upload",
                "modalities": "image",
                "input": "Synthetic library signage image",
                "expected_location": "Library",
                "expected_intent": "find_location",
            },
            {
                "scenario": "Student union voice query",
                "modalities": "audio",
                "input": "Where is the student union?",
                "expected_location": "Student Union",
                "expected_intent": "find_location",
            },
            {
                "scenario": "Cafeteria opening hours question",
                "modalities": "text",
                "input": "What time does the cafeteria close today?",
                "expected_location": "Cafeteria",
                "expected_intent": "opening_hours",
            },
            {
                "scenario": "Study space request",
                "modalities": "text",
                "input": "Where can I find quiet study rooms?",
                "expected_location": "Library",
                "expected_intent": "study_space",
            },
            {
                "scenario": "Event lookup",
                "modalities": "text",
                "input": "What events are happening at the Innovation Hub?",
                "expected_location": "Innovation Hub",
                "expected_intent": "event_query",
            },
            {
                "scenario": "Accessibility support",
                "modalities": "text",
                "input": "I need step free access and disability support",
                "expected_location": "Accessibility Office",
                "expected_intent": "accessibility",
            },
            {
                "scenario": "Sports facility request",
                "modalities": "text",
                "input": "Where can I use the gym?",
                "expected_location": "Sports Centre",
                "expected_intent": "facility_information",
            },
            {
                "scenario": "Combined photo and text",
                "modalities": "image,text",
                "input": "Library image plus 'what time does this open?'",
                "expected_location": "Library",
                "expected_intent": "opening_hours",
            },
            {
                "scenario": "Food services",
                "modalities": "text",
                "input": "Where can I get vegetarian lunch?",
                "expected_location": "Cafeteria",
                "expected_intent": "food_services",
            },
            {
                "scenario": "Career event query",
                "modalities": "text",
                "input": "Are there any CV or employer events?",
                "expected_location": "Career Centre",
                "expected_intent": "event_query",
            },
        ]
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(RESULTS_DIR / "testing_results.csv", index=False)
    return scenarios


def save_model_comparison(results: dict[str, dict]) -> None:
    rows = []
    if "vision" in results:
        rows.append({"model": "CLIP + FAISS", "metric": "top1_accuracy", "score": results["vision"].get("top1_accuracy", 0)})
        rows.append({"model": "CLIP + FAISS", "metric": "top3_accuracy", "score": results["vision"].get("top3_accuracy", 0)})
    if "intent" in results:
        rows.append({"model": "DistilBERT intent", "metric": "f1", "score": results["intent"].get("f1", 0)})
        rows.append({"model": "DistilBERT intent", "metric": "accuracy", "score": results["intent"].get("accuracy", 0)})
    if "fusion" in results:
        rows.append(
            {
                "model": "Masked MLP fusion",
                "metric": "top1_accuracy",
                "score": results["fusion"].get("end_to_end_retrieval_accuracy", 0),
            }
        )
        rows.append(
            {
                "model": "Masked MLP fusion",
                "metric": "top3_accuracy",
                "score": results["fusion"].get("top3_retrieval_accuracy", 0),
            }
        )
    if not rows:
        return
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = comparison["model"] + "\n" + comparison["metric"]
    ax.bar(labels, comparison["score"], color=["#2f6f9f", "#457f6f", "#9f3f5f", "#b06f3f", "#4f5f9f", "#6f4f9f"][: len(labels)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Summary")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def save_error_analysis(test_results: pd.DataFrame) -> None:
    errors = test_results[
        (test_results["location_pass"] == False) | (test_results["intent_pass"] == False)  # noqa: E712
    ].copy()
    if errors.empty:
        errors = pd.DataFrame(
            [
                {
                    "error_type": "none",
                    "analysis": "All structured scenarios passed under the current retrieval assumptions.",
                    "recommendation": "Extend evaluation with real images and real audio to expose deployment errors.",
                }
            ]
        )
    else:
        errors["error_type"] = np.where(~errors["location_pass"], "retrieval", "intent")
        errors["recommendation"] = "Inspect query wording, add labelled examples, or tune semantic retrieval prompts."
    errors.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)


def run_all(skip_heavy: bool = False) -> dict:
    bootstrap_project_assets()
    kb = CampusKnowledgeBase()
    classifier = IntentClassifier()
    fusion = MultiModalFusionEngine(kb)

    results: dict[str, dict] = {}
    results["intent"] = classifier.evaluate()
    results["fusion"] = fusion.evaluate()
    if not skip_heavy:
        vision = CLIPCampusRetriever(kb)
        results["vision"] = vision.evaluate()

    scenarios = create_testing_scenarios()
    rows = []
    for _, scenario in scenarios.iterrows():
        query = str(scenario["input"])
        intent_result = classifier.predict(query)
        fusion_result = fusion.retrieve(FusionInput(text_query=query), top_k=3)
        predicted = fusion_result["top_prediction"]["name"] if fusion_result["top_prediction"] else ""
        top3 = [item["location"]["name"] for item in fusion_result["top_3"]]
        rows.append(
            {
                **scenario.to_dict(),
                "predicted_location": predicted,
                "predicted_intent": intent_result["intent"],
                "intent_confidence": intent_result["confidence"],
                "retrieval_confidence": fusion_result["confidence"],
                "top3": "; ".join(top3),
                "location_pass": scenario["expected_location"] in top3,
                "intent_pass": scenario["expected_intent"] == intent_result["intent"],
            }
        )
    test_results = pd.DataFrame(rows)
    test_results.to_csv(RESULTS_DIR / "testing_results.csv", index=False)
    save_error_analysis(test_results)

    plot_precision_recall()
    plot_word_cloud()
    plot_embedding_similarity(kb)
    save_model_comparison(results)
    with (RESULTS_DIR / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Smart Campus evaluations.")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip CLIP/Whisper-heavy evaluations.")
    args = parser.parse_args()
    print(json.dumps(run_all(skip_heavy=args.skip_heavy), indent=2))


if __name__ == "__main__":
    main()
