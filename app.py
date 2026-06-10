from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.knowledge_base import CampusKnowledgeBase
from src.utils import bootstrap_project_assets


st.set_page_config(
    page_title="Smart Campus Multimodal Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .main > div { padding-top: 1rem; }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #f7f9fb !important;
        color: #102033 !important;
    }
    [data-testid="stHeader"] {
        background: #10141a !important;
    }
    section[data-testid="stSidebar"] {
        background: #102033 !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stMain"], [data-testid="stMain"] *,
    .main, .main *, .block-container, .block-container *,
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #102033 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    textarea, input {
        background: #ffffff !important;
        color: #102033 !important;
        border: 1px solid #c7d3df !important;
    }
    .result-panel {
        border: 1px solid #d9e1ea;
        border-radius: 8px;
        padding: 1rem;
        background: white;
        color: #102033 !important;
    }
    .result-panel *, .result-panel p, .result-panel li, .result-panel div,
    .result-panel span, .result-panel h1, .result-panel h2, .result-panel h3 {
        color: #102033 !important;
    }
    div[data-testid="stDataFrame"] * { color: #102033 !important; }
    div[data-testid="stTable"] * { color: #102033 !important; }
    .small-muted { color: #607084; font-size: 0.92rem; }
    .confidence-label { font-weight: 650; color: #102033; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_knowledge_base():
    bootstrap_project_assets(generate_figures=False)
    return CampusKnowledgeBase()


@st.cache_resource(show_spinner="Loading text intent model...")
def get_intent_model():
    from src.text_pipeline import IntentClassifier

    return IntentClassifier()


@st.cache_resource(show_spinner="Loading Whisper speech model...")
def get_audio_model():
    from src.audio_pipeline import WhisperAudioPipeline

    return WhisperAudioPipeline(model_size="base")


@st.cache_resource(show_spinner="Loading CLIP image retriever...")
def get_vision_model(_kb: CampusKnowledgeBase):
    from src.image_pipeline import CLIPCampusRetriever

    return CLIPCampusRetriever(_kb)


@st.cache_resource(show_spinner="Loading multimodal fusion engine...")
def get_fusion_model(_kb: CampusKnowledgeBase):
    from src.fusion_model import MultiModalFusionEngine

    return MultiModalFusionEngine(_kb)


def save_uploaded_file(uploaded_file, suffix: str) -> Path:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getbuffer())
    temp.flush()
    return Path(temp.name)


def location_map(location: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"lat": location["coordinates"]["lat"], "lon": location["coordinates"]["lon"], "name": location["name"]}]
    )


def render_location(kb: CampusKnowledgeBase, location: dict, confidence: float, top3: list[dict], explanation: str) -> None:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.subheader(location["name"])
    st.caption(location["category"])
    st.write(location["description"])
    col_a, col_b, col_c = st.columns([1.1, 1, 1])
    with col_a:
        st.markdown("**Opening hours**")
        st.write(location["opening_hours"])
    with col_b:
        st.markdown("**Coordinates**")
        st.write(f"{location['coordinates']['lat']:.4f}, {location['coordinates']['lon']:.4f}")
    with col_c:
        st.markdown('<span class="confidence-label">Confidence</span>', unsafe_allow_html=True)
        st.progress(min(max(float(confidence), 0.0), 1.0))
        st.write(f"{confidence:.2f}")

    st.markdown("**Directions**")
    st.write(kb.direction_hint(location))

    st.markdown("**Events**")
    for event in location["events"]:
        st.write(f"- {event}")

    st.markdown("**Top-3 matches**")
    top_df = pd.DataFrame(
        [
            {"Rank": idx + 1, "Location": item["location"]["name"], "Score": round(item["score"], 3)}
            for idx, item in enumerate(top3)
        ]
    )
    st.dataframe(top_df, hide_index=True, use_container_width=True)
    st.markdown("**Retrieval explanation**")
    st.info(explanation)
    st.markdown("</div>", unsafe_allow_html=True)

    st.map(location_map(location), latitude="lat", longitude="lon", size=180, zoom=15)


def main() -> None:
    kb = get_knowledge_base()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("Smart Campus")
        st.caption("Multimodal tour and information assistant")
        input_mode = st.radio(
            "Input mode",
            ["Text only", "Image only", "Voice only", "Combined multimodal"],
            index=0,
        )
        st.divider()
        st.markdown("**Project capabilities**")
        st.write("CLIP + FAISS vision retrieval")
        st.write("Whisper speech-to-text")
        st.write("DistilBERT intent classification")
        st.write("SentenceTransformer semantic search")
        st.write("Masked MLP multimodal fusion")
        st.divider()
        st.markdown("**Model comparison**")
        st.caption("Fusion combines MLP scores with semantic KB retrieval. Text-only and image-only results are shown through the same top-3 interface.")

    st.title("Smart Campus Tour & Information Multi-Modal Chatbot")
    st.write("Ask about buildings, directions, opening hours, events, facilities, food, study spaces, and accessibility.")

    left, right = st.columns([1, 1])
    uploaded_image = None
    uploaded_audio = None
    text_query = ""

    with left:
        if input_mode in {"Image only", "Combined multimodal"}:
            uploaded_image = st.file_uploader("Upload campus building photo", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
        if input_mode in {"Voice only", "Combined multimodal"}:
            uploaded_audio = st.file_uploader("Upload voice query", type=["wav", "mp3", "m4a"])
            if hasattr(st, "audio_input"):
                recorded = st.audio_input("Or record a voice query")
                uploaded_audio = uploaded_audio or recorded
            if uploaded_audio:
                st.audio(uploaded_audio)
        if input_mode in {"Text only", "Combined multimodal"}:
            text_query = st.text_area("Enter your question", placeholder="Example: What events are happening at the Innovation Hub?")

        run = st.button("Ask campus assistant", type="primary", use_container_width=True)

    with right:
        st.subheader("Campus locations")
        df = kb.as_dataframe()
        st.dataframe(df[["name", "category", "opening_hours"]], hide_index=True, use_container_width=True)

    if run:
        from src.fusion_model import FusionInput

        modalities = []
        transcript = ""
        image_embedding = None
        top3 = []
        explanation_parts = []

        if uploaded_audio:
            audio_model = get_audio_model()
            modalities.append("audio")
            suffix = "." + uploaded_audio.name.split(".")[-1] if "." in uploaded_audio.name else ".wav"
            audio_path = save_uploaded_file(uploaded_audio, suffix=suffix)
            audio_result = audio_model.transcribe(audio_path)
            transcript = audio_result.get("transcript", "")
            if transcript:
                st.success(f"Transcript: {transcript}")
                explanation_parts.append("Whisper transcript embedded as a text signal.")
            else:
                st.warning(audio_result.get("warning", "No transcript was produced."))

        if text_query.strip():
            intent_model = get_intent_model()
            modalities.append("text")
            intent = intent_model.predict(text_query)
            st.info(f"Intent: {intent['intent']} | Confidence: {intent['confidence']:.2f} | Model: {intent['model']}")
            explanation_parts.append(f"Text intent classified as {intent['intent']}.")
        elif transcript:
            intent_model = get_intent_model()
            intent = intent_model.predict(transcript)
            st.info(f"Voice intent: {intent['intent']} | Confidence: {intent['confidence']:.2f}")
        else:
            intent = {"intent": "other", "confidence": 0.0}

        if uploaded_image:
            vision_model = get_vision_model(kb)
            modalities.append("image")
            image = Image.open(uploaded_image).convert("RGB")
            vision_result = vision_model.retrieve(image, top_k=3)
            image_embedding = vision_result["embedding"]
            top3 = vision_result["top_3"]
            explanation_parts.append(vision_result["explanation"])

        has_query_signal = bool(text_query.strip() or transcript.strip() or image_embedding is not None)
        if not has_query_signal:
            st.error(
                "No usable input was available. For voice mode, install Whisper dependencies or add a typed question."
            )
            st.stop()

        if input_mode == "Image only" and top3:
            result = {
                "top_prediction": top3[0]["location"],
                "confidence": top3[0]["score"],
                "top_3": top3,
                "explanation": "Image-only mode uses CLIP image-to-text retrieval.",
            }
        else:
            fusion_model = get_fusion_model(kb)
            fused = FusionInput(
                clip_embedding=image_embedding,
                text_query=text_query,
                transcript=transcript,
            )
            result = fusion_model.retrieve(fused, top_k=3)

        location = result["top_prediction"]
        if location:
            confidence = float(result.get("confidence", result.get("similarity_score", 0.0)))
            render_location(
                kb,
                location,
                confidence=confidence,
                top3=result["top_3"],
                explanation=" ".join(explanation_parts + [result.get("explanation", "")]),
            )
            kb.log_interaction(
                query=" ".join([text_query, transcript]).strip(),
                modalities=modalities,
                predicted_location=location["name"],
                confidence=confidence,
            )
            st.session_state.chat_history.append(
                {
                    "query": text_query or transcript or "[image query]",
                    "location": location["name"],
                    "confidence": confidence,
                }
            )
        else:
            st.error("I could not identify a campus record from the current input.")

    if st.session_state.chat_history:
        st.divider()
        st.subheader("Chat history")
        st.dataframe(pd.DataFrame(st.session_state.chat_history), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
