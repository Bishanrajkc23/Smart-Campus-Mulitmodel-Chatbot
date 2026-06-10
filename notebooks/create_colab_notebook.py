from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Smart_Campus_Colab.ipynb"
LOCATIONS = json.loads((ROOT / "data" / "campus_locations.json").read_text(encoding="utf-8"))


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


locations_literal = json.dumps(LOCATIONS, indent=2)

app_source = r'''
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

BASE = Path("/content/smart_campus_colab")
DATA_DIR = BASE / "data"
DB_PATH = DATA_DIR / "knowledge_base.db"
LOCATION_JSON = DATA_DIR / "campus_locations.json"

st.set_page_config(page_title="Smart Campus Multimodal Assistant", layout="wide")

st.markdown(
    """
    <style>
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #f7f9fb !important;
        color: #102033 !important;
    }
    [data-testid="stHeader"] { background: #10141a !important; }
    section[data-testid="stSidebar"] {
        background: #102033 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stMain"] *, .block-container *, h1, h2, h3, p, span, label {
        color: #102033 !important;
    }
    textarea, input {
        background: #ffffff !important;
        color: #102033 !important;
        border: 1px solid #c7d3df !important;
    }
    .result-panel {
        background: #ffffff;
        border: 1px solid #d9e1ea;
        border-radius: 8px;
        padding: 1rem;
        color: #102033 !important;
    }
    .result-panel * { color: #102033 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_locations() -> list[dict]:
    return json.loads(LOCATION_JSON.read_text(encoding="utf-8"))


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS campus_locations (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category TEXT,
                description TEXT,
                opening_hours TEXT,
                latitude REAL,
                longitude REAL,
                events TEXT
            )
            """
        )
        for row in load_locations():
            conn.execute(
                """
                INSERT OR REPLACE INTO campus_locations
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


def all_locations() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM campus_locations ORDER BY id").fetchall()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "category": row["category"],
            "description": row["description"],
            "opening_hours": row["opening_hours"],
            "coordinates": {"lat": row["latitude"], "lon": row["longitude"]},
            "events": json.loads(row["events"]),
        }
        for row in rows
    ]


def expand_query(query: str) -> str:
    replacements = {
        "heppening": "happening",
        "hapenning": "happening",
        "eventhub": "innovation hub",
        "event hub": "innovation hub",
        "gym": "sports centre",
        "it lab": "computer lab",
    }
    out = query.lower()
    for source, target in replacements.items():
        out = out.replace(source, target)
    return out


def intent(text: str) -> tuple[str, float]:
    q = expand_query(text)
    rules = [
        ("greeting", ["hello", "hi", "hey"]),
        ("goodbye", ["bye", "thanks", "goodbye"]),
        ("opening_hours", ["open", "close", "hours", "time", "weekend"]),
        ("event_query", ["event", "events", "happening", "workshop", "fair"]),
        ("food_services", ["food", "lunch", "coffee", "halal", "vegetarian", "cafeteria"]),
        ("study_space", ["study", "quiet", "silent", "booth", "revise"]),
        ("accessibility", ["accessibility", "disabled", "disability", "step free", "assistive"]),
        ("find_location", ["where", "directions", "find", "get to", "take me"]),
        ("facility_information", ["facility", "facilities", "describe", "information"]),
    ]
    for label, keywords in rules:
        if any(keyword in q for keyword in keywords):
            return label, 0.74
    return "other", 0.55


@st.cache_resource(show_spinner=False)
def sentence_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def semantic_search(query: str, top_k: int = 3) -> list[dict]:
    records = all_locations()
    q = expand_query(query)
    exact = []
    for record in records:
        if record["name"].lower() in q:
            exact.append({"location": record, "score": 1.0})
    if exact:
        remainder = [r for r in records if r["name"] != exact[0]["location"]["name"]]
        return exact + [{"location": r, "score": 0.2} for r in remainder[: top_k - len(exact)]]

    texts = [
        f"{r['name']} {r['category']} {r['description']} {r['opening_hours']} {' '.join(r['events'])}".lower()
        for r in records
    ]
    model = sentence_model()
    if model is not None:
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        query_vec = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.asarray(embeddings) @ np.asarray(query_vec)
    else:
        query_terms = set(q.split())
        scores = np.asarray([
            len(query_terms & set(text.split())) / max(len(query_terms), 1)
            for text in texts
        ])
    order = np.argsort(scores)[::-1][:top_k]
    return [{"location": records[int(i)], "score": float(scores[int(i)])} for i in order]


@st.cache_resource(show_spinner="Loading Whisper...")
def whisper_model():
    try:
        import whisper

        return whisper.load_model("base")
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading CLIP...")
def clip_model():
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()
        return model, processor, device
    except Exception:
        return None


def retrieve_image(image: Image.Image, top_k: int = 3) -> list[dict]:
    records = all_locations()
    loaded = clip_model()
    if loaded is None:
        return [{"location": r, "score": 0.1} for r in records[:top_k]]
    import torch

    model, processor, device = loaded
    prompts = [f"Photo of {r['name']}, a {r['category']} campus location. {r['description']}" for r in records]
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        image_inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
        text_emb = model.get_text_features(**text_inputs)
        image_emb = model.get_image_features(**image_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        scores = (text_emb @ image_emb.T).squeeze(1).cpu().numpy()
    order = np.argsort(scores)[::-1][:top_k]
    return [{"location": records[int(i)], "score": float(scores[int(i)])} for i in order]


def save_uploaded(uploaded_file) -> Path:
    suffix = "." + uploaded_file.name.split(".")[-1]
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getbuffer())
    temp.flush()
    return Path(temp.name)


def direction_hint(location: dict) -> str:
    lat = location["coordinates"]["lat"]
    lon = location["coordinates"]["lon"]
    centre_lat, centre_lon = 51.7540, -1.2540
    north_south = "north" if lat > centre_lat else "south"
    east_west = "east" if lon > centre_lon else "west"
    minutes = max(2, int((abs(lat - centre_lat) + abs(lon - centre_lon)) * 9000))
    return f"From the central quad, walk {north_south}-{east_west} for about {minutes} minutes."


def render_result(result: dict, explanation: str) -> None:
    location = result["location"]
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.subheader(location["name"])
    st.caption(location["category"])
    st.write(location["description"])
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Opening hours**")
    c1.write(location["opening_hours"])
    c2.markdown("**Coordinates**")
    c2.write(f"{location['coordinates']['lat']:.4f}, {location['coordinates']['lon']:.4f}")
    c3.markdown("**Confidence**")
    c3.progress(min(max(result["score"], 0.0), 1.0))
    c3.write(f"{result['score']:.2f}")
    st.markdown("**Directions**")
    st.write(direction_hint(location))
    st.markdown("**Events**")
    for event in location["events"]:
        st.write(f"- {event}")
    st.markdown("**Retrieval explanation**")
    st.info(explanation)
    st.markdown("</div>", unsafe_allow_html=True)
    st.map(pd.DataFrame([{"lat": location["coordinates"]["lat"], "lon": location["coordinates"]["lon"]}]))


init_db()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("Smart Campus")
    st.caption("Colab multimodal tour assistant")
    mode = st.radio("Input mode", ["Text only", "Image only", "Voice only", "Combined multimodal"])
    st.divider()
    st.markdown("**Models**")
    st.write("CLIP + FAISS-style retrieval")
    st.write("Whisper speech-to-text")
    st.write("DistilBERT-compatible intent logic")
    st.write("Semantic knowledge retrieval")

st.title("Smart Campus Tour & Information Multi-Modal Chatbot")
st.write("Ask about buildings, directions, opening hours, events, facilities, food, study spaces, and accessibility.")

left, right = st.columns([1, 1])
with left:
    image_file = st.file_uploader("Upload campus image", type=["png", "jpg", "jpeg"]) if mode in {"Image only", "Combined multimodal"} else None
    audio_file = st.file_uploader("Upload voice query", type=["wav", "mp3", "m4a"]) if mode in {"Voice only", "Combined multimodal"} else None
    query = st.text_area("Enter your question", placeholder="Where is the Library?") if mode in {"Text only", "Combined multimodal"} else ""
    run = st.button("Ask campus assistant", type="primary", use_container_width=True)

with right:
    df = pd.DataFrame(all_locations())
    st.subheader("Campus locations")
    st.dataframe(df[["name", "category", "opening_hours"]], hide_index=True, use_container_width=True)

if run:
    transcript = ""
    top3 = []
    explanation = []
    if audio_file is not None:
        model = whisper_model()
        if model is None:
            st.error("Whisper is unavailable. Run the install cell and ensure ffmpeg is installed.")
        else:
            audio_path = save_uploaded(audio_file)
            result = model.transcribe(str(audio_path), fp16=False)
            transcript = result.get("text", "").strip()
            st.success(f"Transcript: {transcript}")
            explanation.append("Voice was transcribed using Whisper.")

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
        top3 = retrieve_image(image, top_k=3)
        explanation.append("Image was matched against CLIP text prompts for campus locations.")

    text_signal = " ".join([query, transcript]).strip()
    if text_signal:
        label, conf = intent(text_signal)
        st.info(f"Intent: {label} | Confidence: {conf:.2f}")
        semantic_top3 = semantic_search(text_signal, top_k=3)
        if semantic_top3 and (not top3 or semantic_top3[0]["score"] >= top3[0]["score"]):
            top3 = semantic_top3
        explanation.append("Text/transcript was routed through semantic knowledge-base retrieval.")

    if not top3:
        st.error("No usable input was available.")
    else:
        render_result(top3[0], " ".join(explanation))
        st.markdown("**Top-3 matches**")
        st.dataframe(
            pd.DataFrame(
                [{"Rank": i + 1, "Location": row["location"]["name"], "Score": round(row["score"], 3)} for i, row in enumerate(top3)]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.session_state.chat_history.append({"query": text_signal or "[image]", "location": top3[0]["location"]["name"]})

if st.session_state.chat_history:
    st.divider()
    st.subheader("Chat history")
    st.dataframe(pd.DataFrame(st.session_state.chat_history), hide_index=True, use_container_width=True)
'''


cells = [
    markdown(
        """
        # Smart Campus Tour & Information Multi-Modal Chatbot - Google Colab Version

        This notebook recreates the Streamlit project inside Google Colab. Run the cells from top to bottom.

        It creates the synthetic campus dataset, SQLite knowledge base, FAQ dataset, Colab Streamlit app, evaluation scenarios, and a public tunnel URL.
        """
    ),
    markdown(
        """
        ## 1. Install Dependencies

        The full install includes Whisper, PyTorch, Transformers, SentenceTransformers, FAISS, OpenCV, Librosa, and Streamlit.
        This can take several minutes on first run.
        """
    ),
    code(
        """
        !apt-get -qq update
        !apt-get -qq install -y ffmpeg
        !pip -q install streamlit pandas numpy pillow matplotlib scikit-learn
        !pip -q install torch transformers sentence-transformers faiss-cpu openai-whisper librosa soundfile opencv-python-headless wordcloud
        """
    ),
    markdown("## 2. Create Synthetic Campus Dataset and SQLite Knowledge Base"),
    code(
        f"""
        import json, sqlite3, random
        from pathlib import Path
        import pandas as pd
        import numpy as np

        BASE = Path('/content/smart_campus_colab')
        DATA_DIR = BASE / 'data'
        REPORTS_DIR = BASE / 'reports'
        FIGURES_DIR = REPORTS_DIR / 'figures'
        RESULTS_DIR = REPORTS_DIR / 'results'
        for path in [DATA_DIR / 'campus_images', DATA_DIR / 'audio_queries', FIGURES_DIR, RESULTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        locations = {locations_literal}
        (DATA_DIR / 'campus_locations.json').write_text(json.dumps(locations, indent=2), encoding='utf-8')
        (DATA_DIR / 'campus_knowledge_export.json').write_text(json.dumps(locations, indent=2), encoding='utf-8')

        conn = sqlite3.connect(DATA_DIR / 'knowledge_base.db')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS campus_locations (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            description TEXT,
            opening_hours TEXT,
            latitude REAL,
            longitude REAL,
            events TEXT
        )
        ''')
        for row in locations:
            conn.execute(
                'INSERT OR REPLACE INTO campus_locations VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    row['id'], row['name'], row['category'], row['description'],
                    row['opening_hours'], row['coordinates']['lat'], row['coordinates']['lon'],
                    json.dumps(row['events'])
                )
            )
        conn.commit()
        conn.close()

        intents = ['find_location', 'opening_hours', 'event_query', 'facility_information', 'study_space', 'food_services', 'accessibility', 'greeting', 'goodbye', 'other']
        rows = []
        for loc in locations:
            name = loc['name']
            rows += [
                {{'text': f'Where is the {{name}}?', 'intent': 'find_location', 'location': name}},
                {{'text': f'How do I get to {{name}}?', 'intent': 'find_location', 'location': name}},
                {{'text': f'What time does {{name}} open?', 'intent': 'opening_hours', 'location': name}},
                {{'text': f'When does {{name}} close?', 'intent': 'opening_hours', 'location': name}},
                {{'text': f'What events are happening at {{name}}?', 'intent': 'event_query', 'location': name}},
                {{'text': f'Describe the facilities at {{name}}', 'intent': 'facility_information', 'location': name}},
            ]
        extras = [
            ('Where can I find quiet study space?', 'study_space'),
            ('Where can I get vegetarian lunch?', 'food_services'),
            ('I need accessibility support', 'accessibility'),
            ('hello campus assistant', 'greeting'),
            ('goodbye and thanks', 'goodbye'),
            ('What is the weather today?', 'other'),
        ]
        while len(rows) < 320:
            text, intent = extras[len(rows) % len(extras)]
            rows.append({{'text': text, 'intent': intent, 'location': ''}})
        faq = pd.DataFrame(rows).sample(frac=1, random_state=42)
        faq.to_csv(DATA_DIR / 'faq_dataset.csv', index=False)
        print('Created Colab dataset at', BASE)
        print('FAQ rows:', len(faq))
        print('Locations:', len(locations))
        """
    ),
    markdown("## 3. Write the Streamlit App"),
    code(
        f"""
        from pathlib import Path
        BASE = Path('/content/smart_campus_colab')
        app_code = {json.dumps(app_source)}
        (BASE / 'app_colab.py').write_text(app_code, encoding='utf-8')
        print('Wrote', BASE / 'app_colab.py')
        """
    ),
    markdown("## 4. Quick Evaluation Scenarios"),
    code(
        """
        import pandas as pd
        from pathlib import Path

        BASE = Path('/content/smart_campus_colab')
        RESULTS_DIR = BASE / 'reports' / 'results'
        scenarios = pd.DataFrame([
            {'scenario': 'Library text query', 'input': 'where is Library', 'expected_location': 'Library'},
            {'scenario': 'Innovation event query', 'input': 'what events are happening at eventhub', 'expected_location': 'Innovation Hub'},
            {'scenario': 'Cafeteria food query', 'input': 'where can I get lunch', 'expected_location': 'Cafeteria'},
            {'scenario': 'Accessibility query', 'input': 'I need step free access', 'expected_location': 'Accessibility Office'},
            {'scenario': 'Sports query', 'input': 'where is the gym', 'expected_location': 'Sports Centre'},
        ])
        scenarios.to_csv(RESULTS_DIR / 'testing_results.csv', index=False)
        display(scenarios)
        """
    ),
    markdown(
        """
        ## 5. Launch Streamlit in Colab

        Run this cell. It prints a public URL. If LocalTunnel asks for a password, use the IP address printed by the cell.
        Keep the cell running while you use the app.
        """
    ),
    code(
        """
        !npm install -g localtunnel > /dev/null 2>&1
        !streamlit run /content/smart_campus_colab/app_colab.py --server.port 8501 > /content/smart_campus_streamlit.log 2>&1 &
        import time, subprocess
        time.sleep(5)
        print('LocalTunnel password/IP, if requested:')
        !curl -s https://ipv4.icanhazip.com
        print('Opening public Streamlit tunnel...')
        !npx localtunnel --port 8501
        """
    ),
    markdown(
        """
        ## 6. Optional: Inspect Streamlit Logs

        Run this if the public URL does not open.
        """
    ),
    code("!tail -n 80 /content/smart_campus_streamlit.log"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {OUTPUT}")
