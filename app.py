# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import json
import base64

# ------------------------
# Config
# ------------------------
TFIDF_VECTORIZER_PATH = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = "tfidf_matrix.npz"
NUMERIC_SCALER_PATH = "numeric_scaler.pkl"
NUMERIC_MATRIX_PATH = "numeric_matrix.npz"
INDEX_PATH = "index.parquet"
SCHEMA_PATH = "schema.json"

TEXT_WEIGHT = 0.85
NUMERIC_WEIGHT = 0.15
ARTIST_BOOST = 0.1

BACKGROUND_IMAGE = "bg.jpg"   # <-- put any background image in project folder


# ------------------------
# Background Function
# ------------------------
def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}

        /* Hide top black menu bar and Streamlit footer */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True
    )


# ------------------------
# Load Artifacts
# ------------------------
@st.cache_resource
def load_artifacts():
    vec = joblib.load(TFIDF_VECTORIZER_PATH)
    X_text = load_npz(TFIDF_MATRIX_PATH)

    try:
        scaler = joblib.load(NUMERIC_SCALER_PATH)
        X_num = load_npz(NUMERIC_MATRIX_PATH)
    except:
        scaler, X_num = None, None

    index_df = pd.read_parquet(INDEX_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    return vec, X_text, scaler, X_num, index_df, schema


vec, X_text, scaler, X_num, index_df, schema = load_artifacts()


# ------------------------
# Recommender Functions
# ------------------------
def recommend_by_index(idx, top_n=10):
    sim_text = cosine_similarity(X_text[idx], X_text).flatten()
    if X_num is not None:
        sim_num = cosine_similarity(X_num[idx], X_num).flatten()
    else:
        sim_num = np.zeros_like(sim_text)

    sim = TEXT_WEIGHT * sim_text + NUMERIC_WEIGHT * sim_num

    query_artist = index_df.iloc[idx][schema["artist"]]
    artist_mask = index_df[schema["artist"]] == query_artist
    sim = sim + ARTIST_BOOST * artist_mask.astype(float)

    sim[idx] = -1
    rec_idx = np.argsort(sim)[::-1][:top_n]
    return index_df.iloc[rec_idx].assign(similarity=sim[rec_idx])


def recommend_song(title, top_n=10):
    mask = index_df[schema["track_name"]].str.lower().str.contains(title.lower(), na=False)
    if not mask.any():
        return None
    idx = mask.idxmax()
    return recommend_by_index(idx, top_n)


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="🎵 Music Recommender", layout="wide")
add_bg(BACKGROUND_IMAGE)

# Custom CSS
st.markdown(
    """
    <style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 6px #000000;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 25px;
    }
    /* Short, modern search bar */
    div[data-baseweb="input"] {
        max-width: 400px;
        margin: auto;
        border-radius: 25px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.25);
        overflow: hidden;
    }
    input {
        text-align: center;
        font-size: 16px !important;
        border-radius: 25px !important;
    }
    .stDataFrame { background: rgba(255,255,255,0.9); border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="title"> Music Recommendation Dashboard </div>', unsafe_allow_html=True)
# st.title("")
st.markdown('<div class="subtitle"> Discover your next favorite track </div>', unsafe_allow_html=True)
# st.header("")
# Search bar
song_query = st.text_input(" ", placeholder="🔍 What do you want to listen to ?")

# Results
if song_query:
    results = recommend_song(song_query, top_n=10)

    if results is None:
        st.error(f"No match found for **{song_query}**")
    else:
        st.success(f"Top recommendations for: **{song_query}**")
        st.dataframe(results.reset_index(drop=True)[[schema["track_name"], schema["artist"], "similarity"]])
