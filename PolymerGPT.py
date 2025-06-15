# -*- coding: utf-8 -*-
"""
Created on Tue May  6 23:25:07 2025

@author: Shivraj.Karewar
"""

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Polymer Similarity Search", layout="wide")

st.title("Polymer Similarity & Property Explorer")

# Sidebar inputs
st.sidebar.header("Configuration")
parquet_path = st.sidebar.text_input("Path to Parquet file", value="polyOne_bb.parquet")
model_name = st.sidebar.text_input("polyBERT Model", value="kuelumbus/polyBERT")
similarity_threshold = st.sidebar.slider("Cosine Similarity Threshold", 0.0, 1.0, 0.8)
tg_window = st.sidebar.slider("Tg Window (°C)", 0, 200, 100)

# Input query SMILES
query_smiles = st.text_input("Query Polymer pSMILES (e.g., [*]CC[*]):", value="")
if not query_smiles:
    st.info("Please enter a polymer pSMILES string to search for similar polymers.")
    st.stop()

# Cached data loader
def load_data(path):
    df = pd.read_parquet(path)
    df = df.dropna(subset=["smiles"])
    return df

# Cached model loader
def load_model(name):
    return SentenceTransformer(name)

# Load data and model
with st.spinner("Loading data..."):
    try:
        df = load_data(parquet_path)
        df = df[:10000]
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

with st.spinner("Loading model..."):
    try:
        model = load_model(model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Show overview
st.markdown(f"**Dataset:** {df.shape[0]} polymers loaded")
st.dataframe(df.head())

# Compute embeddings (cached)
@st.cache_resource(show_spinner=False)
def compute_embeddings(smiles_list):
    return model.encode(smiles_list)

embeddings = compute_embeddings(df['smiles'].tolist())

# Encode query polymer
t = model.encode([query_smiles])
query_embedding = t[0].reshape(1, -1)

# Attempt to get query Tg if present
def get_query_tg(df, query_smiles):
    matches = df[df['smiles'] == query_smiles]
    if not matches.empty and 'Tg' in matches.columns:
        return matches['Tg'].iloc[0]
    return None

query_tg = get_query_tg(df, query_smiles)

# Similarity search
results = []
for i, row in df.iterrows():
    sim = cosine_similarity(query_embedding, embeddings[i].reshape(1, -1))[0][0]
    if sim < similarity_threshold:
        continue
    # Filter by Tg window only if query Tg known and row Tg present
    if query_tg is not None and pd.notna(row.get('Tg')):
        if abs(row['Tg'] - query_tg) > tg_window:
            continue
    entry = row.to_dict()
    entry['similarity'] = sim
    results.append(entry)

# Sort results by similarity
results_sorted = sorted(results, key=lambda x: -x['similarity'])

# Display results
st.header("Similar Polymers")
if results_sorted:
    df_res = pd.DataFrame(results_sorted)
    st.dataframe(df_res.reset_index(drop=True))
else:
    st.info("No similar polymers found with the given threshold.")
