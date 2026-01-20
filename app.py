import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# --- Load SOP files dynamically ---
def load_sops():
    sops = {}
    for filename in os.listdir():
        if filename.endswith(".txt"):
            with open(filename, "r") as f:
                sops[filename.replace(".txt", "")] = f.read()
    return sops

sops = load_sops()

# --- Train or load TF-IDF vectorizer ---
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("sop_matrix.pkl", "rb") as f:
        X = pickle.load(f)
except:
    corpus = list(sops.values())
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    # Save for reuse
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("sop_matrix.pkl", "wb") as f:
        pickle.dump(X, f)

# --- Function to find SOP ---
def find_sop(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    best_match = similarities.argmax()
    sop_name = list(sops.keys())[best_match]
    return sop_name, sops[sop_name]

# --- Streamlit UI ---
st.title("Krishnendu's Help Support (TF-IDF Prototype)")

# Text input
user_input = st.text_input("Describe your issue:")

# Dropdown selection
selected_sop = st.selectbox("Or choose SOP directly:", ["-- Select SOP --"] + list(sops.keys()))

# Logic
if user_input:
    sop_name, sop_text = find_sop(user_input)
    st.write(f"Suggested SOP (by query): **{sop_name}**")
    st.text(sop_text)
elif selected_sop != "-- Select SOP --":
    sop_text = sops[selected_sop]
    st.write(f"Selected SOP: **{selected_sop}**")
    st.text(sop_text)
else:
    st.info("Please describe your issue or select a SOP from the dropdown.")
