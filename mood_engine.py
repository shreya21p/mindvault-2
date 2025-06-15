# mood_engine.py (Streamlit + Gemini + FAISS version)

import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import uuid
import pickle
from datetime import datetime
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# ========================
# API KEY HANDLING
# ========================
def get_google_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except:
        return os.environ.get("GOOGLE_API_KEY")

GOOGLE_API_KEY = get_google_api_key()
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found.")

genai.configure(api_key=GOOGLE_API_KEY)

# ========================
# Gemini Model
# ========================
text_model = genai.GenerativeModel("models/gemini-1.5-flash")

# ========================
# Embedding Model (FAISS)
# ========================
embedder = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384  # size for MiniLM

# ========================
# FAISS Index Setup
# ========================
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"

if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
else:
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = []

# ========================
# Persona Modes
# ========================
COACH_PROMPT = "You are a tough love life coach. Give direct, actionable advice."
LISTENER_PROMPT = "You are a warm, empathetic listener. Validate feelings without judgment."
CHEERLEADER_PROMPT = "You're an energetic cheerleader! Respond with enthusiasm and emojis!"
persona_instruction = LISTENER_PROMPT

# ========================
# Emotion Detection
# ========================
def detect_emotion(text):
    prompt = f"Classify the emotion in the following text in one word (like happy, sad, angry, anxious, excited):\n{text}"
    try:
        response = text_model.generate_content(prompt)
        return response.text.strip().lower()
    except:
        return "neutral"

# ========================
# Store Journal Entry
# ========================
def store_message(text, emotion=None):
    global faiss_index, metadata

    if emotion is None:
        emotion = detect_emotion(text)

    embedding = embedder.encode([text])
    faiss_index.add(np.array(embedding).astype("float32"))

    entry = {
        "text": text,
        "emotion": emotion,
        "timestamp": datetime.now().isoformat()
    }
    metadata.append(entry)

    # Save updated index and metadata
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    # Also log into plain text journal
    with open("journal.txt", "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {emotion}: {text}\n")

# ========================
# Chat Function
# ========================
def chat_with_bot(user_input, mode="/listener"):
    global persona_instruction

    if user_input.strip().startswith("/"):
        if "coach" in user_input:
            persona_instruction = COACH_PROMPT
            return "Switched to Coach mode 🏋️"
        elif "cheer" in user_input:
            persona_instruction = CHEERLEADER_PROMPT
            return "Switched to Cheerleader mode 🎉"
        else:
            persona_instruction = LISTENER_PROMPT
            return "Switched to Listener mode 🧘"

    mood = detect_emotion(user_input)
    store_message(user_input, mood)

    prompt = f"""{persona_instruction}
    User ({mood} mood): {user_input}
    Respond compassionately:"""

    try:
        response = text_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "I'm having trouble responding right now."

# ========================
# Mood Wordcloud
# ========================
def mood_wordcloud():
    try:
        with open("journal.txt", "r") as f:
            entries = f.read().splitlines()

        clean_words = []
        for line in entries:
            line = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", line)
            line = re.sub(r"^\s*\w+:\s*", "", line)
            clean_words.extend(line.split())

        if not clean_words:
            st.warning("No text entries found for word cloud generation.")
            return

        word_freq = Counter(clean_words)
        wc = WordCloud(background_color='white', colormap='viridis',
                       width=800, height=400, random_state=42).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt.gcf())
        plt.clf()

    except FileNotFoundError:
        st.error("No journal file found.")
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")

# ========================
# Mood Trends
# ========================
def plot_mood_trends():
    try:
        with open("journal.txt", "r") as f:
            entries = f.read().splitlines()

        moods = []
        for line in entries:
            match = re.match(r"\[\d{2}:\d{2}:\d{2}\]\s*(\w+):", line)
            if match:
                moods.append(match.group(1).lower())

        if not moods:
            st.warning("No mood data found.")
            return

        mood_counts = Counter(moods)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(mood_counts.keys(), mood_counts.values(), color='#6C63FF')
        ax.set_title("Mood Frequency Over Time")
        ax.set_xlabel("Mood")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    except FileNotFoundError:
        st.error("No journal found.")
    except Exception as e:
        st.error(f"Error: {e}")

# ========================
# Search Similar Memory
# ========================
def search_similar_memory(query_text, k=3):
    try:
        query_vec = embedder.encode([query_text]).astype("float32")
        distances, indices = faiss_index.search(query_vec, k)

        st.subheader("🔍 Similar Memories")
        for i in indices[0]:
            if i < len(metadata):
                entry = metadata[i]
                st.write(f"🕒 {entry['timestamp']} ({entry['emotion']}): {entry['text']}")
    except Exception as e:
        st.error(f"Error retrieving similar memories: {e}")
