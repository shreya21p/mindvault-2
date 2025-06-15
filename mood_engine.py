# mood_engine.py

import google.generativeai as genai
import chromadb
from chromadb import PersistentClient
#from kaggle_secrets import UserSecretsClient
import streamlit as st
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

import uuid
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

import os
import google.generativeai as genai

def get_google_api_key():
    try:
        # Try Kaggle secrets (only works inside Kaggle notebooks)
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GOOGLE_API_KEY")
    except ImportError:
        pass

    try:
        # Try Streamlit secrets
        import streamlit as st
        return st.secrets["GOOGLE_API_KEY"]
    except:
        pass

    # Fallback: system environment variable (e.g., for local dev)
    return os.environ.get("GOOGLE_API_KEY")

# 🔐 Final setup
GOOGLE_API_KEY = get_google_api_key()

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in any source (Kaggle, Streamlit, or env var)")

genai.configure(api_key=GOOGLE_API_KEY)


# Load generative and embedding models
text_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Persistent ChromaDB vector database
chroma_client = PersistentClient(path="./chroma_db_2")


# Reset and create memory collection
try:
    chroma_client.delete_collection("memories")
except:
    pass

collection = chroma_client.create_collection(name="memories")

# Local log for moods
mood_log = []

# Few-shot instruction styles
COACH_PROMPT = "You are a tough love life coach. Give direct, actionable advice."
LISTENER_PROMPT = "You are a warm, empathetic listener. Validate feelings without judgment."
CHEERLEADER_PROMPT = "You're an energetic cheerleader! Respond with enthusiasm and emojis!"
persona_instruction = LISTENER_PROMPT


def detect_emotion(text):
    """Rough emotion detector using Gemini"""
    prompt = f"Classify the emotion in the following text in one word (like happy, sad, angry, anxious, excited):\n{text}"
    try:
        response = text_model.generate_content(prompt)
        return response.text.strip().lower()
    except:
        return "neutral"


def store_message(text, emotion=None):
    """Store journal entry with vector embedding and emotion"""
    if emotion is None:
        emotion = detect_emotion(text)

    entry_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    try:
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )["embedding"]

        collection.add(
            documents=[text],
            metadatas=[{"timestamp": timestamp, "emotion": emotion}],
            embeddings=[embedding],
            ids=[entry_id]
        )
    except Exception as e:
        print(f"⚠️ Error while adding document: {e}")
        return False

    mood_log.append({
        "text": text,
        "emotion": emotion,
        "timestamp": timestamp,
        "chroma_id": entry_id
    })
    return True
    


def chat_with_bot(user_input, mode="/listener"):
    """Generate chatbot reply and store journal"""
    global persona_instruction

    # Handle mode switching
    if mode == "/coach":
        persona_instruction = COACH_PROMPT
    elif mode == "/cheer":
        persona_instruction = CHEERLEADER_PROMPT
    else:
        persona_instruction = LISTENER_PROMPT

    # If user input is an internal command (e.g. /coach), just return switch confirmation
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

    # Detect emotion
    emotion = detect_emotion(user_input)
    store_message(user_input, emotion)

    prompt = f"""{persona_instruction}
    User ({emotion} mood): {user_input}
    Respond compassionately:"""

    try:
        response = text_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "I'm having trouble responding right now"
    
def store_message(message, mood):
    with open("journal.txt", "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {mood}: {message}\n")


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
            st.warning("No mood data found to plot trends.")
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
        st.error("No journal file found. Please write something first.")
    except Exception as e:
        st.error(f"Error generating mood trends: {e}")

import re
def mood_wordcloud():
    try:
        with open("journal.txt", "r") as f:
            entries = f.read().splitlines()

        clean_words = []
        for line in entries:
            # Remove timestamps like [02:37:43]
            line = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", line)
            # Remove mood label before the colon (e.g., anxious: message)
            line = re.sub(r"^\s*\w+:\s*", "", line)
            clean_words.extend(line.split())

        if not clean_words:
            st.warning("No text entries found for word cloud generation.")
            return

        word_freq = Counter(clean_words)
        wc = WordCloud(
            background_color='white',
            colormap='viridis',
            width=800,
            height=400,
            random_state=42
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")

        st.pyplot(plt.gcf())
        plt.clf()

    except FileNotFoundError:
        st.error("No journal file found. Please write something first.")
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")


def recall_entries_by_day(date_str):
    """Print all entries for a specific day (YYYY-MM-DD)"""
    found = False
    for entry in mood_log:
        if entry["timestamp"][:10] == date_str:
            print(f"[{entry['timestamp'][11:19]}] {entry['emotion']}: {entry['text']}")
            found = True
    if not found:
        print("No entries found for that date.")


def search_similar_memory(query_text, threshold=0.4):
    """Retrieve similar journal entries based on embeddings"""
    try:
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query_text,
            task_type="retrieval_query"
        )["embedding"]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        print("Similar Memories:")
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            print(f"{meta['timestamp']} ({meta['emotion']}): {doc}")

    except Exception as e:
        print(f"Error retrieving similar memories: {e}")
