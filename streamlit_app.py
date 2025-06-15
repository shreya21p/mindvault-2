import streamlit as st
from mood_engine import (
    chat_with_bot,
    plot_mood_trends,
    mood_wordcloud,
    recall_entries_by_day,
    search_similar_memory
)

st.set_page_config(page_title=" MindVault", layout="centered")

# === Theme Colors per Persona ===
themes = {
    "Coach": {
        "bg": "#EEEFE0",
        "header": "#819A91",
        "subheader": "#A7C1A8",
        "button": "#D1D8BE",
        "hover": "#A7C1A8"
    },
    "Listener": {
        "bg": "#DCD7C9",
        "header": "#2C3930",
        "subheader": "#3F4F44",
        "button": "#A27B5C",
        "hover": "#3F4F44"
    },
    "Cheerleader": {
        "bg": "#F4F8D3",
        "header": "#F7CFD8",
        "subheader": "#8E7DBE",
        "button": "#A6D6D6",
        "hover": "#F7CFD8"
    }
}

# === Sidebar: Persona Selection ===
st.sidebar.markdown("## 🎭 Choose Persona")
persona = st.sidebar.radio("How should MindVault respond?", ["Listener", "Coach", "Cheerleader"])
colors = themes[persona]

# === Inject Custom CSS ===
st.markdown(f"""
    <style>
        html, body {{
            background: linear-gradient(to bottom right, {colors['bg']}, #ffffff);
        }}
        .stApp {{
            background: linear-gradient(to bottom right, {colors['bg']}, #ffffff);
            font-family: 'Segoe UI', sans-serif;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {colors['bg']}55;
            border-right: 1px solid {colors['header']}30;
        }}
        h1, h2, h3 {{
            color: {colors['header']};
        }}
        .stButton>button {{
            background-color: {colors['button']};
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            border: none;
            transition: 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {colors['hover']};
            color: black;
        }}
        .chatbox {{
            background-color: #ffffff;
            border-left: 6px solid {colors['subheader']};
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            box-shadow: 0 1px 5px rgba(0,0,0,0.1);
            color: #333;
        }}
        textarea, input {{
            border-radius: 10px !important;
        }}
        footer, .css-1v0mbdj.e1f1d6gn1 {{
            text-align: center;
            font-size: 13px;
            color: #888;
            margin-top: 2rem;
        }}
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown(f"<h1 style='text-align: center;'> MindVault</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: gray;'>Your Mental Health Chat Companion ({persona})</p>", unsafe_allow_html=True)
st.caption(" Powered by Gemini + ChromaDB + Streamlit")

# === Chat Section ===
if persona == "Coach":
    endpoint = "/coach"
elif persona == "Cheerleader":
    endpoint = "/cheer"
else:
    endpoint = "/listener"

st.markdown("### 💬 What's on your mind?")
user_input = st.text_area("", placeholder="Write your thoughts here...", height=100)

if st.button("💌 Send"):
    if user_input.strip():
        response = chat_with_bot(user_input, mode=endpoint)
        st.markdown(f"<div class='chatbox'><b>MindVault:</b> {response}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please write something before sending!")

# === Mood Tools ===
st.markdown("---")
st.subheader("📊 Mood Tools")

col1, col2 = st.columns(2)

with col1:
    if st.button("📈 View Mood Trends"):
        plot_mood_trends()

with col2:
    if st.button("☁️ Generate Word Cloud"):
        mood_wordcloud()

# === Memory Recall ===

# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align:center; font-size: 13px; color: gray;'> Built for self-reflection, powered by empathy and AI.</p>", unsafe_allow_html=True)
