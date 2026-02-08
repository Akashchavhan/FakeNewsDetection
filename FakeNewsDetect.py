import streamlit as st
import re
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import pipeline
from serpapi import GoogleSearch
import plotly.graph_objects as go

st.set_page_config(page_title="Fake News Verifier", layout="centered")
st.title("📰 Advanced Fake News Detector")

# ---------------- SUMMARIZER (FIXED) ----------------
@st.cache_resource(show_spinner=True)
def load_summarizer():
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-small"
    )

summarizer = load_summarizer()

# ---------------- TRUSTED SOURCES ----------------
TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "ndtv.com", "cnn.com", "indiatoday.in",
    "thehindu.com", "timesofindia.indiatimes.com", "hindustantimes.com",
    "aljazeera.com", "apnews.com", "foxnews.com", "washingtonpost.com",
    "nytimes.com", "economictimes.indiatimes.com", "scroll.in", "bbc.co.uk",
    "cbc.ca", "theguardian.com", "cnbc.com", "dw.com", "npr.org",
    "bbcnews.com", "news18.com", "thewire.in", "indianexpress.com"
]

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    return text.lower()

# ---------------- TRUST CHECK ----------------
def is_trusted_source(url):
    domain = urlparse(url).netloc.replace("www.", "").lower()
    return any(trusted_domain in domain for trusted_domain in TRUSTED_SOURCES)

# ---------------- SEARCH NEWS ----------------
def search_news(query, max_results=5):
    matches = []
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": st.secrets["SERPAPI_KEY"],
            "num": max_results
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        for result in results.get('organic_results', [])[:max_results]:
            url = result.get('link')
            if not url or not url.startswith("http"):
                continue

            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text_snippet = ' '.join([p.get_text() for p in paragraphs[:5]])

                if text_snippet.strip():
                    matches.append((url, text_snippet))

                time.sleep(1)

            except Exception:
                continue

    except Exception as e:
        st.error(f"❌ Search error: {e}")

    return matches

# ---------------- EVALUATE NEWS ----------------
def evaluate_news(query):
    matches = search_news(query)
    trusted_hits = []
    total_hits = len(matches)

    for url, snippet in matches:
        if is_trusted_source(url):
            trusted_hits.append((url, snippet))

    if total_hits == 0:
        return {
            "status": "Unable to verify",
            "confidence": 0.0,
            "matches": [],
            "summary": "No data found to summarize."
        }

    confidence = round((len(trusted_hits) / total_hits) * 100, 2)
    status = "REAL" if confidence >= 50 else "FAKE"

    all_text = ' '.join(snippet for _, snippet in trusted_hits or matches)
    clean_input = clean_text(all_text)

    # ----------- SUMMARIZATION FIX -----------
    if len(clean_input.split()) > 20:
        try:
            summary = summarizer(
                f"summarize: {clean_input[:1024]}",
                max_length=120,
                min_length=30,
                do_sample=False
            )[0]['generated_text']
        except Exception as e:
            summary = f"Failed to summarize: {e}"
    else:
        summary = "Not enough data to generate a reliable summary."

    return {
        "status": status,
        "confidence": confidence,
        "matches": trusted_hits,
        "summary": summary
    }

# ---------- Animated Donut ----------
def animated_confidence_donut(confidence, status):
    color = "#2ecc71" if status == "REAL" else "#e74c3c"
    placeholder = st.empty()

    for val in range(0, int(confidence) + 1, 2):
        fig = go.Figure(data=[
            go.Pie(
                values=[val, 100 - val],
                hole=0.7,
                marker_colors=[color, "rgba(200,200,200,0.15)"],
                textinfo="none",
                sort=False,
                direction="clockwise"
            )
        ])

        fig.update_layout(
            showlegend=False,
            height=350,
            margin=dict(t=40, b=20, l=10, r=10),
            annotations=[dict(
                text=f"<b>{val}%</b><br>Confidence",
                x=0.5, y=0.5, font_size=22, showarrow=False
            )]
        )

        placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        time.sleep(0.03)

# ---------------- MAIN APP ----------------
query = st.text_input("Enter a news headline to verify:")

if query:
    try:
        with st.spinner("Analyzing..."):
            result = evaluate_news(query)

        st.subheader("Prediction")
        st.write(f"**Status**: {result['status']}")
        st.write(f"**Confidence**: {result['confidence']}%")

        st.subheader("Summary")
        st.write(result['summary'])

        st.subheader("Trusted Sources")
        if result['matches']:
            for idx, (url, snippet) in enumerate(result['matches'], 1):
                st.markdown(f"**{idx}.** [{url}]({url})")
                st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))
        else:
            st.info("No matches found on trusted news sites.")

        st.subheader("Confidence Visualization")
        animated_confidence_donut(result['confidence'], result['status'])

    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
