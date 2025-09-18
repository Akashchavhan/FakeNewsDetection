import streamlit as st
st.set_page_config(page_title="Fake News Verifier", layout="centered")
st.title("📰 Advanced Fake News Detector")

import re
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from transformers import pipeline
from serpapi import GoogleSearch
import plotly.graph_objects as go  # For improved confidence visualization

# --- Caching the model to avoid reloading ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# --- Trusted sources list ---
TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "ndtv.com", "cnn.com", "indiatoday.in",
    "thehindu.com", "timesofindia.indiatimes.com", "hindustantimes.com",
    "aljazeera.com", "apnews.com", "theguardian.com", "washingtonpost.com"
]

# --- Text cleaning ---
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    return text.lower()

# --- Trusted source checker ---
def is_trusted_source(url):
    domain = urlparse(url).netloc.replace("www.", "")
    return domain in TRUSTED_SOURCES

# --- SerpAPI Google search + content extraction ---
def search_news(query, max_results=5):
    matches = []
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": "4094c609a9bf63a1576c56a51453ff1ae4ecfa9b8ad08d333a2d15884e3dbdda",  # Replace with your SerpAPI key
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
                time.sleep(1)  # To avoid hammering servers
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
    except Exception as e:
        st.error(f"❌ Search error: {e}")
    return matches

# --- Evaluation Logic ---
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

    # Use trusted or fallback to all matches
    all_text = ' '.join(snippet for _, snippet in trusted_hits or matches)
    clean_input = clean_text(all_text)
    if len(clean_input.split()) > 20:
        try:
            summary = summarizer(clean_input[:1024], max_length=128, min_length=32, do_sample=False)[0]['summary_text']
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

# --- Advanced confidence visualization using Plotly gauge ---
def plot_confidence_gauge(confidence, status):
    # Define color based on status
    bar_color = "green" if status == 'REAL' else "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 75], 'color': "lightyellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit Interface ---
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
        plot_confidence_gauge(result['confidence'], result['status'])

    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
