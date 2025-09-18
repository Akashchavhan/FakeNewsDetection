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

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "ndtv.com", "cnn.com", "indiatoday.in",
    "thehindu.com", "timesofindia.indiatimes.com", "hindustantimes.com",
    "aljazeera.com", "apnews.com", "foxnews.com", "washingtonpost.com",
    "nytimes.com", "economictimes.indiatimes.com", "scroll.in", "bbc.co.uk",
    "cbc.ca", "theguardian.com", "cnbc.com", "dw.com", "npr.org",
    "bbcnews.com", "news18.com", "thewire.in", "indianexpress.com"
]

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    return text.lower()

def is_trusted_source(url):
    domain = urlparse(url).netloc.replace("www.", "").lower()
    return any(trusted_domain in domain for trusted_domain in TRUSTED_SOURCES)

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

# ---------- Animated Gauge ----------
def animated_confidence_gauge(confidence, status):
    bar_color = "#2ecc71" if status == "REAL" else "#e74c3c"
    placeholder = st.empty()

    for val in range(0, int(confidence) + 1, 2):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=val,
                number={'suffix': "%", 'font': {'size': 52, 'color': bar_color, "family": "Arial Black"}},
                title={'text': "<b>Confidence</b>", 'font': {'size': 24, 'family': "Arial"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                    'bar': {'color': bar_color, 'thickness': 0.25},
                    'bgcolor': "white",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(231, 76, 60,0.1)"},
                        {'range': [50, 100], 'color': "rgba(46, 204, 113,0.1)"}
                    ],
                }
            )
        )
        fig.update_layout(
            height=350,
            margin=dict(t=40, b=20, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            template='plotly_white',
        )
        placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        time.sleep(0.03)

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
                sort=False,  # no flip
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

# ---------------- Main App ----------------
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

        # Chart selector
        chart_type = st.radio("Select Visualization:", ["Gauge", "Donut"], horizontal=True)

        st.subheader("Confidence Visualization")
        if chart_type == "Gauge":
            animated_confidence_gauge(result['confidence'], result['status'])
        else:
            animated_confidence_donut(result['confidence'], result['status'])

    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
