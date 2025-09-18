import streamlit as st
st.set_page_config(page_title="Fake News Verifier", layout="centered")
st.title("📰 Advanced Fake News Detector")

import re
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import pipeline
from serpapi import GoogleSearch
import plotly.graph_objects as go

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

def animated_confidence_gauge(confidence, status):
    bar_color = "#2ecc71" if status == "REAL" else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=0,  # start at 0, will animate to actual confidence
        number={'suffix': "%", 'font': {'size': 48, 'color': bar_color}},
        delta={'reference': 50, 'increasing': {'color': "#2ecc71"}, 'decreasing': {'color': "#e74c3c"}},
        title={'text': "<b>Confidence</b>", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "lightgray",
            'steps': [
                {'range': [0, 20], 'color': "#fdecea"},
                {'range': [20, 40], 'color': "#f9bdbb"},
                {'range': [40, 60], 'color': "#f38a7d"},
                {'range': [60, 80], 'color': "#ed6a5a"},
                {'range': [80, 100], 'color': "#e74c3c"} if status == "FAKE" else {'range': [80, 100], 'color': "#2ecc71"},
            ],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    # Animate using frame generator
    frames = [go.Frame(data=[go.Indicator(value=v)]) for v in range(0, int(confidence) + 1)]
    fig.frames = frames

    fig.update_layout(
        height=350,
        margin=dict(t=40, b=20, l=10, r=10),
        paper_bgcolor='white',
        template='plotly_white',
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play", method="animate", args=[None, {
                "frame": {"duration": 30, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0}
            }])],
            x=0.1, y=0
        )],
        sliders=[{
            "steps": [{
                "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": str(i),
                "method": "animate"
            } for i, f in enumerate(fig.frames)],
            "transition": {"duration": 0},
            "x": 0, "len": 1.0
        }]
    )

    return fig

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
        fig = animated_confidence_gauge(result['confidence'], result['status'])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
