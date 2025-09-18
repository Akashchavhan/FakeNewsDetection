import streamlit as st
st.set_page_config(page_title="Fake News Verifier", layout="centered")
st.title("📰 Advanced Fake News Detector")

import re
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import pipeline
from serpapi import GoogleSearch  # Make sure serpapi is installed and you have your API key
import plotly.graph_objects as go  # For animated confidence gauge

# --- Caching the model to avoid reloading ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# --- Expanded Trusted sources list ---
TRUSTED_SOURCES = [
    "bbc.com",
    "reuters.com",
    "ndtv.com",
    "cnn.com",
    "indiatoday.in",
    "thehindu.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "aljazeera.com",
    "apnews.com",
    "foxnews.com",
    "washingtonpost.com",
    "nytimes.com",
    "economictimes.indiatimes.com",
    "scroll.in",
    "bbc.co.uk",
    "cbc.ca",
    "theguardian.com",
    "cnbc.com",
    "dw.com",
    "npr.org",
    "bbcnews.com",
    "news18.com",
    "thewire.in"
]

# --- Text cleaning ---
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    return text.lower()

# --- Trusted source checker with partial matching ---
def is_trusted_source(url):
    domain = urlparse(url).netloc.replace("www.", "").lower()
    return any(trusted_domain in domain for trusted_domain in TRUSTED_SOURCES)

# --- SerpAPI Google search + content extraction ---
def search_news(query, max_results=5):
    matches = []
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": st.secrets["SERPAPI_KEY"],  # Use Streamlit secrets here
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

# --- Animated confidence gauge using Plotly ---
def animated_confidence_gauge(confidence, status):
    bar_color = "#2ecc71" if status == "REAL" else "#e74c3c"  # green or red
    step_colors = ["#d5f5e3", "#abebc6", "#82d18a", "#52c41a", "#2ecc71"] if status == "REAL" else ["#fdecea", "#f9bdbb", "#f38a7d", "#ed6a5a", "#e74c3c"]

    steps = 30  # number of frames for animation
    values = [confidence * i / steps for i in range(steps + 1)]

    frames = []
    for val in values:
        frames.append(go.Frame(data=[go.Indicator(
            mode="gauge+number+delta",
            value=val,
            number={'font': {'size': 48, 'color': bar_color}, 'suffix': "%"},
            delta={'reference': 50, 'increasing': {'color': "#2ecc71"}, 'decreasing': {'color': "#e74c3c"}},
            title={'text': "<b>Confidence</b>", 'font': {'size': 24, 'color': 'black'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkgray", 'nticks': 10},
                'bar': {'color': bar_color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "lightgray",
                'steps': [
                    {'range': [0, 20], 'color': step_colors[0]},
                    {'range': [20, 40], 'color': step_colors[1]},
                    {'range': [40, 60], 'color': step_colors[2]},
                    {'range': [60, 80], 'color': step_colors[3]},
                    {'range': [80, 100], 'color': step_colors[4]},
                ],
                'threshold': {
                    'line': {'color': "blue", 'width': 5},
                    'thickness': 0.85,
                    'value': 50,
                }
            }
        )]))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames[1:]
    )

    fig.update_layout(
        height=350,
        margin={'t': 50, 'b': 0, 'l': 0, 'r': 0},
        paper_bgcolor='rgba(240,240,240,0.95)',
        font=dict(family="Helvetica, Arial, sans-serif", color="black"),
        annotations=[
            dict(
                x=0.5,
                y=0,
                showarrow=False,
                text="Threshold at 50% confidence",
                font=dict(size=12, color="blue"),
                xanchor='center',
                yanchor='top',
            )
        ],
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [],
            'visible': False,  # hide buttons
        }]
    )

    # Trick to autoplay animation on load without controls
    fig.update_layout(
        sliders=[],
        updatemenus=[]
    )

    # The animation settings: play all frames on load
    fig.layout.update(
        transition={'duration': 50, 'easing': 'linear'},
        frame={'duration': 50, 'redraw': True},
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
