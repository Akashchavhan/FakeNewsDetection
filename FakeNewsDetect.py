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
from serpapi import GoogleSearch  # Make sure serpapi is installed and you have your API key
import plotly.graph_objects as go  # For animated confidence gauge
import streamlit.components.v1 as components  # To inject JS for autoplay

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

# --- Animated confidence gauge using Plotly with autoplay ---
def animated_confidence_gauge(confidence, status):
    steps = 20  # Number of animation frames
    values = [confidence * i / steps for i in range(steps + 1)]

    frames = [go.Frame(data=[go.Indicator(
                mode="gauge+number+delta",
                value=val,
                title={'text': "Confidence (%)"},
                delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green" if status == 'REAL' else "red"},
                    'steps': [
                        {'range': [0, 49], 'color': 'red'},
                        {'range': [50, 100], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "blue", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            )]) for val in values]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames[1:],
    )

    fig.update_layout(
        height=300,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}],
            }]
        }],
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
        st.plotly_chart(fig, use_container_width=True)

        # Autoplay animation by injecting JS to click hidden play button automatically
        components.html(
            """
            <script>
            const plot = document.querySelector("div.js-plotly-plot");
            if(plot){
              const btns = plot.querySelectorAll('button[title="Play"]');
              if(btns.length > 0){
                btns[0].click();  // Auto-click hidden play button to start animation
              }
            }
            </script>
            """,
            height=0,
            width=0,
        )

    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
