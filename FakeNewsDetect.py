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

# --- Caching the model to avoid reloading ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# --- Trusted sources list ---
TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "ndtv.com", "cnn.com", "indiatoday.in",
    "thehindu.com", "timesofindia.indiatimes.com", "hindustantimes.com"
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

# --- Google search + content extraction ---
def search_news(query, max_results=5):
    matches = []
    try:
        # Construct the Google search URL
        query = query.replace(' ', '+')
        url = f"https://www.google.com/search?q={query}&num={max_results}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            st.error(f"Google search request failed with status code {response.status_code}")
            return matches
        
        soup = BeautifulSoup(response.text, 'html.parser')

        # Google search result containers
        search_results = soup.select('div.g')
        count = 0
        
        for result in search_results:
            if count >= max_results:
                break
            
            link_tag = result.find('a', href=True)
            if link_tag:
                link = link_tag['href']
                if link.startswith('/url?q='):
                    link = link[7:]
                    link = link.split('&')[0]  # Clean URL
                
                if not link.startswith('http'):
                    continue
                
                try:
                    page_response = requests.get(link, timeout=10)
                    if page_response.status_code != 200:
                        continue
                    
                    page_soup = BeautifulSoup(page_response.text, 'html.parser')
                    paragraphs = page_soup.find_all('p')
                    text_snippet = ' '.join([p.get_text() for p in paragraphs[:5]])
                    if text_snippet.strip():
                        matches.append((link, text_snippet))
                        count += 1
                except Exception as e:
                    print(f"Error processing {link}: {e}")
                    continue
                
                time.sleep(1)  # to avoid being blocked
                
    except Exception as e:
        st.error(f"❌ Google search error: {e}")
    
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
        fig, ax = plt.subplots()
        ax.bar(["Confidence"], [result['confidence']], color="green" if result['status'] == 'REAL' else "red")
        ax.set_ylim([0, 100])
        ax.set_ylabel("%")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")
