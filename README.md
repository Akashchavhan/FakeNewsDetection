# FakeNewsDetection

# 📰 Advanced Fake News Detector

This project allows users to verify the authenticity of any news headline by searching across trusted news sources and using a simple AI-backed credibility checker. It also summarizes the most relevant news articles and provides a confidence score along with visual insights.

---

## 🔍 How It Works

1. **User Inputs a Headline**  
   Example: _"Behind the Pahalgam attack is Pakistan"_

2. **Google Search Is Triggered**  
   The app searches the headline using the `googlesearch` Python module.

3. **Trusted Source Verification**  
   It checks if results appear on pre-approved trusted sources like **BBC**, **Reuters**, **CNN**, **NDTV**, etc.

4. **AI-based News Summarization**  
   If a trusted article is found, the app summarizes it using a HuggingFace transformer model.

5. **Prediction with Confidence Score**  
   Based on presence in trusted sources, it predicts whether the news is **REAL** or **FAKE**, with a confidence percentage.

6. **Visualization**  
   A simple graph shows the confidence level visually.

---

## 📊 Example Output

**Input**  
> "Behind the Pahalgam attack is Pakistan"

**Prediction**  
- **Status:** REAL  
- **Confidence:** 50.0%

**Summary**  
> The US has urged India and Pakistan to work together to de-escalate tensions. A deadly militant attack in Indian-administered Kashmir last week killed 26 civilians. Secretary of State Marco Rubio held separate talks with India’s foreign minister and Pakistan’s prime minister.

**Trusted Sources**
1. [BBC Article 1](https://www.bbc.com/news/articles/cvgnw9kydgqo)  
2. [BBC Article 2](https://www.bbc.com/news/articles/cn4wk22vk4zo)

**Confidence Visualization**
![Confidence Graph](confidence_graph.png)

---

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
