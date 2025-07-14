import streamlit as st
import pandas as pd
import requests
import joblib
import re

# Load model and vectorizer
model = joblib.load('fake_news_model_v2.pkl')
vectorizer = joblib.load('tfidf_vectorizer_v2.pkl')

# Preprocess function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Fetch news from NewsAPI
def fetch_news(api_key):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'India',
        'language': 'en',
        'pageSize': 20,
        'sortBy': 'publishedAt',
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get('articles', [])
    df = pd.DataFrame([{
        'title': a['title'],
        'description': a['description'],
        'content': a['content'],
        'url': a['url']
    } for a in articles])
    return df

# App UI
st.title("🛡️ LiveFakeGuard: Real-Time Fake News Detection")
api_key = st.text_input("🔑 Enter your NewsAPI key:", type="password")

if st.button("🔍 Fetch & Analyze News"):
    if api_key:
        with st.spinner("Fetching and analyzing news..."):
            df = fetch_news(api_key)
            df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
            df['clean_text'] = df['text'].apply(clean_text)
            X = vectorizer.transform(df['clean_text'])
            preds = model.predict(X)
            df['Prediction'] = ['Real' if p == 1 else 'Fake' for p in preds]

            st.success("✅ Analysis complete!")
            st.dataframe(df[['title', 'Prediction', 'url']])

            # Download button
            csv = df[['title', 'Prediction', 'url']].to_csv(index=False)
            st.download_button("📥 Download Results as CSV", csv, "predictions.csv", "text/csv")
    else:
        st.warning("Please enter your NewsAPI key.")
