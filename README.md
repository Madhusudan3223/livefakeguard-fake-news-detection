# 🛡️ LiveFakeGuard: Real-Time Fake News Detection

> Detect whether live news headlines are Fake or Real using a pre-trained ML model, NewsAPI, and Streamlit.

## 📌 Project Overview

- 🔍 Fetches real-time news headlines from NewsAPI
- 🧹 Cleans and preprocesses the content
- 🧠 Uses a trained ML model to classify as **Fake** or **Real**
- 📊 Shows predictions in Colab and Streamlit UI
- 📁 Exports results as downloadable CSV

---

## 🚀 Tech Stack

- Python, Pandas, Scikit-learn
- TF-IDF Vectorizer + Logistic Regression
- NewsAPI.org (Live data source)
- Google Colab (Notebook)
- Streamlit (Web App UI)

---

## 📦 Files

| File | Description |
|------|-------------|
| `fake_news_model_v2.pkl` | Trained Logistic Regression model |
| `tfidf_vectorizer_v2.pkl` | TF-IDF transformer for news text |
| `live_news_predictions.csv` | Output CSV file of live news predictions |

---

## 🧪 How to Run in Colab

1. [Open this notebook in Colab](https://colab.research.google.com/drive/1SZly8lxx3HHnFGhg6y3drVIMxs3A8vif?usp=sharing)
2. Upload the `.pkl` files (model + vectorizer)
3. Paste your NewsAPI key
4. Run the cells step-by-step
5. View predictions and download CSV

---

## 🌐 Try the Web App

🖥️ **Live App:**  
[https://livefakeguard-fake-news-detection.streamlit.app](https://livefakeguard-fake-news-detection-eknyx8uixfzgbqsyuzi7zy.streamlit.app/)

📋 Steps:
- Get your free NewsAPI key from [https://newsapi.org](https://newsapi.org)
- Paste into the app and click "Fetch & Analyze News"
- View predictions and download results

---

## 👨‍💻 Author

**Madhusudan Mandal**  
🔗 [GitHub](https://github.com/Madhusudan3223) | ✉️ madhumandal49@gmail.com

---

## 📄 License

This project is licensed for educational and demo use.
