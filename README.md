# 🎵 Music Recommendation System

A *content-based music recommendation system* built as part of my internship at *Infotact Solutions*.  
The project uses *song lyrics* and *artist similarity* to recommend tracks, providing users with a seamless way to discover new music.  
It features an interactive *Streamlit dashboard* with a modern UI and background theme.

---

## 📊 Dataset
This project uses the *Spotify Million Song Dataset (Millsongdata)* available on Kaggle:  
🔗 [Spotify Million Song Dataset (Millsongdata)](https://www.kaggle.com/datasets/rymnikski/spotify-million-song-dataset)

---

## ⚙ Tech Stack
- *Python*
- *Pandas, **NumPy*
- *Scikit-learn* (TF-IDF, cosine similarity)
- *Streamlit* (dashboard)
- *NLTK / text preprocessing*

---

## 🚀 Features
- *Content-based recommendations* using *TF-IDF on lyrics*
- *Hybrid scoring* with artist similarity boosting
- *Cosine similarity* for finding closest matches
- *Modern UI dashboard* built with Streamlit
- *Instant search*: type any track name and get top song recommendations

---

## 📂 Project Structure

├── app.py                # Main Streamlit dashboard
├── recommender.ipynb     # Jupyter notebook for building & testing model
├── data/                 # Dataset folder (not included in repo)
├── requirements.txt      # Dependencies
└── README.md             # Project documentation


---

## 🛠 Installation & Setup
1. Clone this repository:
   bash
   git clone https://github.com/your-username/music-recommender.git
   cd music-recommender
   

2. Install dependencies:
   bash
   pip install -r requirements.txt
   

3. Run the Streamlit app:
   bash
   streamlit run app.py
   

---

## 🎯 Future Improvements
- Add collaborative filtering for user-personalized recommendations  
- Integrate audio features (tempo, key, energy)  
- Deploy on cloud (Heroku/Streamlit Sharing) for public access  

---
