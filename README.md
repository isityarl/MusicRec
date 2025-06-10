---
title: Sentiment-Driven Music Recommender
emoji: 🎧
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: '5.32.1'
app_file: app.py
pinned: false
---

# 🎤 Sentiment-Driven Multilingual Music Recommender 🎧

This application analyzes the sentiment and emotions in your text and recommends music that matches your mood. The model understands text in English, Russian, and Kazakh.

## 🚀 How It Works

The recommendation process follows a few simple steps:

1. Emotion Analysis  
   You enter text describing your mood. A fine-tuned mBERT model processes the text to predict one or more associated emotions (e.g., *sadness*, *joy*, *admiration*).

2. Rule-Based Filtering  
   The system takes the primary emotion and uses a predefined set of rules to find songs with matching audio features (like *danceability*, *energy*, *valence*, etc.).

3. Curated Recommendations  
   To ensure quality, the app filters the matching songs, sorts them by popularity, and creates a candidate pool from the top 50 tracks.

4. Dynamic Output  
   It then shuffles this high-quality pool and presents 10 random recommendations, ensuring a fresh list every time you click submit.

5. Interactive Player  
   You can listen to the recommended tracks directly in the app using the embedded Spotify players.

## 📊 Datasets

This project relies on two key datasets:

- Emotion Rules (EmotionWithFeatures.csv)  
  A custom-made dataset that maps specific emotions to ideal ranges for Spotify's audio features.  
  [Download Link]

- Songs Database (MusicsUpdated.csv)  
  A large collection of songs and their corresponding audio features from Spotify.  
  [Download Link]

---

This Space is the final result of the Gradio app development.