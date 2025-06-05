---
title: Sentiment-Driven Music Recommender
emoji: 🎧 # Changed emoji to be more relevant
colorFrom: green
colorTo: blue
sdk: gradio
# sdk_version: (You can often remove this if using a recent Gradio version specified in requirements.txt)
app_file: app.py
pinned: false
license: apache-2.0 # Or your chosen license
---

# Sentiment-Driven Multilingual Music Recommender

This application predicts emotions from text input (English, Russian, Kazakh) and recommends music based on audio feature rules.

## Data Sources (For Development - Not directly used by deployed app if loading from HF Hub)
* EmotionsWithFeatures.csv (Rules): Describes audio feature ranges for different emotions.
* MusicsUpdated.csv (Song Database): Contains songs and their audio features.
* Sentiment Model (mbert based): Fine-tuned for multilingual emotion detection.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference