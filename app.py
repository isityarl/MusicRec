import gradio as gr
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import re
import os
import random
import requests
import base64
import time
from huggingface_hub import hf_hub_download

SENTIMENT_MODEL_HF_ID_CONST = "yarlspace/mbert"
MLB_FILENAME_IN_REPO_CONST = "mlb.joblib"

DATASET_HF_ID_CONST = "yarlspace/forMusic"
EMOTION_RULES_FILENAME_IN_DATASET_CONST = "EmotionsWithFeatures.csv"
SONGS_DATABASE_FILENAME_IN_DATASET_CONST = "MusicsUpdated.csv"

CLIENT_ID_VAL = '0af487b64b5f4865b451cc6ab2269327'
CLIENT_SECRET_VAL = 'd68a84bc64bc402caf48296e29293bfa'

sentiment_tokenizer = None
sentiment_model = None
mlb_object = None
device = None
emotion_rules_df = None
songs_df = None
is_initialized = False
_spotify_access_token_cache = None
_spotify_token_expiry_time_cache = 0


class SpotifyAuthenticator:
    """Handles Spotify API token generation."""
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry_time = 0

    def get_token(self):
        if self.access_token and time.time() < self.token_expiry_time - 300:
            return self.access_token

        if not self.client_id or not self.client_secret:
            print("Spotify client ID or secret is missing for token generation.")
            return None

        auth_str = f"{self.client_id}:{self.client_secret}"
        b64_auth_str = base64.b64encode(auth_str.encode()).decode()
        token_url = 'https://accounts.spotify.com/api/token'

        try:
            response = requests.post(
                token_url,
                data={'grant_type': 'client_credentials'},
                headers={'Authorization': f'Basic {b64_auth_str}'}
            )
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.token_expiry_time = time.time() + token_data.get('expires_in', 3600)
            print("New Spotify access token obtained.")
            return self.access_token
        except requests.exceptions.RequestException as e:
            print(f"Failed to get Spotify token: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while getting Spotify token: {e}")

        self.access_token = None
        self.token_expiry_time = 0
        return None

class SentimentPipeline:
    """Handles sentiment model loading, text cleaning, and emotion prediction."""
    def __init__(self, model_hf_id, mlb_filename_in_repo):
        self.model_hf_id = model_hf_id
        self.mlb_filename_in_repo = mlb_filename_in_repo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.mlb_object = None
        self._load_model()
        print(f"SentimentPipeline initialized. Using device: {self.device}")

    def _load_model(self):
        try:
            print(f"Loading tokenizer and model from Hugging Face Hub: {self.model_hf_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_hf_id).to(self.device)
            self.model.eval()
            print("Tokenizer and model loaded from Hub.")

            print(f"Downloading MLB file '{self.mlb_filename_in_repo}' from repo '{self.model_hf_id}'...")
            mlb_local_path = hf_hub_download(
                repo_id=self.model_hf_id,
                filename=self.mlb_filename_in_repo
            )
            self.mlb_object = joblib.load(mlb_local_path)
            print(f"MLB object loaded from downloaded file: {mlb_local_path}")

            print(f"Sentiment components loaded successfully from Hugging Face Hub: {self.model_hf_id}")
            if self.model.config.num_labels != len(self.mlb_object.classes_):
                print(f"CRITICAL WARNING: Model labels ({self.model.config.num_labels}) != MLB classes ({len(self.mlb_object.classes_)}).")
        except Exception as e:
            print(f"FATAL: Error loading sentiment components from Hugging Face Hub: {e}")
            raise

    def clean_text(self, text):
        text = str(text)
        match = re.match(r"\[(KZ|RU|EN)\]", text)
        lang_tag = match.group(0) if match else ""
        text_wo_tag = text.replace(lang_tag, "") if lang_tag else text
        text_wo_tag = text_wo_tag.lower()
        text_wo_tag = re.sub(r"http\S+|www\S+|https\S+", '', text_wo_tag)
        text_wo_tag = re.sub(r"\s+", " ", text_wo_tag).strip()
        return f"{lang_tag} {text_wo_tag}".strip() if lang_tag else text_wo_tag

    def predict_emotions(self, text_input, threshold=0.1):
        if not all([self.model, self.tokenizer, self.mlb_object]):
            return ["Error: Sentiment components not properly loaded."], None, "Internal error: Sentiment components missing."

        cleaned_text = self.clean_text(text_input)
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        binary_predictions = (probabilities >= threshold).astype(int).reshape(1, -1)

        if binary_predictions.shape[1] != len(self.mlb_object.classes_):
            error_msg = f"Prediction shape mismatch. Model: {binary_predictions.shape[1]}, MLB: {len(self.mlb_object.classes_)}."
            return [error_msg], probabilities, error_msg

        predicted_emotion_tuples = self.mlb_object.inverse_transform(binary_predictions)
        predicted_emotions_list = list(predicted_emotion_tuples[0]) if predicted_emotion_tuples and predicted_emotion_tuples[0] else []

        # Modified to return sorted emotion probabilities for HTML rendering
        emotion_probs_for_html = []
        all_emotion_probs_tuples = []
        for i, prob_val in enumerate(probabilities):
            all_emotion_probs_tuples.append((self.mlb_object.classes_[i], prob_val))
        all_emotion_probs_tuples.sort(key=lambda x: x[1], reverse=True)

        if predicted_emotions_list:
            # Filter to include only predicted emotions that passed the threshold
            for emotion, prob in all_emotion_probs_tuples:
                if emotion in predicted_emotions_list:
                    emotion_probs_for_html.append((emotion, prob))
        else:
            # If no emotions above threshold, show top 5 overall
            emotion_probs_for_html = all_emotion_probs_tuples[:5]


        # Generate HTML string for emotion probabilities with updated structure
        prob_details_html = "<div class='emotion-probabilities-container'>"
        if not emotion_probs_for_html:
            prob_details_html += "<p>No emotions detected above threshold.</p>"
        else:
            for emotion, prob in emotion_probs_for_html:
                percentage = round(prob * 100, 2)
                # No <br> tag here now, handled by CSS
                prob_details_html += f"""
                <div class='emotion-item-wrapper'>
                    <div class='emotion-text-details'>
                        <span class='emotion-name' style='font-size: 1.0rem !important;'>{emotion.capitalize()}:</span>
                        <span class='probability-text' style='font-size: 0.8rem !important;'>{percentage:.2f}%</span>
                    </div>
                    <div class='progress-bar-container'>
                        <div class='progress-bar-fill' style='width: {percentage}%;'></div>
                    </div>
                </div>
                """
        prob_details_html += "</div>"

        return predicted_emotions_list, probabilities, prob_details_html

class MusicDatabase:
    """Loads and manages music rules and song features."""
    def __init__(self, dataset_hf_id, emotion_rules_filename, songs_db_filename):
        self.dataset_hf_id = dataset_hf_id
        self.emotion_rules_filename = emotion_rules_filename
        self.songs_db_filename = songs_db_filename
        self.emotion_rules_df = None
        self.songs_df = None
        self._load_data()
        print("MusicDatabase initialized.")

    def _load_data(self):
        try:
            print(f"Downloading emotion rules '{self.emotion_rules_filename}' from dataset '{self.dataset_hf_id}'...")
            emotion_rules_local_path = hf_hub_download(
                repo_id=self.dataset_hf_id,
                filename=self.emotion_rules_filename,
                repo_type="dataset"
            )
            self.emotion_rules_df = pd.read_csv(emotion_rules_local_path)
            print(f"Emotion rules loaded from Hub. Shape: {self.emotion_rules_df.shape}")
        except Exception as e:
            print(f"FATAL: Error loading emotion rules from Hub: {e}")
            raise RuntimeError(f"Failed to load emotion rules: {e}") from e

        try:
            print(f"Downloading songs database '{self.songs_db_filename}' from dataset '{self.dataset_hf_id}'...")
            songs_db_local_path = hf_hub_download(
                repo_id=self.dataset_hf_id,
                filename=self.songs_db_filename,
                repo_type="dataset"
            )
            self.songs_df = pd.read_csv(songs_db_local_path)
            print(f"Songs database loaded from Hub. Shape: {self.songs_df.shape}")

            audio_feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode',
                                     'speechiness', 'acousticness', 'instrumentalness',
                                     'liveness', 'valence', 'tempo', 'time_signature',
                                     'duration_ms', 'popularity']
            for col in audio_feature_columns:
                if col in self.songs_df.columns:
                    self.songs_df[col] = pd.to_numeric(self.songs_df[col], errors='coerce')
                else:
                    print(f"Warning: Expected column '{col}' not found in songs_df.")

            if 'name' not in self.songs_df.columns:
                print("Warning: 'name' column missing in songs_df. Adding default.")
                self.songs_df['name'] = "Unknown Track"
            else:
                self.songs_df['name'] = self.songs_df['name'].fillna("Unknown Track")

            if 'artists' not in self.songs_df.columns:
                print("Warning: 'artists' column missing in songs_df. Adding default.")
                self.songs_df['artists'] = "Unknown Artist"
            else:
                self.songs_df['artists'] = self.songs_df['artists'].fillna("Unknown Artist")

            if 'track_id' not in self.songs_df.columns:
                print("CRITICAL WARNING: 'track_id' column missing in songs_df.")
            if 'popularity' not in self.songs_df.columns:
                print("Warning: 'popularity' column is missing in songs_df. Adding default popularity=0.")
                self.songs_df['popularity'] = 0
            else:
                self.songs_df['popularity'] = self.songs_df['popularity'].fillna(0)

        except Exception as e:
            print(f"FATAL: Error loading or processing songs database from Hub: {e}")
            raise RuntimeError(f"Failed to load or process songs database: {e}") from e

    def get_emotion_rule(self, emotion_string):
        if self.emotion_rules_df is None: return None
        emotion_rule_row = self.emotion_rules_df[self.emotion_rules_df['Emotion'].str.lower() == emotion_string.lower()]
        return emotion_rule_row.iloc[0] if not emotion_rule_row.empty else None

    def get_all_songs(self):
        return self.songs_df.copy() if self.songs_df is not None else pd.DataFrame()

 
class MusicRecommender:
    """Recommends music based on emotions and feature rules."""
    def __init__(self, music_db_service, spotify_auth_service=None):
        self.music_db = music_db_service
        self.spotify_auth = spotify_auth_service
        print("MusicRecommender initialized.")

    def _get_spotify_tracks_popularity(self, track_ids):
        if not self.spotify_auth:
            print("SpotifyAuthenticator not provided. Cannot fetch live popularity.")
            return {track_id: 0 for track_id in track_ids}

        token = self.spotify_auth.get_token()
        if not token:
            print("Failed to get Spotify token. Cannot fetch live popularity.")
            return {track_id: 0 for track_id in track_ids}

        # The original code had a global get_tracks_popularity, which is not defined.
        # This part assumes you might add a function for live popularity later.
        # For now, it will return 0 popularity if not available.
        print("Warning: Live Spotify popularity fetching is not fully implemented or 'get_tracks_popularity' is missing.")
        return {track_id: 0 for track_id in track_ids}


    def recommend_music(self, predicted_emotions_list, num_recommendations=10, sort_by_popularity=True, use_live_popularity=False):
        if not predicted_emotions_list: return pd.DataFrame(columns=['name', 'artists'])

        primary_emotion = predicted_emotions_list[0].lower()
        rule = self.music_db.get_emotion_rule(primary_emotion)

        if rule is None: return pd.DataFrame(columns=['name', 'artists'])

        current_filter_df = self.music_db.get_all_songs()
        if current_filter_df.empty: return pd.DataFrame(columns=['name', 'artists'])

        audio_features_in_rules = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode',
                                     'speechiness', 'acousticness', 'instrumentalness',
                                     'liveness', 'valence', 'tempo', 'time_signature']

        for feature in audio_features_in_rules:
            min_col, max_col = f"{feature}_Min", f"{feature}_Max"
            if min_col in rule.index and max_col in rule.index and feature in current_filter_df.columns:
                min_val, max_val = rule[min_col], rule[max_col]
                if pd.notna(min_val) and pd.notna(max_val):
                    if not pd.api.types.is_numeric_dtype(current_filter_df[feature]):
                        continue

                    if feature == 'mode':
                        if min_val >= 0.5 and max_val >= 0.5:
                            current_filter_df = current_filter_df[current_filter_df[feature] == 1]
                        elif min_val <= 0.5 and max_val <= 0.5:
                            current_filter_df = current_filter_df[current_filter_df[feature] == 0]
                    elif feature == 'time_signature':
                        current_filter_df = current_filter_df[(current_filter_df[feature].astype(int) >= int(min_val)) & (current_filter_df[feature].astype(int) <= int(max_val))]
                    else:
                        current_filter_df = current_filter_df[(current_filter_df[feature] >= min_val) & (current_filter_df[feature] <= max_val)]

        if current_filter_df.empty:
            return pd.DataFrame(columns=['name', 'artists'])

        sorted_by_popularity_df = current_filter_df.sort_values(by='popularity', ascending=False)
        
        top_50_pool = sorted_by_popularity_df.head(50)

        num_to_sample = min(num_recommendations, len(top_50_pool))
        recommended_sample = top_50_pool.sample(n=num_to_sample) if num_to_sample > 0 else pd.DataFrame()
        print(f"Returning {len(recommended_sample)} shuffled recommendations from the top pool.")

        output_cols = ['name', 'artists', 'popularity', 'track_id']
        available_cols = [col for col in output_cols if col in recommended_sample.columns]

        output_df = recommended_sample[available_cols] if not recommended_sample.empty else pd.DataFrame(columns=available_cols)
        
        output_df = output_df.rename(columns={'name': 'Track Name', 'artists': 'Artists', 'popularity': 'Popularity'})
        
        return output_df
    

class GradioApp:
    """Orchestrates the components and builds the Gradio UI."""
    def __init__(self):
        print("Initializing GradioApp...")
        self.spotify_authenticator = SpotifyAuthenticator(CLIENT_ID_VAL, CLIENT_SECRET_VAL)
        self.sentiment_pipeline = SentimentPipeline(SENTIMENT_MODEL_HF_ID_CONST, MLB_FILENAME_IN_REPO_CONST)
        self.music_database = MusicDatabase(DATASET_HF_ID_CONST, EMOTION_RULES_FILENAME_IN_DATASET_CONST, SONGS_DATABASE_FILENAME_IN_DATASET_CONST)
        self.music_recommender = MusicRecommender(self.music_database, self.spotify_authenticator)
        self.is_app_initialized = True
        print("GradioApp fully initialized.")

        # Define example texts for the "Example" button
        self.example_texts = [
            "I can't believe I got the job, I am absolutely thrilled!",
            "Это очень печально и вызывает у меня тревогу.",
            "Бүгін менің көңіл-күйім тамаша, жаңа ән тыңдағым келеді!",
            "This performance is breathtaking, I'm in complete awe of their talent.",
            "Какая смешная шутка, я не могу перестать смеяться!",
            "The train has been delayed again, this is so frustrating and annoying!",
            "Мен қатты шаршадым, біраз демалғым келеді.",
            "Спасибо вам огромное за вашу помощь и поддержку!",
            "I'm very curious what will happen next in this story.",
            "Осыны күтпеп едім, қандай тосын сый!",
            "I studied so hard for the test but still failed. I feel so down.",
            "Бұл әділетсіздік менің ашуыма тиді."
        ]

    def get_random_example_text(self):
        """Returns a random example text from the predefined list."""
        return random.choice(self.example_texts)


    def get_recommendations_interface(self, text_input):
        if not self.is_app_initialized:
            # Return empty values for all outputs
            return "", pd.DataFrame(), ""
        if not text_input or not text_input.strip():
            return "Please enter text.", pd.DataFrame(), ""

        predicted_emotions, _, prob_details_html = self.sentiment_pipeline.predict_emotions(text_input)

        if not predicted_emotions or ("Error:" in predicted_emotions[0] and len(predicted_emotions) == 1):
            return prob_details_html, pd.DataFrame(), ""

        recommended_songs_df = self.music_recommender.recommend_music(
            predicted_emotions,
            num_recommendations=10
        )
        
        # --- Generate Spotify Player HTML ---
        # Sort the final shuffled list by popularity again just for display order in the UI
        display_df = recommended_songs_df.sort_values(by='Popularity', ascending=False)
        
        spotify_players_html_list = []
        if not display_df.empty:
            spotify_players_html_list.append("<h4 class='spotify-players-header'>Listen Now:</h4>")
        
        for index, row in display_df.iterrows():
            if 'track_id' in row and pd.notna(row['track_id']):
                track_id = row['track_id']
                track_name = row.get('Track Name', 'Unknown Track')
                artists = row.get('Artists', 'Unknown Artist')
                spotify_embed_url = f"https://open.spotify.com/embed/track/{track_id}"
                player_html = f"""
                <div class='spotify-player-container'>
                    <iframe title="Spotify Web Player for {track_name} by {artists}"
                            style="border-radius:12px; width:100%;"
                            src="{spotify_embed_url}"
                            height="80"
                            frameBorder="0"
                            allowfullscreen=""
                            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                            loading="lazy">
                    </iframe>
                </div>
                """
                spotify_players_html_list.append(player_html)

        full_spotify_players_html = "\n".join(spotify_players_html_list)
        if not full_spotify_players_html or len(display_df) == 0:
            full_spotify_players_html = "<p class='no-songs-message'>No playable Spotify tracks found for the recommendations.</p>"

        # Prepare DataFrame for UI
        ui_cols = ['Track Name', 'Artists', 'Popularity']
        ui_df = display_df[ui_cols] if not display_df.empty and all(col in display_df.columns for col in ui_cols) else pd.DataFrame()
        
        if 'Popularity' in ui_df.columns:
            ui_df['Popularity'] = ui_df['Popularity'].astype(int)

        return prob_details_html, ui_df, full_spotify_players_html

    def launch(self):
        # Read CSS from the external file
        css_file_path = "styles.css"
        css_app_dark_theme = ""
        try:
            with open(css_file_path, "r") as f:
                css_app_dark_theme = f.read()
            print(f"Successfully loaded CSS from {css_file_path}")
        except FileNotFoundError:
            print(f"Error: CSS file '{css_file_path}' not found. Please ensure it's in the same directory.")
        except Exception as e:
            print(f"Error reading CSS file: {e}")

        with gr.Blocks(theme=gr.themes.Base(), css=css_app_dark_theme) as demo:
            gr.Markdown("<h1>🎤 Sentiment-Driven Multilingual Music Recommender 🎧</h1>", elem_classes="gr-title")
            gr.Markdown(
                "Enter your text in English, Russian, or Kazakh (e.g., [EN] I'm happy!). "
                "The system will analyze your emotions and recommend music based on your mood.",
                elem_classes="gr-description"
            )

            text_input = gr.Textbox(
                lines=4,
                placeholder="E.g., '[EN] I'm so happy today!' or '[RU] Мне очень грустно.' or '[KZ] Көңіл-күйім тамаша!'...",
                label="Your text:"
            )
            with gr.Row():
                clear_button = gr.ClearButton(value="Clear", variant="secondary")
                example_button = gr.Button("Example", variant="secondary")
                submit_button = gr.Button("Submit", variant="primary")


            gr.Markdown("### Emotion Probabilities:", elem_id="emotion_prob_label")
            emotion_output_html = gr.HTML(elem_id="emotion-output-display")
            
            recommendation_df_output = gr.DataFrame(
                headers=['Track Name', 'Artists', 'Popularity'],
                label="🎵 Music Recommendations (Randomly selected from Top 50)",
                wrap=True,
                row_count=(10,"dynamic"),
                col_count=(3,"fixed")
            )

            gr.Markdown("<h3>🎧 Play Recommended Tracks:</h3>", elem_id="player_section_title")
            spotify_player_html_output = gr.HTML(elem_id="spotify-players-container")

            submit_button.click(
                fn=self.get_recommendations_interface,
                inputs=[text_input],
                outputs=[emotion_output_html, recommendation_df_output, spotify_player_html_output]
            )
            clear_button.click(
                lambda: (None, pd.DataFrame(), ""),
                outputs=[emotion_output_html, recommendation_df_output, spotify_player_html_output]
            )
            example_button.click(
                fn=self.get_random_example_text,
                inputs=[],
                outputs=[text_input]
            )

        demo.launch()

if __name__ == '__main__':
    app = GradioApp()
    app.launch()
