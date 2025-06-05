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

# --- Configuration Constants ---
SENTIMENT_MODEL_DIR_PATH_CONST = "mbert" 
MLB_FILENAME_PATH_CONST = "mlb.joblib"
EMOTION_RULES_CSV_FILE_PATH_CONST = "EmotionsWithFeatures.csv" 
SONGS_DATABASE_CSV_FILE_PATH_CONST = "MusicsUpdated.csv" 

# --- Spotify API Credentials (Hardcoded as per user's script) ---
# WARNING: For actual deployment, these should be moved to Hugging Face Space Secrets
# and accessed via os.environ.get("YOUR_SECRET_NAME")
CLIENT_ID_VAL = '0af487b64b5f4865b451cc6ab2269327'
CLIENT_SECRET_VAL = 'd68a84bc64bc402caf48296e29293bfa'

class SpotifyAuthenticator:
    """Handles Spotify API token generation."""
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry_time = 0
        # print("SpotifyAuthenticator initialized.") # Less verbose for deployment

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
    def __init__(self, model_dir_path, mlb_filename):
        self.model_dir_path = model_dir_path
        self.mlb_filename = mlb_filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.mlb_object = None
        self._load_model()
        print(f"SentimentPipeline initialized. Using device: {self.device}")

    def _load_model(self):
        mlb_path_full = os.path.join(self.model_dir_path, self.mlb_filename)
        try:
            if not os.path.isdir(self.model_dir_path): 
                raise FileNotFoundError(f"Sentiment model directory not found: ./{self.model_dir_path}")
            if not os.path.exists(mlb_path_full): 
                raise FileNotFoundError(f"MLB file not found: ./{mlb_path_full}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir_path).to(self.device)
            self.model.eval()
            self.mlb_object = joblib.load(mlb_path_full)
            print(f"Sentiment components loaded from ./{self.model_dir_path}.")
            if self.model.config.num_labels != len(self.mlb_object.classes_):
                 print(f"CRITICAL WARNING: Model labels ({self.model.config.num_labels}) != MLB classes ({len(self.mlb_object.classes_)}).")
        except Exception as e:
            print(f"FATAL: Error loading sentiment components: {e}")
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
        
        emotion_probs_tuples = []
        if predicted_emotions_list: 
            for emotion in predicted_emotions_list:
                try:
                    idx = list(self.mlb_object.classes_).index(emotion)
                    emotion_probs_tuples.append((emotion, probabilities[idx]))
                except ValueError:
                    emotion_probs_tuples.append((emotion, 0.0)) 
            emotion_probs_tuples.sort(key=lambda x: x[1], reverse=True)
            prob_details = "Emotion Probabilities:\n" 
            for emotion, prob in emotion_probs_tuples:
                prob_details += f"- {emotion}: {prob:.2f}\n"
        else: 
            prob_details = "Emotion Probabilities:\n_(No emotions above threshold)_\n\nTop 5 Overall (sorted):\n"
            all_emotion_probs_tuples = []
            for i, prob_val in enumerate(probabilities):
                all_emotion_probs_tuples.append((self.mlb_object.classes_[i], prob_val))
            all_emotion_probs_tuples.sort(key=lambda x: x[1], reverse=True)
            
            for emotion, prob_val in all_emotion_probs_tuples[:5]: 
                prob_details += f"- {emotion}: {prob_val:.2f}\n"
                
        return predicted_emotions_list, probabilities, prob_details.strip()


class MusicDatabase:
    """Loads and manages music rules and song features."""
    def __init__(self, emotion_rules_path, songs_db_path):
        self.emotion_rules_path = emotion_rules_path
        self.songs_db_path = songs_db_path
        self.emotion_rules_df = None
        self.songs_df = None
        self._load_data()
        print("MusicDatabase initialized.")

    def _load_data(self):
        try:
            if not os.path.exists(self.emotion_rules_path): 
                raise FileNotFoundError(f"Emotion rules CSV not found: ./{self.emotion_rules_path}")
            self.emotion_rules_df = pd.read_csv(self.emotion_rules_path)
            print(f"Emotion rules loaded from ./{self.emotion_rules_path}. Shape: {self.emotion_rules_df.shape}")
        except Exception as e:
            print(f"FATAL: Error loading emotion rules: {e}")
            raise RuntimeError(f"Failed to load emotion rules: {e}") from e

        try:
            if not os.path.exists(self.songs_db_path): 
                raise FileNotFoundError(f"Songs database CSV not found: ./{self.songs_db_path}")
            self.songs_df = pd.read_csv(self.songs_db_path)
            print(f"Songs database loaded from ./{self.songs_db_path}. Shape: {self.songs_df.shape}")
            
            audio_feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                     'speechiness', 'acousticness', 'instrumentalness', 
                                     'liveness', 'valence', 'tempo', 'time_signature', 
                                     'duration_ms', 'popularity'] 
            for col in audio_feature_columns:
                if col in self.songs_df.columns:
                    self.songs_df[col] = pd.to_numeric(self.songs_df[col], errors='coerce')
                else:
                    print(f"Warning: Expected column '{col}' not found in songs_df.")
            
            if 'track_id' not in self.songs_df.columns:
                print("CRITICAL WARNING: 'track_id' column missing in songs_df.")
            if 'popularity' not in self.songs_df.columns:
                 print("Warning: 'popularity' column is missing in songs_df. Adding default popularity=0.")
                 self.songs_df['popularity'] = 0 
            else:
                self.songs_df['popularity'].fillna(0, inplace=True)
        except Exception as e:
            print(f"FATAL: Error loading or processing songs database: {e}")
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
        
        # This calls the global get_tracks_popularity function.
        # For purer OOP, this logic would ideally be part of the SpotifyAuthenticator class.
        if 'get_tracks_popularity' in globals():
            return get_tracks_popularity(track_ids, token) 
        else:
            print("Warning: Global get_tracks_popularity function not found for live fetch.")
            return {track_id: 0 for track_id in track_ids}


    def recommend_music(self, predicted_emotions_list, num_recommendations=10, sort_by_popularity=True, use_live_popularity=False):
        if not predicted_emotions_list: return pd.DataFrame() 
        
        primary_emotion = predicted_emotions_list[0].lower()
        rule = self.music_db.get_emotion_rule(primary_emotion)

        if rule is None: return pd.DataFrame()

        current_filter_df = self.music_db.get_all_songs()
        if current_filter_df.empty: return pd.DataFrame()

        audio_features_in_rules = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode',
                                   'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'time_signature']

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
            return pd.DataFrame()

        if sort_by_popularity:
            if use_live_popularity and 'track_id' in current_filter_df.columns and self.spotify_auth and CLIENT_ID_VAL and CLIENT_SECRET_VAL:
                print("Attempting to fetch LIVE Spotify popularity...")
                track_ids_to_fetch = current_filter_df['track_id'].dropna().unique().tolist()
                MAX_POPULARITY_FETCH = 50 
                ids_for_popularity = track_ids_to_fetch
                if len(track_ids_to_fetch) > MAX_POPULARITY_FETCH:
                    ids_for_popularity = random.sample(track_ids_to_fetch, MAX_POPULARITY_FETCH)
                
                if ids_for_popularity:
                    live_popularities = self._get_spotify_tracks_popularity(ids_for_popularity)
                    current_filter_df['live_popularity'] = current_filter_df['track_id'].map(live_popularities).fillna(0)
                    current_filter_df.sort_values(by='live_popularity', ascending=False, inplace=True)
                    if 'popularity' not in current_filter_df.columns: 
                        current_filter_df['popularity'] = current_filter_df['live_popularity']
            elif 'popularity' in current_filter_df.columns and pd.api.types.is_numeric_dtype(current_filter_df['popularity']):
                current_filter_df.sort_values(by='popularity', ascending=False, inplace=True)
            else:
                print("Warning: Cannot sort by popularity. 'popularity' column missing, not numeric, or live fetch disabled/failed.")
                if 'popularity' not in current_filter_df.columns:
                    current_filter_df['popularity'] = 0 
        else: 
             if 'popularity' not in current_filter_df.columns: current_filter_df['popularity'] = 0


        num_to_sample = min(num_recommendations, len(current_filter_df))
        recommended_sample = current_filter_df.head(num_to_sample) if num_to_sample > 0 else pd.DataFrame()
        
        display_cols = ['name', 'artists'] 
            
        final_display_cols = [col for col in display_cols if col in recommended_sample.columns]
        if not final_display_cols and not recommended_sample.empty: 
            final_display_cols = recommended_sample.columns.tolist()[:2] 

        recommendations_to_display = recommended_sample[final_display_cols].copy() if final_display_cols and not recommended_sample.empty else pd.DataFrame()
        if not recommendations_to_display.empty:
            recommendations_to_display.rename(columns={'name': 'Track Name', 'artists': 'Artists'}, inplace=True)
        
        return recommendations_to_display


class GradioApp:
    """Orchestrates the components and builds the Gradio UI."""
    def __init__(self):
        print("Initializing GradioApp...")
        self.spotify_authenticator = SpotifyAuthenticator(CLIENT_ID_VAL, CLIENT_SECRET_VAL) 
        self.sentiment_pipeline = SentimentPipeline(SENTIMENT_MODEL_DIR_PATH_CONST, MLB_FILENAME_PATH_CONST)
        self.music_database = MusicDatabase(EMOTION_RULES_CSV_FILE_PATH_CONST, SONGS_DATABASE_CSV_FILE_PATH_CONST)
        self.music_recommender = MusicRecommender(self.music_database, self.spotify_authenticator)
        self.is_app_initialized = True
        print("GradioApp fully initialized.")


    def get_recommendations_interface(self, text_input):
        if not self.is_app_initialized: 
            return "Application components not fully initialized.", pd.DataFrame(columns=['Track Name', 'Artists'])
        if not text_input or not text_input.strip(): 
            return "Please enter text.", pd.DataFrame(columns=['Track Name', 'Artists'])

        predicted_emotions, _, prob_details_str = self.sentiment_pipeline.predict_emotions(text_input)
        
        emotion_analysis_output = f"{prob_details_str.strip()}" 

        df_headers_for_ui = ['Track Name', 'Artists'] 

        if not predicted_emotions or ("Error:" in predicted_emotions[0] and len(predicted_emotions) == 1):
            return emotion_analysis_output, pd.DataFrame(columns=df_headers_for_ui)

        recommended_songs_df = self.music_recommender.recommend_music(
            predicted_emotions, 
            num_recommendations=10, 
            sort_by_popularity=True, 
            use_live_popularity=False 
        ) 
        
        if recommended_songs_df.empty:
            recommended_songs_df = pd.DataFrame(columns=df_headers_for_ui)
        else:
            cols_to_show_in_gradio = [col for col in df_headers_for_ui if col in recommended_songs_df.columns]
            recommended_songs_df = recommended_songs_df[cols_to_show_in_gradio]
            
        return emotion_analysis_output, recommended_songs_df

    def launch(self):
        css_app_spotify_light = """
        .gradio-container { 
            background-color: #323232 !important;
            font-family: 'CircularSp', 'Helvetica Neue', Helvetica, Arial, sans-serif;
            max-width: 850px; 
            margin: auto;
            color: #bb3e3e !important;
        }
        .gr-panel { 
            border-radius: 8px !important; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important; 
            padding: 20px !important; 
            margin-bottom: 20px !important; 
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
        }

        /* Primary Submit Button - Spotify Green */
        .gradio-app button.gr-button.lg.primary.svelte-cmf5ev,
        button.gr-button.lg.primary.svelte-cmf5ev { 
            background-color: #1DB954 !important; 
            color: #FFFFFF !important; 
            border-radius: 500px !important;
            font-weight: 700 !important; 
            padding: 10px 24px !important; 
            border: none !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: background-color 0.2s ease;
        }
        .gradio-app button.gr-button.lg.primary.svelte-cmf5ev:hover,
        button.gr-button.lg.primary.svelte-cmf5ev:hover {
            background-color: #1ED760 !important;
        }

        /* Secondary Button - White background with Green border */
        .gradio-app button.gr-button.lg.secondary.svelte-cmf5ev,
        button.gr-button.sm.secondary.svelte-cmf5ev { 
            background-color: #FFFFFF !important; 
            color: #1DB954 !important;
            border: 2px solid #1DB954 !important;
            border-radius: 500px !important;
            font-weight: 700 !important; 
            padding: 10px 24px !important; 
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: background-color 0.2s ease, color 0.2s ease;
        }
        .gradio-app button.gr-button.lg.secondary.svelte-cmf5ev:hover,
        button.gr-button.sm.secondary.svelte-cmf5ev:hover {
            background-color: #1DB954 !important;
            color: #FFFFFF !important;
        }

        .gr-input textarea { 
            border-radius: 6px !important; 
            border: 1px solid #B3B3B3 !important; 
            padding: 12px !important; 
            font-size: 1rem; 
            color: #191414 !important;
            background-color: #FFFFFF !important;
        }

        .gr-markdown { 
            padding: 15px; 
            border-radius: 8px; 
            background-color: #FFFFFF !important; 
            margin-top: 20px;
            border: 1px solid #E0E0E0 !important;
        }
        .gr-markdown p, .gr-markdown li {
            margin-bottom: 0.5em !important; 
            line-height: 1.6; 
            color: #191414 !important;
        }
        .gr-markdown strong {
            color: #1DB954 !important;
        }

        .gr-dataframe {
            border-radius: 8px !important; 
            box-shadow: none; 
            margin-top: 20px; 
            border: 1px solid #E0E0E0 !important;
        }
        .gr-dataframe table { border-collapse: separate; border-spacing: 0; width:100%; }
        .gr-dataframe th {
            background-color: #F0F0F0 !important; 
            color: #191414 !important; 
            font-weight: 600; 
            text-align: left; 
            padding: 10px 12px !important; 
            border-bottom: 1px solid #DCDCDC !important;
        }
        .gr-dataframe td {
            padding: 10px 12px !important; 
            border-bottom: 1px solid #EFEFEF; 
            color: #191414 !important;
        }
        .gr-dataframe tr:last-child td { border-bottom: none; }

        h1.gr-title {
            color: #191414 !important; 
            text-align: center; 
            font-size: 2rem; 
            font-weight: 700; 
            margin-bottom: 0.5rem !important;
        }
        .gr-description {
            color: #535353 !important; 
            text-align: center; 
            margin-bottom: 25px !important; 
            font-size: 1rem; 
            line-height: 1.5;
        }
        footer { visibility: hidden !important; }

        .gr-examples {
            padding: 15px; 
            background-color: #FFFFFF; 
            border-radius: 8px; 
            border: 1px solid #E0E0E0; 
            margin-top: 15px;
        }
        .gr-examples-label span {
            color: #191414 !important; 
            font-weight: 600;
        }
        .gr-example-inputs { padding: 5px 0px !important; }

        #emotion_prob_label {
            font-weight: bold; 
            margin-bottom: 5px; 
            color: #1DB954;
        }

        .gr-input > label > span.label-text,
        .gr-output > label > span.label-text,
        .gr-markdown > label > span.label-text,
        .gr-dataframe > label > span.label-text {
            color: #191414 !important; 
            font-weight: 600 !important;
        }
        """

        
        with gr.Blocks(theme=gr.themes.Base(), css=css_app_spotify_light) as demo: # Use Base theme and override with CSS
            gr.Markdown("🎤 Sentiment-Driven Multilingual Music Recommender 🎧", elem_classes="gr-title") 
            gr.Markdown(
                "Enter text in English, Russian, or Kazakh. Use [EN], [RU], [KZ] tags for best results. " 
                "The system predicts emotions and recommends songs based on audio feature rules. \n"
                "Ағылшын, орыс немесе қазақ тілдерінде мәтін енгізіңіз. Үздік нәтиже үшін [EN], [RU], [KZ] тегтерін қолданыңыз. "
                "Жүйе эмоцияларды анықтап, әндерді ұсынады. \n"
                "Введите текст на английском, русском или казахском языке. Для лучших результатов используйте теги [EN], [RU], [KZ]. "
                "Система определит эмоции и порекомендует песни.",
                elem_classes="gr-description"
            )

            with gr.Column(): 
                text_input = gr.Textbox(
                    lines=4, 
                    placeholder="E.g., '[EN] I'm so happy today!' or '[RU] Мне очень грустно.' or '[KZ] Көңіл-күйім тамаша!'...", 
                    label="Your text:" 
                )
                with gr.Row(): 
                    # Ensure Gradio uses the 'secondary' variant for ClearButton if available, or rely on general .gr-button for styling
                    clear_button = gr.ClearButton(value="Clear", components=[text_input], variant="secondary") # Attempt to use secondary variant
                    submit_button = gr.Button("Submit", variant="primary")
                
                examples_component = gr.Examples( 
                    examples=[
                        ["[EN] I feel ecstatic and overjoyed by this news!"],
                        ["[RU] Это очень печально и вызывает у меня тревогу."],
                        ["[KZ] Бүгін менің көңіл-күйім тамаша, жаңа ән тыңдағым келеді!"],
                        ["I am feeling quite down and disappointed."],
                        ["[RU] Какая прекрасная погода, хочется гулять и слушать музыку!"],
                        ["[KZ] Мен қатты шаршадым, біраз демалғым келеді."]
                    ],
                    inputs=[text_input],
                    label="Examples:" 
                )
                
                gr.Markdown("### Emotion Probabilities:", elem_id="emotion_prob_label") 
                emotion_output_markdown = gr.Markdown() 
                
                recommendation_df_output = gr.DataFrame(
                    headers=['Track Name', 'Artists'], 
                    label="🎵 Music Recommendations", 
                    wrap=True, 
                    row_count=(10,"dynamic"), 
                    col_count=(2,"fixed") 
                )

            submit_button.click(
                fn=self.get_recommendations_interface,
                inputs=[text_input],
                outputs=[emotion_output_markdown, recommendation_df_output]
            )
            clear_button.add(components=[emotion_output_markdown, recommendation_df_output]) # Clear outputs too

        demo.launch()

if __name__ == '__main__':
    app = GradioApp() 
    app.launch()      
