
"""
Translation and text-to-speech services.
"""
import streamlit as st
import requests
import tempfile
import os
from gradio_client import Client
from deep_translator import GoogleTranslator
from config.settings import (
    ELEVEN_LABS_API_KEY, ELEVEN_LABS_API_URL, VOICE_IDS,
    HUGGINGFACE_ENDPOINT
)

# Initialize the client with the Hugging Face API endpoint
try:
    client = Client(HUGGINGFACE_ENDPOINT)
except Exception as e:
    st.error(f"Failed to initialize translation client: {e}")
    client = None

def elevenlabs_text_to_speech(text: str, language: str = "english") -> str:
    """Convert text to speech using ElevenLabs API."""
    try:
        # Select the appropriate voice for the language
        voice_id = VOICE_IDS.get(language, VOICE_IDS["english"])
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVEN_LABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(
            f"{ELEVEN_LABS_API_URL}/{voice_id}",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                temp_audio.write(response.content)
                temp_audio_path = temp_audio.name
            
            return temp_audio_path
        else:
            st.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def play_audio(audio_path: str):
    """Play the audio file using Streamlit's audio component."""
    if audio_path and os.path.exists(audio_path):
        # Read audio file
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        # Display audio player in Streamlit
        st.audio(audio_bytes, format="audio/mp3")
        
        # Clean up the temporary file
        try:
            os.remove(audio_path)
        except:
            pass

def translate_gloss(gloss_list: list) -> str:
    """Translate a list of sign language glosses into a coherent English sentence."""
    if not client:
        return " ".join(gloss_list)
    
    try:
        # Convert list to a more natural sentence-like input
        gloss_sentence = " ".join(gloss_list)
        
        # Format the input prompt to generate a natural translation
        prompt = f"give me only answer.Translate the following sign language glosses '{gloss_sentence}' into a natural English sentence. Preserve the meaning while creating a grammatically correct sentence."
        
        # Call the API to process the gloss input
        result = client.predict(
            inputs=prompt,
            top_p=1,
            temperature=1,
            chat_counter=0,
            chatbot=[],
            api_name="/predict_1"
        )
        
        # Return the translated text
        return result[0][0][1]
    except Exception as e:
        st.error(f"Translation error: {e}")
        return " ".join(gloss_list)

def convert_to_gloss(sentence: str) -> tuple:
    """Convert a natural English sentence to ISL gloss using an AI model."""
    if not client:
        return sentence, {}
    
    try:
        prompt = f"give me only answer (dont put any punctuation signs). Convert the natural English sentence '{sentence}' into ISL gloss format while maintaining the original meaning."
        
        result = client.predict(
            inputs=prompt,
            top_p=1,
            temperature=1,
            chat_counter=0,
            chatbot=[],
            api_name="/predict_1"
        )
        
        gloss_output = result[0][0][1]
        
        # Import here to avoid circular imports
        from config.model_config import SIGN_VIDEOS
        
        # Map words in gloss to their respective video file paths
        gloss_with_videos = {word: SIGN_VIDEOS.get(word.lower(), None) for word in gloss_output.split()}
        
        return gloss_output, gloss_with_videos
    except Exception as e:
        st.error(f"Gloss conversion error: {e}")
        return sentence, {}

def translate_additional_language(text: str, target_language: str) -> str:
    """Translate text to specified language using deep_translator."""
    try:
        translator = GoogleTranslator(source='auto', target=target_language)
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text
