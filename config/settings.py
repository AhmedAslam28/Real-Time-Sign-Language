
"""
Application configuration and settings.
"""
import os

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Sign Language System",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ElevenLabs API configuration
ELEVEN_LABS_API_KEY = "sk_24a6b33acc2241bce3a4bcb52a9240ffdd27e9a7d3afd5a1"
ELEVEN_LABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Voice IDs for different languages
VOICE_IDS = {
    "english": "21m00Tcm4TlvDq8ikWAM",
    "ta": "IKne3meq5aSn9XLyUdCD",
    "te": "AZnzlk1XvdvUeBnXmlld"
}

# Directory paths
MODELS_DIR = "models"
ASSETS_DIR = "assets"
SIGN_VIDEOS_DIR = os.path.join(ASSETS_DIR, "sign_vd")
DATA_DIR = "MP_Data"
LOGS_DIR = "Logs"
RESULTS_DIR = "Results"

# Detection parameters
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.7
TEMPORAL_THRESHOLD = 3

# Training parameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_PATIENCE = 20
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

# Data collection parameters
DEFAULT_SEQUENCES = 30
DEFAULT_SEQUENCE_LENGTH = 30

# Language options
LANGUAGE_OPTIONS = {
    "English": "en-IN",
    "Telugu": "te-IN",
    "Tamil": "ta-IN"
}

TRANSLATION_LANGUAGES = {
    'Tamil': 'ta',
    'Telugu': 'te'
}

# HuggingFace API endpoint
HUGGINGFACE_ENDPOINT = "yuntian-deng/ChatGPT"

# Create necessary directories
for directory in [MODELS_DIR, ASSETS_DIR, DATA_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
