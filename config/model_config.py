
"""
Model configurations and mappings.
"""
import os
from config.settings import MODELS_DIR, SIGN_VIDEOS_DIR

# Model configurations
MODELS_CONFIG = {
    'Greetings': {
        'path': os.path.join(MODELS_DIR, 'sign_language_model.h5'),
        'actions': ['hello', 'thanks', 'love']
    },
    'Pronouns': {
        'path': os.path.join(MODELS_DIR, 'Pronoun_sign_language_model.h5'),
        'actions': ['you', 'me', 'no sign']
    },
    'Actions': {
        'path': os.path.join(MODELS_DIR, 'verb_sign_language_model.h5'),
        'actions': ['angry', 'bring', 'care', 'cry']
    },
    'Question': {
        'path': os.path.join(MODELS_DIR, 'Question_sign_language_model.h5'),
        'actions': ['why', 'how\what\where', 'no sign']
    },
    'Other': {
        'path': os.path.join(MODELS_DIR, 'other_sign_language_model.h5'),
        'actions': ['help', 'water', 'doing']
    }
}

# Sign language video mappings
SIGN_VIDEOS = {
    "hello": os.path.join(SIGN_VIDEOS_DIR, "hello.mp4"),
    "thanks": os.path.join(SIGN_VIDEOS_DIR, "thanks.mp4"),
    "love": os.path.join(SIGN_VIDEOS_DIR, "love.mp4"),
    "you": os.path.join(SIGN_VIDEOS_DIR, "YOU.mp4"),
    "me": os.path.join(SIGN_VIDEOS_DIR, "me1.mp4"),
    "angry": os.path.join(SIGN_VIDEOS_DIR, "angry.mp4"),
    "bring": os.path.join(SIGN_VIDEOS_DIR, "bring.mp4"),
    "why": os.path.join(SIGN_VIDEOS_DIR, "why.mp4"),
    "how": os.path.join(SIGN_VIDEOS_DIR, "what.mp4"),
    "what": os.path.join(SIGN_VIDEOS_DIR, "what.mp4"),
    "where": os.path.join(SIGN_VIDEOS_DIR, "what.mp4"),
    "care": os.path.join(SIGN_VIDEOS_DIR, "Care.mp4"),
    "cry": os.path.join(SIGN_VIDEOS_DIR, "Cry.mp4"),
    "help": os.path.join(SIGN_VIDEOS_DIR, "help.mp4"),
    "water": os.path.join(SIGN_VIDEOS_DIR, "water.mp4"),
    "doing": os.path.join(SIGN_VIDEOS_DIR, "doing.mp4")
}

# Data collection configurations
DATA_COLLECTION_CONFIGS = {
    'Greetings': {
        'actions': ['hello', 'thanks', 'love'],
        'sequences': 30,
        'sequence_length': 30
    },
    'Pronouns': {
        'actions': ['you', 'me', 'no sign'],
        'sequences': 30,
        'sequence_length': 30
    },
    'Actions': {
        'actions': ['angry', 'bring', 'care', 'cry'],
        'sequences': 30,
        'sequence_length': 30
    },
    'Questions': {
        'actions': ['why', 'how\what\where', 'no sign'],
        'sequences': 30,
        'sequence_length': 30
    },
    'Others': {
        'actions': ['help', 'water', 'doing'],
        'sequences': 30,
        'sequence_length': 30
    }
}
