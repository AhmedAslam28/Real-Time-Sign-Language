
"""
Session state management utilities.
"""
import streamlit as st
import tempfile
import shutil
import os

def initialize_session_state():
    """Initialize all session state variables."""
    # App control
    if 'app_started' not in st.session_state:
        st.session_state.app_started = False
    
    # Temp directory
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    # Recognition mode states
    if 'accumulated_words' not in st.session_state:
        st.session_state.accumulated_words = []
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'detected_words' not in st.session_state:
        st.session_state.detected_words = set()
    if 'last_sequence' not in st.session_state:
        st.session_state.last_sequence = None
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'translated_words' not in st.session_state:
        st.session_state.translated_words = []
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
    if 'translated_sentence' not in st.session_state:
        st.session_state.translated_sentence = None
    
    # Generation mode states
    if 'generated_gloss_result' not in st.session_state:
        st.session_state.generated_gloss_result = None
    if 'combined_video_path' not in st.session_state:
        st.session_state.combined_video_path = os.path.join(st.session_state.temp_dir, "combined_video.mp4")
    if 'last_input_method' not in st.session_state:
        st.session_state.last_input_method = None
    
    # Data collection states
    if 'collection_active' not in st.session_state:
        st.session_state.collection_active = False
    if 'collection_progress' not in st.session_state:
        st.session_state.collection_progress = {}
    
    # Training states
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = {}
    
    # Comparative study states
    if 'study_active' not in st.session_state:
        st.session_state.study_active = False
    if 'study_results' not in st.session_state:
        st.session_state.study_results = None

def cleanup_session():
    """Clean up session resources."""
    if hasattr(st.session_state, 'temp_dir') and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass
    
    # Clean up audio file if it exists
    if hasattr(st.session_state, 'audio_file_path') and st.session_state.audio_file_path:
        if os.path.exists(st.session_state.audio_file_path):
            try:
                os.remove(st.session_state.audio_file_path)
            except:
                pass

def reset_recognition_state():
    """Reset recognition mode session state."""
    st.session_state.accumulated_words = []
    st.session_state.detected_words = set()
    st.session_state.last_sequence = None
    st.session_state.translated_sentence = None
    if st.session_state.detector:
        st.session_state.detector.clear_frame_predictions()
    
    # Clean up audio file
    if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
        try:
            os.remove(st.session_state.audio_file_path)
        except:
            pass
    st.session_state.audio_file_path = None

def reset_generation_state():
    """Reset generation mode session state."""
    st.session_state.generated_gloss_result = None
    st.session_state.last_input_method = None
