
"""
Speech recognition service.
"""
import streamlit as st
import speech_recognition as sr
from typing import Tuple, Optional

def recognize_speech(language_code: str) -> Tuple[Optional[str], Optional[str]]:
    """Recognize speech and return the recognized text with improved error handling."""
    recognizer = sr.Recognizer()
    
    # Create a placeholder for real-time feedback
    status_placeholder = st.empty()
    
    try:
        with sr.Microphone() as source:
            status_placeholder.info("Adjusting for ambient noise... Please wait.")
            # Longer adjustment for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            status_placeholder.info("Ready! Please speak clearly. (Listening will timeout after 10 seconds of silence)")
            
            try:
                # Increase timeout values
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
                status_placeholder.info("Processing speech...")
                
                # Increase recognition timeout
                text = recognizer.recognize_google(audio, language=language_code, show_all=False)
                
                status_placeholder.success("Speech recognized successfully!")
                return text, text
            
            except sr.WaitTimeoutError:
                status_placeholder.error("Listening timed out. Please try again and speak clearly when prompted.")
                return None, None
    
    except sr.UnknownValueError:
        status_placeholder.error("Could not understand the audio. Please speak more clearly and try again.")
        return None, None
    except sr.RequestError as e:
        status_placeholder.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None, None
    except Exception as e:
        status_placeholder.error(f"An error occurred: {str(e)}")
        return None, None
