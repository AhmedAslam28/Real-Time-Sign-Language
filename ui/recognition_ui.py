
"""
Sign recognition user interface.
"""
import streamlit as st
import cv2
import time
from typing import List

from core.detector import SignLanguageDetector
from core.mediapipe_utils import mediapipe_detection, draw_landmarks, extract_keypoints, get_holistic_model
from services.translation_service import translate_gloss, elevenlabs_text_to_speech, play_audio, translate_additional_language
from config.model_config import MODELS_CONFIG
from config.settings import TRANSLATION_LANGUAGES
from utils.session_utils import reset_recognition_state

def add_feedback_interface(detector: SignLanguageDetector, sequence: List):
    """Add feedback interface for user corrections."""
    st.write("### Provide Feedback")
    correction = st.text_input("If the detection was incorrect, enter the correct sign:")
    
    if st.button("Submit Feedback"):
        if correction:
            detector.add_feedback(sequence, correction)
            st.success("Thank you for your feedback! This will help improve future detections.")

def run_sign_recognition_mode():
    """Run the sign recognition interface."""
    st.header("🖐️ Sign Language Recognition")
    
    # Sidebar with supported signs
    with st.sidebar:
        with st.expander("Supported Signs"):
            st.markdown("""
            ### Available Signs
            - **Greetings**: hello, thanks, love
            - **Pronouns**: you, me
            - **Actions**: angry, bring, care, cry
            - **Questions**: why, how, what, where
            - **Others**: help, water, doing
            """)
    
    # Model selection
    st.write("### Model Selection")
    selected_models = st.multiselect(
        "Choose which sign language models to use:",
        options=list(MODELS_CONFIG.keys()),
        default=list(MODELS_CONFIG.keys()),
        help="Select one or more models for sign detection"
    )

    # Initialize detector when models are selected
    if selected_models and st.session_state.detector is None:
        st.session_state.detector = SignLanguageDetector(selected_models)
    # Reinitialize detector if model selection changes
    elif selected_models and hasattr(st.session_state.detector, 'models') and set(selected_models) != set(st.session_state.detector.models.keys()):
        st.session_state.detector = SignLanguageDetector(selected_models)

    # Display current accumulated words
    st.write("### Current Sentence:")
    if st.session_state.accumulated_words:
        st.info(" ".join(st.session_state.accumulated_words))
    else:
        st.info("No words detected yet")

    # Control buttons
    col1, col2 = st.columns(2)
    start_button = col1.button("🎥 Start Detection", disabled=not selected_models, type="primary")
    clear_button = col2.button("🗑️ Clear All Words", type="secondary")

    if clear_button:
        reset_recognition_state()
        st.rerun()

    if start_button and st.session_state.detector:
        st.session_state.detection_active = True
        st.session_state.detected_words = set()
        st.session_state.detector.clear_frame_predictions()

    # Detection process
    if st.session_state.detection_active and st.session_state.detector:
        run_detection_process()

    # Word selection and sentence building
    if st.session_state.detected_words:
        handle_word_selection()

    # Display current sentence and translation options
    if st.session_state.accumulated_words:
        handle_translation_interface()
    
    # Continue detection options
    if not st.session_state.detection_active and st.session_state.detector:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Continue Detection", type="secondary"):
                st.session_state.detection_active = True
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Sentence", type="secondary"):
                reset_recognition_state()
                st.rerun()

def run_detection_process():
    """Run the real-time detection process."""
    detector = st.session_state.detector
    
    # Detection duration settings
    detection_duration = st.slider("Detection Duration (seconds)", 
                                 min_value=5, max_value=30, value=10,
                                 help="How long to run detection for")
    
    cap = cv2.VideoCapture(0)
    sequence = []
    
    # Create placeholders for detection results and webcam feed
    col1, col2 = st.columns([2, 1])
    with col1:
        webcam_placeholder = st.empty()
    with col2:
        detection_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    with get_holistic_model() as holistic:
        start_time = time.time()
        
        while time.time() - start_time < detection_duration:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera")
                break
            
            # Update progress bar
            progress = (time.time() - start_time) / detection_duration
            progress_bar.progress(progress)
            
            # Make detections and draw landmarks
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            
            # Extract keypoints and make prediction
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep only last 30 frames
            
            if len(sequence) == 30:
                predictions, confidences = detector.predict(sequence)
                st.session_state.last_sequence = sequence.copy()
                
                # Add detected words to set
                for word in predictions.values():
                    st.session_state.detected_words.add(word)
                
                # Display predictions on image
                for i, (model_key, word) in enumerate(predictions.items()):
                    confidence = confidences[model_key]
                    cv2.putText(image, f'{word} ({confidence:.2f})', 
                              (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the webcam feed with landmarks
            webcam_placeholder.image(image, channels="BGR", use_column_width=True)
            
            # Update detection results
            if st.session_state.detected_words:
                detection_placeholder.write(f"**Detected signs:** {', '.join(st.session_state.detected_words)}")
    
    cap.release()
    cv2.destroyAllWindows()
    st.session_state.detection_active = False

    # Get majority words after detection period
    majority_words = detector.get_majority_words()
    
    # Update detected words with majority words
    if majority_words:
        majority_word_set = set(majority_words.values())
        st.session_state.detected_words = majority_word_set
        st.success(f"✅ Recommended words based on majority detection: {', '.join(majority_word_set)}")

    # Add feedback interface after detection
    if st.session_state.last_sequence is not None:
        add_feedback_interface(detector, st.session_state.last_sequence)

def handle_word_selection():
    """Handle word selection interface."""
    st.write("### Add detected words to your sentence:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_words = st.multiselect(
            "Choose words to add:",
            options=list(st.session_state.detected_words),
            default=list(st.session_state.detected_words),
            help="Select which detected words to add to your sentence"
        )
    
    with col2:
        if st.button("➕ Add to Sentence", type="primary"):
            if selected_words:
                # Update accumulated words
                st.session_state.accumulated_words.extend(selected_words)
                st.session_state.accumulated_words = list(dict.fromkeys(st.session_state.accumulated_words))
                
                # Clear detected words to allow adding more
                st.session_state.detected_words = set()
                st.success("Words added to sentence!")
                st.rerun()

def handle_translation_interface():
    """Handle translation and TTS interface."""
    st.write("### Current Sentence:")
    st.info("**Glosses:** " + " ".join(st.session_state.accumulated_words))
    
    # Translate and Text-to-Speech buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔤 Translate Sentence", type="primary"):
            # Translate the entire accumulated sentence 
            translated_sentence = translate_gloss(st.session_state.accumulated_words)
            st.session_state.translated_sentence = translated_sentence
            
            # Generate text-to-speech for English translation
            st.session_state.audio_file_path = elevenlabs_text_to_speech(translated_sentence, "english")
    
    with col2:
        if st.button("🔊 Speak Translation", type="secondary", 
                    disabled=not hasattr(st.session_state, 'audio_file_path') or not st.session_state.audio_file_path):
            # Play the previously generated audio
            play_audio(st.session_state.audio_file_path)
    
    # Display translation
    if hasattr(st.session_state, 'translated_sentence') and st.session_state.translated_sentence:
        st.success("**Translation:** " + st.session_state.translated_sentence)
        
        # Additional language translation options
        st.write("### Translate to Additional Language:")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_language = st.selectbox(
                "Choose a language to translate to:",
                list(TRANSLATION_LANGUAGES.keys()),
                help="Select target language for additional translation"
            )
        
        with col2:
            if st.button("🌐 Translate and Speak", type="secondary"):
                target_language_code = TRANSLATION_LANGUAGES[selected_language]
                additional_translation = translate_additional_language(
                    st.session_state.translated_sentence, 
                    target_language_code
                )
                st.write(f"**{selected_language} Translation:** {additional_translation}")
                
                # Generate and play TTS for the translated text
                audio_path = elevenlabs_text_to_speech(additional_translation, target_language_code)
                if audio_path:
                    play_audio(audio_path)
