
"""
Sign generation user interface.
"""
import streamlit as st
import os
from services.translation_service import convert_to_gloss
from services.speech_service import recognize_speech
from services.video_service import combine_videos
from config.settings import LANGUAGE_OPTIONS
from utils.session_utils import reset_generation_state

def run_sign_generation_mode():
    """Run the sign generation interface."""
    st.header("💬 Sign Language Generation")
    st.markdown("Convert text or speech to Indian Sign Language videos")
    
    # Initialize session state variables for generation mode if they don't exist
    if 'generated_gloss_result' not in st.session_state:
        st.session_state.generated_gloss_result = None
    if 'combined_video_path' not in st.session_state:
        st.session_state.combined_video_path = os.path.join(st.session_state.temp_dir, "combined_video.mp4")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:", 
        ["Text Input", "Speech Input"], 
        help="Select how you want to provide input"
    )
    
    current_input_method = input_method
    
    if 'last_input_method' not in st.session_state:
        st.session_state.last_input_method = current_input_method
    elif st.session_state.last_input_method != current_input_method:
        # Input method changed, reset results
        reset_generation_state()
        st.session_state.last_input_method = current_input_method
    
    try:
        if input_method == "Text Input":
            handle_text_input()
        elif input_method == "Speech Input":
            handle_speech_input()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def handle_text_input():
    """Handle text input processing."""
    # Text input
    sentence_input = st.text_input(
        "Enter a natural English sentence:", 
        "", 
        help="Type the sentence you want to convert to sign language"
    )
    
    process_button = st.button("🔄 Convert to ISL Gloss", type="primary")
    
    if process_button and sentence_input.strip():
        # Clear previous results
        st.session_state.generated_gloss_result = None
        
        with st.spinner("Converting to ISL Gloss..."):
            # Generate new gloss and store it
            gloss_result, video_mapping = convert_to_gloss(sentence_input)
            st.session_state.generated_gloss_result = gloss_result
            
        # Display and process results
        display_generation_results(gloss_result, video_mapping)

def handle_speech_input():
    """Handle speech input processing."""
    st.markdown("""
    ### 🎤 Speech Recognition Tips:
    - Speak clearly and at a normal pace
    - Make sure your microphone is properly connected
    - Reduce background noise if possible
    - Start speaking shortly after clicking the button
    """)
    
    selected_language = st.selectbox(
        "Select language for speech recognition:", 
        list(LANGUAGE_OPTIONS.keys()),
        help="Choose the language you will speak in"
    )
    language_code = LANGUAGE_OPTIONS[selected_language]
    
    if selected_language != "English":
        st.warning("⚠️ Note: Currently, non-English speech input will not be automatically translated.")
    
    if st.button("🎙️ Start Speech Recognition", type="primary"):
        # Clear previous results
        st.session_state.generated_gloss_result = None
        
        original_text, english_text = recognize_speech(language_code)
        
        if original_text and english_text:
            st.success("✅ Speech recognized successfully!")
            
            st.subheader("Recognized speech:")
            st.info(original_text)
            
            # Process the recognized text
            with st.spinner("Converting to ISL Gloss..."):
                gloss_result, video_mapping = convert_to_gloss(original_text)
            
            # Display and process results
            display_generation_results(gloss_result, video_mapping)
        else:
            st.error("❌ Speech recognition failed. Please try again or use text input instead.")

def display_generation_results(gloss_result: str, video_mapping: dict):
    """Display the generation results with videos."""
    # Display gloss result
    st.subheader("📝 ISL Gloss:")
    st.success(gloss_result)
    
    # Extract available video paths in correct order
    valid_video_paths = []
    missing_words = []
    
    for word in gloss_result.split():
        video_path = video_mapping.get(word)
        if video_path and os.path.exists(video_path):
            valid_video_paths.append(video_path)
        else:
            missing_words.append(word)
    
    # Show combined video if available
    if valid_video_paths:
        st.subheader("🎬 Combined Sign Language Video:")
        with st.spinner("Generating combined video..."):
            combined_path = combine_videos(valid_video_paths, st.session_state.combined_video_path)
            
        if combined_path and os.path.exists(combined_path):
            st.video(combined_path)
            st.success("✅ Combined video generated successfully!")
        else:
            st.error("❌ Failed to create combined video.")
        
        # Show individual videos for reference
        with st.expander("👁️ Show Individual Videos"):
            cols = st.columns(3)  # Create 3 columns for better layout
            col_idx = 0
            
            for word, video_path in video_mapping.items():
                with cols[col_idx % 3]:
                    if video_path and os.path.exists(video_path):
                        st.write(f"**{word.capitalize()}**")
                        st.video(video_path)
                    else:
                        st.write(f"**{word.capitalize()}**")
                        st.info("No video available")
                col_idx += 1
    
    # Display missing words notification
    if missing_words:
        st.warning(f"⚠️ No videos available for: {', '.join(missing_words)}")
        st.info("💡 These words will be skipped in the combined video.")
