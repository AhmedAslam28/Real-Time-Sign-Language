
"""
Sign Language System - Main Application Entry Point
"""
import streamlit as st
from ui.welcome_ui import show_welcome_page
from ui.recognition_ui import run_sign_recognition_mode
from ui.generation_ui import run_sign_generation_mode
from ui.data_collection_ui import run_data_collection_mode
from ui.training_ui import run_training_mode
from ui.comparative_study_ui import run_comparative_study_mode
from utils.session_utils import initialize_session_state, cleanup_session
from config.settings import PAGE_CONFIG
import atexit

# Set page configuration
st.set_page_config(**PAGE_CONFIG)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Check if the app has been started
    if not st.session_state.app_started:
        show_welcome_page()
    else:
        run_main_application()

def run_main_application():
    """Run the main application with mode selection."""
    st.title("Indian Sign Language System")
    
    # Mode selection tabs
    mode_tabs = st.tabs([
        "🖐️ Sign Recognition", 
        "💬 Sign Generation", 
        "📊 Data Collection",
        "🎯 Model Training",
        "📈 Comparative Study"
    ])
    
    with mode_tabs[0]:
        run_sign_recognition_mode()
    
    with mode_tabs[1]:
        run_sign_generation_mode()
    
    with mode_tabs[2]:
        run_data_collection_mode()
    
    with mode_tabs[3]:
        run_training_mode()
    
    with mode_tabs[4]:
        run_comparative_study_mode()
    
    # Add sidebar with return to welcome
    with st.sidebar:
        st.markdown("---")
        if st.button("🏠 Return to Welcome", type="secondary", use_container_width=True):
            st.session_state.app_started = False
            st.rerun()

# Register cleanup function
atexit.register(cleanup_session)

if __name__ == "__main__":
    main()
