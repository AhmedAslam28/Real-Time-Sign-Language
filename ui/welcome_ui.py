
"""
Welcome page user interface.
"""
import streamlit as st

def show_welcome_page():
    """Display the welcome page."""
    st.title("Welcome to Indian Sign Language System")
    
    # Hero section with logo/image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("https://www.livelaw.in/h-upload/2019/07/09/750x450_362025-isl.jpg", use_column_width=True)
        except:
            st.info("📖 Indian Sign Language System")
    
    # Main description
    st.markdown("""
    ## Bridging Communication Gaps
    
    The Indian Sign Language (ISL) System is an innovative application designed to facilitate 
    communication between sign language users and non-signers. Our system offers multiple modes:
    
    ### 🖐️ Sign Recognition
    - Detect and interpret sign language gestures through your webcam
    - Build sentences from recognized signs
    - Translate sign language to text and speech
    
    ### 💬 Sign Generation
    - Convert natural language text or speech into Indian Sign Language videos
    - Support for multiple input languages including English, Tamil, and Telugu
    - Create continuous sign language sequences
    
    ### 📊 Data Collection
    - Collect training data for new signs
    - Structured data collection workflow
    - Support for multiple sign categories
    
    ### 🎯 Model Training
    - Train new models on collected data
    - Multiple model architectures supported
    - Real-time training progress monitoring
    
    ### 📈 Comparative Study
    - Compare different model architectures
    - Performance analysis and visualization
    - Model selection and optimization
    """)
    
    # Features section
    st.markdown("## Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Recognition Features
        - Real-time sign detection
        - Multiple model support
        - Sentence building
        - Text-to-speech translation
        """)
    
    with col2:
        st.markdown("""
        ### Generation Features
        - Text-to-sign conversion
        - Speech-to-sign conversion
        - Language translation
        - Combined video generation
        """)
    
    # Supported signs
    with st.expander("Supported Signs"):
        st.markdown("""
        ### Currently Available Signs
        - **Greetings**: hello, thanks, love
        - **Pronouns**: you, me
        - **Actions**: angry, bring, care, cry
        - **Questions**: why, how, what, where
        - **Others**: help, water, doing
        """)
    
    # Usage instructions
    with st.expander("How to Use"):
        st.markdown("""
        ### Sign Recognition Mode
        1. Select the models you want to use
        2. Start detection and perform signs in front of your webcam
        3. Add detected words to build a sentence
        4. Translate and listen to your sentence
        
        ### Sign Generation Mode
        1. Enter text or use speech recognition
        2. The system will convert to ISL gloss format
        3. Watch the combined sign language video
        
        ### Data Collection Mode
        1. Select the sign category to collect data for
        2. Follow the on-screen instructions
        3. Perform signs in front of your webcam
        
        ### Training Mode
        1. Select the data to train on
        2. Choose model architecture
        3. Monitor training progress
        
        ### Comparative Study Mode
        1. Select models to compare
        2. Run the comparative analysis
        3. View results and performance metrics
        """)
    
    # Tips section
    with st.expander("Tips for Best Results"):
        st.markdown("""
        ### For Sign Recognition & Data Collection
        - Ensure good lighting conditions
        - Position yourself clearly in the camera view
        - Perform signs slowly and clearly
        - Keep a consistent distance from the camera
        
        ### For Speech Input
        - Speak clearly and at a normal pace
        - Minimize background noise
        - Use a good quality microphone
        - Start speaking shortly after clicking the button
        """)
    
    # Start button
    st.markdown("## Ready to Begin?")
    if st.button("START APPLICATION", key="start_button", type="primary", use_container_width=True):
        st.session_state.app_started = True
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ by the ISL Team | © 2025")
