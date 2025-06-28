
"""
Data collection user interface.
"""
import streamlit as st
import cv2
import numpy as np
import os
import time
from core.mediapipe_utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, get_holistic_model
from config.model_config import DATA_COLLECTION_CONFIGS
from config.settings import DATA_DIR

def run_data_collection_mode():
    """Run the data collection interface."""
    st.header("📊 Data Collection")
    st.markdown("Collect training data for sign language models")
    
    # Category selection
    st.write("### Select Sign Category")
    selected_category = st.selectbox(
        "Choose which category to collect data for:",
        options=list(DATA_COLLECTION_CONFIGS.keys()),
        help="Select the sign category you want to collect data for"
    )
    
    if selected_category:
        config = DATA_COLLECTION_CONFIGS[selected_category]
        
        # Display category info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Actions:** {', '.join(config['actions'])}")
        with col2:
            st.info(f"**Sequences per action:** {config['sequences']}")
        
        # Action selection
        selected_action = st.selectbox(
            "Choose which action to collect data for:",
            options=config['actions'],
            help="Select the specific sign/action to collect data for"
        )
        
        # Collection parameters
        col1, col2 = st.columns(2)
        with col1:
            num_sequences = st.number_input(
                "Number of sequences to collect:", 
                min_value=1, 
                max_value=50, 
                value=config['sequences'],
                help="How many sequences to collect for this action"
            )
        with col2:
            sequence_length = st.number_input(
                "Frames per sequence:", 
                min_value=10, 
                max_value=60, 
                value=config['sequence_length'],
                help="Number of frames in each sequence"
            )
        
        # Start collection button
        if st.button("🎥 Start Data Collection", type="primary"):
            st.session_state.collection_active = True
            collect_data_for_action(selected_action, num_sequences, sequence_length)

def collect_data_for_action(action: str, num_sequences: int, sequence_length: int):
    """Collect data for a specific action."""
    # Create data directory
    action_path = os.path.join(DATA_DIR, action)
    os.makedirs(action_path, exist_ok=True)
    
    # Create sequence directories
    for sequence in range(1, num_sequences + 1):
        sequence_path = os.path.join(action_path, str(sequence))
        os.makedirs(sequence_path, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not open camera. Please check your camera connection.")
        return
    
    # Create placeholders
    video_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Overall progress
    total_frames = num_sequences * sequence_length
    frames_collected = 0
    
    try:
        with get_holistic_model() as holistic:
            # Loop through sequences
            for sequence in range(1, num_sequences + 1):
                status_placeholder.info(f"🎬 Preparing to collect sequence {sequence}/{num_sequences} for action: **{action}**")
                
                # Countdown before starting sequence
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if ret:
                        # Make detections and draw landmarks
                        image, results = mediapipe_detection(frame, holistic)
                        draw_styled_landmarks(image, results)
                        
                        # Add countdown text
                        cv2.putText(image, f'Starting in {countdown}...', (120, 200), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                        
                        video_placeholder.image(image, channels="BGR", use_column_width=True)
                        time.sleep(1)
                
                # Collect frames for this sequence
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to grab frame from camera")
                        break
                    
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    # Add collection status text
                    cv2.putText(image, f'Collecting: {action}', (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Sequence: {sequence}/{num_sequences}', (15, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Frame: {frame_num + 1}/{sequence_length}', (15, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Display the frame
                    video_placeholder.image(image, channels="BGR", use_column_width=True)
                    
                    # Extract and save keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_DIR, action, str(sequence), f"{frame_num}.npy")
                    np.save(npy_path, keypoints)
                    
                    # Update progress
                    frames_collected += 1
                    progress = frames_collected / total_frames
                    progress_placeholder.progress(progress)
                    
                    # Small delay for better visualization
                    time.sleep(0.1)
                
                # Brief pause between sequences
                if sequence < num_sequences:
                    status_placeholder.success(f"✅ Sequence {sequence} completed! Preparing for next sequence...")
                    time.sleep(2)
        
        # Collection completed
        cap.release()
        status_placeholder.success(f"🎉 Data collection completed! Collected {frames_collected} frames for action: **{action}**")
        
        # Show collection summary
        st.balloons()
        st.success(f"✅ Successfully collected {num_sequences} sequences with {sequence_length} frames each for action '{action}'")
        
    except Exception as e:
        cap.release()
        st.error(f"❌ Error during data collection: {str(e)}")
    
    finally:
        st.session_state.collection_active = False

def show_data_collection_status():
    """Show current data collection status."""
    st.write("### 📈 Data Collection Status")
    
    total_actions = 0
    collected_actions = 0
    
    for category, config in DATA_COLLECTION_CONFIGS.items():
        with st.expander(f"📂 {category}"):
            for action in config['actions']:
                action_path = os.path.join(DATA_DIR, action)
                total_actions += 1
                
                if os.path.exists(action_path):
                    sequences = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
                    if sequences:
                        collected_actions += 1
                        st.success(f"✅ {action}: {len(sequences)} sequences collected")
                    else:
                        st.info(f"📁 {action}: Directory exists but no sequences found")
                else:
                    st.warning(f"❌ {action}: No data collected")
    
    # Overall progress
    if total_actions > 0:
        progress = collected_actions / total_actions
        st.progress(progress)
        st.info(f"Overall Progress: {collected_actions}/{total_actions} actions have data ({progress:.1%})")

# Add status display to the main function
def run_data_collection_mode():
    """Run the data collection interface."""
    st.header("📊 Data Collection")
    st.markdown("Collect training data for sign language models")
    
    # Show current status
    show_data_collection_status()
    st.divider()
    
    # Category selection
    st.write("### Select Sign Category")
    selected_category = st.selectbox(
        "Choose which category to collect data for:",
        options=list(DATA_COLLECTION_CONFIGS.keys()),
        help="Select the sign category you want to collect data for"
    )
    
    if selected_category:
        config = DATA_COLLECTION_CONFIGS[selected_category]
        
        # Display category info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Actions:** {', '.join(config['actions'])}")
        with col2:
            st.info(f"**Sequences per action:** {config['sequences']}")
        
        # Action selection
        selected_action = st.selectbox(
            "Choose which action to collect data for:",
            options=config['actions'],
            help="Select the specific sign/action to collect data for"
        )
        
        # Collection parameters
        col1, col2 = st.columns(2)
        with col1:
            num_sequences = st.number_input(
                "Number of sequences to collect:", 
                min_value=1, 
                max_value=50, 
                value=config['sequences'],
                help="How many sequences to collect for this action"
            )
        with col2:
            sequence_length = st.number_input(
                "Frames per sequence:", 
                min_value=10, 
                max_value=60, 
                value=config['sequence_length'],
                help="Number of frames in each sequence"
            )
        
        # Start collection button
        if st.button("🎥 Start Data Collection", type="primary"):
            collect_data_for_action(selected_action, num_sequences, sequence_length)
