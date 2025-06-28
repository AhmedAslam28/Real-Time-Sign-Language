
"""
Model training user interface.
"""
import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR, LOGS_DIR, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_PATIENCE, TEST_SIZE, VALIDATION_SPLIT
from config.model_config import DATA_COLLECTION_CONFIGS

def load_training_data(actions: list, sequence_length: int = 30):
    """Load and preprocess training data."""
    # Create label mapping
    label_map = {label: num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    
    # Load sequences
    for action in actions:
        action_path = os.path.join(DATA_DIR, action)
        if not os.path.exists(action_path):
            continue
            
        sequence_dirs = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        
        for sequence_dir in sequence_dirs:
            sequence_path = os.path.join(action_path, sequence_dir)
            window = []
            
            # Load frames for this sequence
            for frame_num in range(sequence_length):
                frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    frame_data = np.load(frame_path)
                    window.append(frame_data)
                else:
                    # If frame doesn't exist, use zeros (shouldn't happen with proper collection)
                    window.append(np.zeros(1662))  # MediaPipe keypoints size
            
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
    
    if not sequences:
        return None, None, actions
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    return X, y, actions

def create_lstm_model(input_shape, num_classes):
    """Create LSTM model architecture."""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

def run_training_ui():
    """Run the training interface."""
    st.header("🎯 Model Training")
    st.markdown("Train new sign language recognition models")
    
    # Check available data
    available_actions = check_available_training_data()
    
    if not available_actions:
        st.warning("⚠️ No training data found. Please collect data first using the Data Collection mode.")
        return
    
    # Training configuration
    st.write("### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_actions = st.multiselect(
            "Select actions to train on:",
            options=available_actions,
            default=available_actions[:3] if len(available_actions) >= 3 else available_actions,
            help="Choose which actions to include in the training"
        )
        
        model_name = st.text_input(
            "Model name:",
            value="custom_sign_model",
            help="Name for the trained model file"
        )
    
    with col2:
        epochs = st.number_input(
            "Number of epochs:",
            min_value=10,
            max_value=2000,
            value=DEFAULT_EPOCHS,
            help="Number of training epochs"
        )
        
        batch_size = st.number_input(
            "Batch size:",
            min_value=8,
            max_value=64,
            value=DEFAULT_BATCH_SIZE,
            help="Training batch size"
        )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            patience = st.number_input(
                "Early stopping patience:",
                min_value=5,
                max_value=50,
                value=DEFAULT_PATIENCE,
                help="Number of epochs with no improvement after which training will be stopped"
            )
        
        with col2:
            sequence_length = st.number_input(
                "Sequence length:",
                min_value=15,
                max_value=60,
                value=30,
                help="Number of frames per sequence"
            )
    
    # Start training button
    if st.button("🚀 Start Training", type="primary", disabled=not selected_actions):
        if selected_actions:
            train_model(selected_actions, model_name, epochs, batch_size, patience, sequence_length)

def check_available_training_data():
    """Check what training data is available."""
    available_actions = []
    
    for category, config in DATA_COLLECTION_CONFIGS.items():
        for action in config['actions']:
            action_path = os.path.join(DATA_DIR, action)
            if os.path.exists(action_path):
                sequences = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
                if len(sequences) >= 5:  # Minimum sequences required
                    available_actions.append(action)
    
    return available_actions

def train_model(actions, model_name, epochs, batch_size, patience, sequence_length):
    """Train the model with the specified parameters."""
    st.write("### 🔄 Training Progress")
    
    # Load data
    with st.spinner("Loading training data..."):
        X, y, action_labels = load_training_data(actions, sequence_length)
    
    if X is None:
        st.error("❌ Failed to load training data. Please check that data exists for the selected actions.")
        return
    
    st.success(f"✅ Loaded {len(X)} sequences for {len(actions)} actions")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    
    st.info(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(actions)
    
    model = create_lstm_model(input_shape, num_classes)
    
    # Display model architecture
    with st.expander("🏗️ Model Architecture"):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.text('\n'.join(model_summary))
    
    # Setup callbacks
    log_dir = os.path.join(LOGS_DIR, model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        TensorBoard(log_dir=log_dir),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Create progress placeholders
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Custom callback for progress updates
    class StreamlitCallback:
        def __init__(self):
            self.epoch = 0
            self.logs_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        def on_epoch_end(self, epoch, logs=None):
            self.epoch = epoch + 1
            progress = self.epoch / epochs
            progress_bar.progress(progress)
            
            # Store metrics
            if logs:
                self.logs_history['loss'].append(logs.get('loss', 0))
                self.logs_history['accuracy'].append(logs.get('categorical_accuracy', 0))
                self.logs_history['val_loss'].append(logs.get('val_loss', 0))
                self.logs_history['val_accuracy'].append(logs.get('val_categorical_accuracy', 0))
                
                # Update metrics display
                metrics_placeholder.write(f"""
                **Epoch {self.epoch}/{epochs}**
                - Loss: {logs.get('loss', 0):.4f}
                - Accuracy: {logs.get('categorical_accuracy', 0):.4f}
                - Val Loss: {logs.get('val_loss', 0):.4f}
                - Val Accuracy: {logs.get('val_categorical_accuracy', 0):.4f}
                """)
                
                # Update chart
                if self.epoch > 1:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Loss plot
                    ax1.plot(self.logs_history['loss'], label='Training Loss')
                    ax1.plot(self.logs_history['val_loss'], label='Validation Loss')
                    ax1.set_title('Model Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    
                    # Accuracy plot
                    ax2.plot(self.logs_history['accuracy'], label='Training Accuracy')
                    ax2.plot(self.logs_history['val_accuracy'], label='Validation Accuracy')
                    ax2.set_title('Model Accuracy')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()
                    
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)
    
    streamlit_callback = StreamlitCallback()
    
    # Start training
    try:
        st.info("🚀 Starting training...")
        
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=0  # We'll handle progress display ourselves
        )
        
        # Manually update progress for each epoch
        for epoch in range(len(history.history['loss'])):
            streamlit_callback.on_epoch_end(epoch, {
                'loss': history.history['loss'][epoch],
                'categorical_accuracy': history.history['categorical_accuracy'][epoch],
                'val_loss': history.history['val_loss'][epoch],
                'val_categorical_accuracy': history.history['val_categorical_accuracy'][epoch]
            })
        
        # Evaluate model
        st.write("### 📊 Model Evaluation")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Loss", f"{test_loss:.4f}")
        with col2:
            st.metric("Test Accuracy", f"{test_accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
        model.save(model_path)
        
        st.success(f"🎉 Training completed! Model saved as '{model_name}.h5'")
        
        # Save training summary
        summary_data = {
            'Model Name': [model_name],
            'Actions': [', '.join(actions)],
            'Training Samples': [len(X_train)],
            'Test Samples': [len(X_test)],
            'Epochs Trained': [len(history.history['loss'])],
            'Final Test Accuracy': [test_accuracy],
            'Final Test Loss': [test_loss]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.write("### 📋 Training Summary")
        st.dataframe(summary_df)
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")

def run_training_mode():
    """Main function for training mode."""
    run_training_ui()
