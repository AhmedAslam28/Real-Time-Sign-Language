
"""
Comparative study user interface.
"""
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import GRU, Bidirectional, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from config.settings import DATA_DIR, MODELS_DIR, RESULTS_DIR, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_PATIENCE, TEST_SIZE
from ui.training_ui import load_training_data

def create_original_model(input_shape, num_classes):
    """Create the original LSTM model architecture."""
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

def create_cnn_model(input_shape, num_classes):
    """Create 1D CNN model architecture."""
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
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

def create_cnn_lstm_model(input_shape, num_classes):
    """Create CNN-LSTM hybrid model architecture."""
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True, activation='relu'),
        LSTM(32, return_sequences=False, activation='relu'),
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

def create_bidirectional_lstm_model(input_shape, num_classes):
    """Create Bidirectional LSTM model architecture."""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
        Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
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

def create_transformer_model(input_shape, num_classes):
    """Create a simple Transformer-based model."""
    inputs = Input(shape=input_shape)
    
    # Project input to a smaller dimension for attention
    projection = Dense(64, activation='relu')(inputs)
    
    # Self-attention mechanism
    attention_output = MultiHeadAttention(
        num_heads=8, key_dim=64
    )(projection, projection)
    
    # Add & Normalize
    attention_output = LayerNormalization()(projection + attention_output)
    
    # Feed-forward network
    x = Dense(128, activation='relu')(attention_output)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # Add & Normalize
    x = LayerNormalization()(attention_output + x)
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

def run_comparative_study_mode():
    """Run the comparative study interface."""
    st.header("📈 Comparative Study")
    st.markdown("Compare different model architectures for sign language recognition")
    
    # Check available data
    available_actions = check_available_data()
    
    if not available_actions:
        st.warning("⚠️ No training data found. Please collect data first using the Data Collection mode.")
        return
    
    # Study configuration
    st.write("### Study Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_actions = st.multiselect(
            "Select actions for comparison:",
            options=available_actions,
            default=available_actions[:3] if len(available_actions) >= 3 else available_actions,
            help="Choose which actions to include in the comparison study"
        )
        
        study_name = st.text_input(
            "Study name:",
            value="comparative_study",
            help="Name for this comparative study"
        )
    
    with col2:
        epochs = st.number_input(
            "Training epochs per model:",
            min_value=10,
            max_value=500,
            value=100,
            help="Number of epochs to train each model"
        )
        
        batch_size = st.number_input(
            "Batch size:",
            min_value=8,
            max_value=64,
            value=DEFAULT_BATCH_SIZE,
            help="Training batch size"
        )
    
    # Model selection
    st.write("### Model Selection")
    model_options = {
        'Proposed LSTM Model': create_original_model,
        'CNN': create_cnn_model,
        'CNN-LSTM Hybrid': create_cnn_lstm_model,
        'Bidirectional LSTM': create_bidirectional_lstm_model,
        'Transformer': create_transformer_model
    }
    
    selected_models = st.multiselect(
        "Select models to compare:",
        options=list(model_options.keys()),
        default=list(model_options.keys()),
        help="Choose which model architectures to compare"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        patience = st.number_input(
            "Early stopping patience:",
            min_value=5,
            max_value=50,
            value=20,
            help="Epochs with no improvement before stopping"
        )
        
        sequence_length = st.number_input(
            "Sequence length:",
            min_value=15,
            max_value=60,
            value=30,
            help="Number of frames per sequence"
        )
    
    # Start study button
    if st.button("🚀 Start Comparative Study", type="primary", disabled=not selected_actions or not selected_models):
        if selected_actions and selected_models:
            run_comparative_study(
                selected_actions, 
                selected_models, 
                model_options,
                study_name, 
                epochs, 
                batch_size, 
                patience, 
                sequence_length
            )

def check_available_data():
    """Check available training data."""
    from ui.training_ui import check_available_training_data
    return check_available_training_data()

def run_comparative_study(actions, selected_models, model_options, study_name, epochs, batch_size, patience, sequence_length):
    """Run the comparative study."""
    st.write("### 🔄 Comparative Study Progress")
    
    # Load data
    with st.spinner("Loading training data..."):
        X, y, action_labels = load_training_data(actions, sequence_length)
    
    if X is None:
        st.error("❌ Failed to load training data.")
        return
    
    st.success(f"✅ Loaded {len(X)} sequences for {len(actions)} actions")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    
    # Define input shape and number of classes
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(actions)
    
    # Store results
    results = {
        'model_name': [],
        'accuracy': [],
        'val_accuracy': [],
        'training_time': [],
        'history': []
    }
    
    # Create results directory
    study_dir = os.path.join(RESULTS_DIR, study_name)
    os.makedirs(study_dir, exist_ok=True)
    
    # Progress tracking
    total_models = len(selected_models)
    model_progress = st.progress(0)
    current_model_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Train and evaluate each model
    for idx, model_name in enumerate(selected_models):
        current_model_placeholder.info(f"🔄 Training {model_name} ({idx + 1}/{total_models})")
        
        # Create model
        model_fn = model_options[model_name]
        model = model_fn(input_shape, num_classes)
        
        # Setup callbacks
        log_dir = os.path.join(study_dir, 'logs', model_name.replace(' ', '_'))
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
            TensorBoard(log_dir=log_dir),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Measure training time
        start_time = time.time()
        
        try:
            # Train model
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            training_time = time.time() - start_time
            
            # Evaluate model
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            val_accuracy = max(history.history['val_categorical_accuracy'])
            
            # Save model
            model_path = os.path.join(study_dir, f"{model_name.replace(' ', '_')}_model.h5")
            model.save(model_path)
            
            # Store results
            results['model_name'].append(model_name)
            results['accuracy'].append(accuracy)
            results['val_accuracy'].append(val_accuracy)
            results['training_time'].append(training_time)
            results['history'].append(history.history)
            
            # Update metrics display
            metrics_placeholder.success(f"✅ {model_name} - Accuracy: {accuracy:.4f}, Time: {training_time:.1f}s")
            
            # Generate confusion matrix
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Create confusion matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=actions, yticklabels=actions)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(study_dir, f'{model_name.replace(" ", "_")}_confusion_matrix.png'))
            plt.close()
            
        except Exception as e:
            st.error(f"❌ Error training {model_name}: {str(e)}")
            # Add failed model with None values
            results['model_name'].append(model_name)
            results['accuracy'].append(0)
            results['val_accuracy'].append(0)
            results['training_time'].append(0)
            results['history'].append(None)
        
        # Update progress
        model_progress.progress((idx + 1) / total_models)
    
    # Display results
    display_comparative_results(results, study_dir, actions)

def display_comparative_results(results, study_dir, actions):
    """Display the comparative study results."""
    st.write("### 📊 Comparative Study Results")
    
    # Create summary table
    summary_data = {
        'Model': results['model_name'],
        'Test Accuracy': [f"{acc:.4f}" for acc in results['accuracy']],
        'Validation Accuracy': [f"{val_acc:.4f}" if val_acc else "N/A" for val_acc in results['val_accuracy']],
        'Training Time (s)': [f"{time:.1f}" if time else "N/A" for time in results['training_time']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        models = results['model_name']
        test_acc = results['accuracy']
        
        bars = ax.bar(models, test_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        ax.set_xlabel('Models')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Model Test Accuracy Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, test_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        # Training time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        training_times = [t for t in results['training_time'] if t > 0]
        model_names = [results['model_name'][i] for i, t in enumerate(results['training_time']) if t > 0]
        
        if training_times:
            bars = ax.bar(model_names, training_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)])
            ax.set_xlabel('Models')
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title('Model Training Time Comparison')
            
            # Add value labels on bars
            for bar, time_val in zip(bars, training_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(training_times) * 0.01,
                       f'{time_val:.1f}s', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    # Learning curves
    st.write("### 📈 Learning Curves")
    
    valid_histories = [(name, hist) for name, hist in zip(results['model_name'], results['history']) if hist is not None]
    
    if valid_histories:
        n_models = len(valid_histories)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, history) in enumerate(valid_histories):
            ax = axes[idx] if n_models > 1 else axes[0]
            
            # Plot training and validation accuracy
            ax.plot(history['categorical_accuracy'], label='Training Accuracy', marker='o', markersize=3)
            ax.plot(history['val_categorical_accuracy'], label='Validation Accuracy', marker='s', markersize=3)
            ax.set_title(f'Learning Curve - {model_name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Best model recommendation
    if results['accuracy']:
        best_idx = np.argmax(results['accuracy'])
        best_model = results['model_name'][best_idx]
        best_accuracy = results['accuracy'][best_idx]
        
        st.success(f"🏆 **Best Model:** {best_model} with {best_accuracy:.4f} test accuracy")
    
    # Save results
    summary_df.to_csv(os.path.join(study_dir, 'comparative_study_results.csv'), index=False)
    st.info(f"📁 Results saved to: {study_dir}")
    
    st.balloons()
