
"""
Sign language detection and prediction processing.
"""
import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from collections import Counter
from typing import Dict, List, Optional, Tuple

from core.predictor import PredictionSmoother
from config.model_config import MODELS_CONFIG
from config.settings import CONFIDENCE_THRESHOLD, TEMPORAL_THRESHOLD

class SignLanguageDetector:
    """Main sign language detection class."""
    
    def __init__(self, selected_models: Optional[List[str]] = None):
        # Initialize all attributes first
        self.models = {}
        self.actions = {}
        self.smoother = PredictionSmoother()
        self.temporal_threshold = TEMPORAL_THRESHOLD
        self.temporal_counters = {}
        self.feedback_data = {'sequences': [], 'corrections': []}
        self.frame_predictions = []
        
        # Load models after initializing all attributes
        self.load_models(selected_models)

    def load_models(self, selected_models: Optional[List[str]] = None):
        """Load specified models."""
        for model_key, config in MODELS_CONFIG.items():
            # Skip if this model wasn't selected
            if selected_models is not None and model_key not in selected_models:
                continue
                
            if os.path.exists(config['path']):
                try:
                    self.models[model_key] = load_model(config['path'])
                    self.actions[model_key] = np.array(config['actions'])
                    # Initialize temporal counters for this model
                    self.temporal_counters[model_key] = {action: 0 for action in config['actions']}
                except Exception as e:
                    st.error(f"Error loading model {model_key}: {e}")
            else:
                st.warning(f"Model file not found: {config['path']}")

    def predict(self, sequence: np.ndarray) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Make predictions on a sequence."""
        predictions = {}
        confidences = {}
        
        # Store this frame's predictions for majority calculation
        frame_pred = {}
        
        for model_key, model in self.models.items():
            try:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                raw_prediction = self.actions[model_key][predicted_idx]
                
                # Add to frame predictions
                frame_pred[model_key] = raw_prediction
                
                # Apply smoothing
                smoothed_prediction, smoothed_confidence = self.smoother.update(
                    model_key, raw_prediction, confidence
                )
                
                # Apply temporal consistency
                if smoothed_confidence > CONFIDENCE_THRESHOLD:
                    # Increment counter for predicted action, reset others
                    for action in self.temporal_counters[model_key]:
                        if action == smoothed_prediction:
                            self.temporal_counters[model_key][action] += 1
                        else:
                            self.temporal_counters[model_key][action] = 0
                    
                    # Check if prediction is temporally consistent
                    if self.temporal_counters[model_key][smoothed_prediction] >= self.temporal_threshold:
                        predictions[model_key] = smoothed_prediction
                        confidences[model_key] = smoothed_confidence
            except Exception as e:
                st.error(f"Error in prediction for model {model_key}: {e}")
        
        # Add frame predictions to our history
        self.frame_predictions.append(frame_pred)
                        
        return predictions, confidences
    
    def get_majority_words(self) -> Dict[str, str]:
        """Calculate the most frequent predictions during the detection period."""
        majority_words = {}
        
        if not self.frame_predictions:
            return majority_words
            
        # For each model, find the most common word
        for model_key in self.models.keys():
            # Get all predictions for this model
            model_preds = [frame.get(model_key) for frame in self.frame_predictions if model_key in frame]
            
            if model_preds:
                # Count occurrences of each prediction
                counter = Counter(model_preds)
                # Get the most common prediction
                most_common = counter.most_common(1)[0]
                word, count = most_common
                
                # Only include if it appeared in a significant portion of frames (>20%)
                if count / len(model_preds) > 0.2:
                    majority_words[model_key] = word
        
        return majority_words
    
    def clear_frame_predictions(self):
        """Clear the frame prediction history."""
        self.frame_predictions = []
    
    def add_feedback(self, sequence: List, correction: str):
        """Store user feedback for active learning."""
        self.feedback_data['sequences'].append(sequence)
        self.feedback_data['corrections'].append(correction)
        
        # If we have enough feedback data, we could retrain the model
        if len(self.feedback_data['sequences']) >= 10:
            self.retrain_model()
    
    def retrain_model(self):
        """Retrain model with feedback data."""
        if self.feedback_data['sequences'] and self.feedback_data['corrections']:
            X = np.array(self.feedback_data['sequences'])
            y = np.array(self.feedback_data['corrections'])
            
            # For each model, update if it has relevant corrections
            for model_key, model in self.models.items():
                relevant_corrections = [i for i, c in enumerate(y) 
                                     if c in self.actions[model_key]]
                if relevant_corrections:
                    try:
                        # Fine-tune model with new data
                        model.fit(
                            X[relevant_corrections],
                            np.array([np.where(self.actions[model_key] == c)[0][0] 
                                    for c in y[relevant_corrections]]),
                            epochs=5,
                            batch_size=16,
                            verbose=0
                        )
                    except Exception as e:
                        st.error(f"Error retraining model {model_key}: {e}")
            
            # Clear feedback data after retraining
            self.feedback_data = {'sequences': [], 'corrections': []}
