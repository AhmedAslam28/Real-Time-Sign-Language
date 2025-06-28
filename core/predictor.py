
"""
Prediction smoothing and temporal consistency for sign language detection.
"""
import numpy as np
from collections import Counter, deque
from typing import Dict, Tuple

class PredictionSmoother:
    """Smooth predictions using sliding window and EMA."""
    
    def __init__(self, window_size: int = 5, ema_alpha: float = 0.3):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.prediction_windows: Dict[str, deque] = {}
        self.confidence_ema: Dict[str, float] = {}
        
    def update(self, model_key: str, prediction: str, confidence: float) -> Tuple[str, float]:
        """Update prediction with smoothing."""
        # Initialize windows if needed
        if model_key not in self.prediction_windows:
            self.prediction_windows[model_key] = deque(maxlen=self.window_size)
            self.confidence_ema[model_key] = confidence
        
        # Update windows
        self.prediction_windows[model_key].append(prediction)
        
        # Update EMA for confidence
        self.confidence_ema[model_key] = (self.ema_alpha * confidence + 
                                        (1 - self.ema_alpha) * self.confidence_ema[model_key])
        
        # Majority voting
        if len(self.prediction_windows[model_key]) == self.window_size:
            counter = Counter(self.prediction_windows[model_key])
            smoothed_prediction = counter.most_common(1)[0][0]
            return smoothed_prediction, self.confidence_ema[model_key]
        
        return prediction, confidence
    
    def reset(self):
        """Reset all prediction windows."""
        self.prediction_windows.clear()
        self.confidence_ema.clear()
