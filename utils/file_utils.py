
"""
File management utilities.
"""
import os
import shutil
import tempfile
from typing import List, Optional

def create_temp_file(suffix: str = '.tmp') -> str:
    """Create a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        return temp_file.name

def ensure_directory_exists(directory_path: str):
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(directory_path, exist_ok=True)

def safe_remove_file(file_path: str) -> bool:
    """Safely remove a file, return True if successful."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except Exception:
        return False

def safe_remove_directory(directory_path: str) -> bool:
    """Safely remove a directory and its contents."""
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        return True
    except Exception:
        return False

def get_available_models() -> List[str]:
    """Get list of available model files."""
    from config.model_config import MODELS_CONFIG
    available_models = []
    
    for model_key, config in MODELS_CONFIG.items():
        if os.path.exists(config['path']):
            available_models.append(model_key)
    
    return available_models

def validate_file_path(file_path: str) -> bool:
    """Validate if a file path exists and is accessible."""
    return os.path.exists(file_path) and os.path.isfile(file_path)
