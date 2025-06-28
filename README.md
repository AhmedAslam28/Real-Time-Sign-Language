
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Supported-blue?logo=docker)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?logo=streamlit)



# Indian Sign Language System

A comprehensive sign language recognition and generation system built with Python, TensorFlow, and Streamlit.

## Features

### 🖐️ Sign Recognition
- Real-time sign language detection using webcam
- Multiple model support for different sign categories
- Sentence building from detected signs
- Translation to natural language with text-to-speech

### 💬 Sign Generation
- Convert text to Indian Sign Language videos
- Speech-to-sign conversion with multi-language support
- Combined video generation for complete sentences

### 📊 Data Collection
- Structured data collection workflow
- Support for multiple sign categories
- Real-time feedback during collection

### 🎯 Model Training
- Train custom models on collected data
- Multiple architectures supported (LSTM, CNN, Transformer)
- Real-time training progress monitoring

### 📈 Comparative Study
- Compare different model architectures
- Performance analysis and visualization
- Automated model selection

## Project Structure

```
sign_language_system/
├── main.py                          # Main application entry point
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker Compose configuration
├── .dockerignore                    # Docker ignore file
├── config/
│   ├── __init__.py
│   ├── settings.py                  # Configuration constants
│   └── model_config.py              # Model configurations
├── core/
│   ├── __init__.py
│   ├── detector.py                  # SignLanguageDetector class
│   ├── predictor.py                 # PredictionSmoother class
│   └── mediapipe_utils.py           # MediaPipe processing
├── services/
│   ├── __init__.py
│   ├── translation_service.py      # Translation and TTS
│   ├── speech_service.py            # Speech recognition
│   └── video_service.py             # Video processing
├── ui/
│   ├── __init__.py
│   ├── welcome_ui.py                # Welcome page
│   ├── recognition_ui.py            # Sign recognition interface
│   ├── generation_ui.py             # Sign generation interface
│   ├── data_collection_ui.py        # Data collection interface
│   ├── training_ui.py               # Model training interface
│   └── comparative_study_ui.py      # Comparative study interface
├── utils/
│   ├── __init__.py
│   ├── file_utils.py                # File handling utilities
│   └── session_utils.py             # Session state management
├── models/                          # ML model files (created automatically)
├── assets/
│   └── sign_vd/                     # Sign language video files
├── MP_Data/                         # Training data (created automatically)
├── Logs/                            # Training logs (created automatically)
└── Results/                         # Study results (created automatically)
```

## Installation

### Option 1: Local Installation

1. Clone the repository or download the files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker Installation (Recommended)

Docker provides a consistent environment and easier setup process, especially for managing system dependencies and ensuring reproducibility across different platforms.

#### Prerequisites
- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)

#### Quick Start with Docker
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sign_language_system
   ```

2. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. Access the application at `http://localhost:8501`

#### Alternative Docker Commands
```bash
# Build the Docker image
docker build -t sign-language-app .

# Run the container
docker run -p 8501:8501 -v $(pwd)/models:/app/models --device=/dev/video0:/dev/video0 sign-language-app
```

#### Docker Notes
- The application runs in a containerized environment with all dependencies pre-installed
- Models, training data, and logs are persisted using Docker volumes
- Camera access is configured for Linux/Mac systems
- For Windows users, camera access may require additional Docker configuration

## Usage

1. Start the application:
   
   **Local:**
   ```bash
   streamlit run main.py
   ```
   
   **Docker:**
   ```bash
   docker-compose up
   ```

2. Open your web browser and navigate to the displayed URL (usually `http://localhost:8501`)

3. Follow the welcome page instructions to get started

## Supported Signs

Currently supports the following sign categories:
- **Greetings**: hello, thanks, love
- **Pronouns**: you, me
- **Actions**: angry, bring, care, cry
- **Questions**: why, how, what, where
- **Others**: help, water, doing

## Requirements

### Local Installation
- Python 3.8 or higher
- Webcam for sign recognition and data collection
- Microphone for speech input (optional)
- Minimum 4GB RAM recommended
- GPU support optional but recommended for training

### Docker Installation
- Docker and Docker Compose
- Webcam for sign recognition and data collection
- Minimum 4GB RAM recommended
- GPU support optional but recommended for training

## Configuration

### API Keys
Update the following in `config/settings.py`:
- ElevenLabs API key for text-to-speech (optional)

### Model Paths
Models are automatically saved to the `models/` directory. Update `config/model_config.py` to modify model configurations.

### Video Assets
Place sign language videos in `assets/sign_vd/` directory following the naming convention in `config/model_config.py`.

## Development

### Adding New Signs
1. Update `config/model_config.py` with new sign categories
2. Add corresponding videos to `assets/sign_vd/`
3. Use Data Collection mode to gather training data
4. Train new models using Training mode

### Adding New Model Architectures
1. Create model function in `ui/comparative_study_ui.py`
2. Add to model options in the comparative study interface
3. Test performance using Comparative Study mode

## Troubleshooting

### Camera Issues
- **Local**: Ensure webcam is connected and not used by other applications
- **Docker**: Verify camera device mapping in docker-compose.yml
- Try changing camera index in OpenCV capture initialization

### Model Loading Errors
- Check that model files exist in the `models/` directory
- Verify model architecture matches expected input shape

### Audio Issues
- Install PyAudio: `pip install pyaudio`
- Check microphone permissions in your browser/system

### Docker Issues
- **Camera not working**: Ensure proper device mapping for your OS
- **Permission errors**: Run Docker with appropriate privileges
- **Build failures**: Check system dependencies and Docker version

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the modular structure
4. Test thoroughly (both locally and with Docker)
5. Submit a pull request

## Acknowledgments

- MediaPipe for hand and pose detection
- TensorFlow for deep learning capabilities
- Streamlit for the web interface
- OpenCV for computer vision processing
