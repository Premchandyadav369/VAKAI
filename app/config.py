import os

# API Configuration
API_KEY = os.getenv("API_KEY", "indic-ai-voice-2026")

# Model Configuration
MODEL_WAV2VEC = "facebook/wav2vec2-base"

# Server Configuration
PORT = int(os.getenv("PORT", 8000))
