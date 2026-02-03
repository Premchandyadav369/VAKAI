from fastapi import FastAPI, HTTPException, Header, Depends, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
import base64

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import logging
import os
import requests

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key from environment or default
API_KEY = os.getenv("API_KEY", "indic-ai-voice-2026")

# FastAPI App
app = FastAPI(
    title="Indic AI Voice Detector",
    description="Detect AI-generated voices with support for Indic languages",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class DetectRequest(BaseModel):
    audio_base64: str | None = None
    audio_url: str | None = None

    @field_validator('audio_base64')
    @classmethod
    def validate_input(cls, v):
        # Logic validation happens in endpoint
        return v

class DetectResponse(BaseModel):
    label: str
    confidence: float
    detected_language: str = "Unknown"
    details: dict = {}
    explanation: str = ""


# --- Security ---
async def verify_api_key(
    x_api_key: str = Header(None), 
    authorization: str = Header(None)
):
    """Verify API key from X-API-Key or Authorization header."""
    api_key_to_check = x_api_key
    
    # Fallback to Authorization: Bearer <key>
    if not api_key_to_check and authorization:
        if authorization.startswith("Bearer "):
            api_key_to_check = authorization.split(" ")[1]
        else:
            api_key_to_check = authorization

    if not api_key_to_check:
        raise HTTPException(status_code=401, detail="Missing API Key. Use X-API-Key or Authorization header.")
    
    if api_key_to_check != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    return api_key_to_check


# --- Error Handlers ---
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid Input", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Server Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Server Error", "detail": "Internal processing failed"}
    )

# --- Endpoints ---

@app.get("/")
async def health_check():
    """Root health check endpoint."""
    return {
        "status": "active",
        "service": "Indic AI Voice Detector",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Alternative health check."""
    return {"status": "healthy"}

@app.post("/detect", response_model=DetectResponse)
async def detect_voice(request: DetectRequest, api_key: str = Depends(verify_api_key)):
    """
    Detect if audio is AI-generated or human.
    Accepts either 'audio_base64' (raw base64 OR url) or 'audio_url'.
    """
    try:
        final_audio_base64 = None
        input_data = request.audio_base64

        # CASE 1: Tester sends URL in audio_base64 field
        if input_data and (input_data.startswith("http://") or input_data.startswith("https://")):
             # It's actually a URL!
             try:
                response = requests.get(input_data, timeout=10)
                response.raise_for_status()
                final_audio_base64 = base64.b64encode(response.content).decode('utf-8')
             except Exception as download_err:
                 raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(download_err)}")
        
        # CASE 2: Real Base64
        elif input_data:
             final_audio_base64 = input_data

        # CASE 3: Explicit audio_url field (for our own testing)
        elif request.audio_url:
             try:
                response = requests.get(request.audio_url, timeout=10)
                response.raise_for_status()
                final_audio_base64 = base64.b64encode(response.content).decode('utf-8')
             except Exception as download_err:
                 raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(download_err)}")

        if not final_audio_base64:
             raise HTTPException(status_code=400, detail="Either 'audio_base64' or 'audio_url' must be provided.")
             
        if len(final_audio_base64) < 100:
             raise HTTPException(status_code=400, detail="Audio data too short or empty")

        # Lazy import to avoid startup delay
        from app.inference import get_voice_detector
        
        # Get detector and run prediction
        detector = get_voice_detector()
        result = detector.predict(final_audio_base64)
        
        return result

        
    except HTTPException as he:
        raise he
    except ValueError as ve:
        logger.warning(f"Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=422, detail="Could not process audio content")

@app.post("/detect-file", response_model=DetectResponse)
async def detect_voice_file(file: UploadFile = File(...), x_api_key: str = Header(None)):
    """
    Detect if audio file is AI-generated or human (Multipart upload).
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    try:
        content = await file.read()
        audio_base64 = base64.b64encode(content).decode('utf-8')
        
        from app.inference import get_voice_detector
        detector = get_voice_detector()
        result = detector.predict(audio_base64)
        
        return result
    except Exception as e:
        logger.error(f"File Prediction Error: {e}")
        raise HTTPException(status_code=422, detail="File processing failed")

# Serve UI
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Indic AI Voice Detector...")
    logger.info(f"📍 API Key: {API_KEY[:10]}...")
