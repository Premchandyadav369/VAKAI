import os
import torch
import torchaudio
import numpy as np
import base64
import io
import logging
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import time
from groq import Groq

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# PASTE YOUR GROQ API KEY HERE
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# Model to use (must be available in Groq)
GROQ_MODEL = "llama-3.3-70b-versatile" 

# ---------------------

class AudioProcessor:
    @staticmethod
    def decode_audio(base64_audio: str) -> tuple:
        try:
            import librosa
            audio_bytes = base64.b64decode(base64_audio)
            audio_buffer = io.BytesIO(audio_bytes)
            # Resample to 16k for local models
            waveform_np, sample_rate = librosa.load(audio_buffer, sr=16000)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            return waveform, sample_rate, audio_bytes
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise ValueError(f"Invalid audio format: {str(e)}")

class VoiceDetector:
    def __init__(self):
        logger.info("🚀 Loading Pro-Tier Indic Detection Engine (XLSR-53)...")
        # 1. Load Local Acoustic Engine (The "Ears")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.model.eval()
        
        # 2. Load Groq Reasoning Engine (The "Brain")
        logger.info(f"Initializing Groq Client with model: {GROQ_MODEL}")
        try:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            # Simple test to verify connection
            self.groq_client.models.list()
            logger.info("✅ Groq API Connected Successfully")
        except Exception as e:
            logger.error(f"❌ Groq Connection Failed: {e}")
            self.groq_client = None

    def _analyze_with_groq(self, stats: dict) -> dict:
        """
        Sends acoustic statistics to Groq LPU for forensic analysis.
        Since Groq is text-based, we describe the audio math to it.
        """
        if not self.groq_client:
            return None

        # prompt engineering for the "Brain"
        system_prompt = (
            "You are an advanced AI Forensic Audio Analyst specializing in Deepfake detection for Indian languages. "
            "Analyze the provided acoustic metrics and determine if the audio is Real Human or AI Generated. "
            "AI Voices typically have abnormally low variance (too perfect) and high embedding norms. "
            "Human voices have stochastic irregularities (higher variance)."
        )

        user_prompt = f"""
        Acoustic Analysis Data:
        - Neural Embedding Standard Deviation (Consistency): {stats['std']:.5f} (Human usually > 0.18, AI < 0.15)
        - Neural Embedding Norm (Signal Strength): {stats['norm']:.2f} (AI often > 4.0)
        - Zero Crossing Rate: {stats.get('zcr', 'N/A')}
        - Audio Duration: {stats.get('duration', 'N/A')}s
        
        Task:
        1. Classify as 'AI_GENERATED' or 'HUMAN'.
        2. Assign a confidence score (0.0 to 1.0).
        3. Provide a 1-sentence technical explanation.
        4. Guess the likely Indian language context based on metadata (assume Telugu/Hindi if distinct features appear).
        
        Respond ONLY in JSON format:
        {{
            "label": "AI_GENERATED" or "HUMAN",
            "confidence": float,
            "explanation": "string",
            "detected_language": "string"
        }}
        """

        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=GROQ_MODEL,
                temperature=0.1, # Low temperature for factual/consistent output
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq Inference Error: {e}")
            return None

    def predict(self, base64_audio: str) -> dict:
        # A. Decode
        waveform, sr, _ = AudioProcessor.decode_audio(base64_audio)
        
        # B. Extract Acoustic Features (Local XLSR-53)
        # This is the "Truth" signal
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over time to get utterance-level embedding
            emb = outputs.last_hidden_state.mean(dim=1)
            
            # Key Biomarkers for AI Detection
            emb_std = float(torch.std(emb))
            emb_norm = float(torch.norm(emb))
            
            # Simple physical stats
            waveform_np = waveform.numpy().squeeze()
            zcr = float(np.mean(np.abs(np.diff(np.sign(waveform_np)))))
            duration = len(waveform_np) / 16000

        stats = {
            "std": emb_std,
            "norm": emb_norm,
            "zcr": zcr,
            "duration": duration
        }

        # C. Groq Reasoning (The "Judge")
        groq_result = self._analyze_with_groq(stats)
        
        if groq_result:
            final_label = groq_result.get("label", "HUMAN")
            final_confidence = groq_result.get("confidence", 0.0)
            explanation = groq_result.get("explanation", "Groq analysis completed.")
            detected_lang = groq_result.get("detected_language", "Unknown")
            engine = f"XLSR-53 + Groq ({GROQ_MODEL})"
        else:
            # D. Fallback Heuristic (Safe Mode)
            engine = "XLSR-53-Local (Groq Failed)"
            # Stricter thresholds for fallback
            if emb_std < 0.165:
                final_label = "AI_GENERATED"
                final_confidence = 0.85
                explanation = "Acoustic embedding variance is suspiciously low, indicating synthetic generation."
            else:
                final_label = "HUMAN"
                final_confidence = 0.90
                explanation = "Acoustic variability matches natural human speech patterns."
            detected_lang = "Unknown"

        return {
            "label": final_label,
            "confidence": round(float(final_confidence), 4),
            "detected_language": detected_lang,
            "details": {
                "neural_variance": round(emb_std, 4),
                "embedding_norm": round(emb_norm, 2),
                "engine": engine
            },
            "explanation": explanation
        }

_detector = None
def get_voice_detector():
    global _detector
    if _detector is None:
        _detector = VoiceDetector()
    return _detector
