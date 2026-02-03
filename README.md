# VAK-AI: The Agentic Indic Voice Sentinel 🛡️🇮🇳

> **Hackathon Submission Level 2**
> **Tagline**: *Unmasking the Synthetic Silence with Hybrid Intelligence.*

---

## 🚀 Basic Description
**VAK-AI** is a high-performance, agentic AI voice detector designed to identify deepfake audio. It is specifically optimized for **Indic languages** (Telugu, Hindi, Tamil) where traditional detectors often fail. 

It uses a unique **Hybrid Architecture**:
1.  **Local Acoustic Engine (XLSR-53)**: A massive 1.2GB model that extracts physical "neural variance" features to detect the mathematical "perfectness" of AI speech.
2.  **Reasoning Engine (Groq LPU)**: A **Llama-3.3** agent running on Groq's ultra-fast hardware acting as a forensic analyst to provide instantaneous, human-readable verdicts.

---

## 🏗️ Architecture

The system operates on a dual-engine agentic workflow:

```mermaid
graph TD
    A[User Audio Upload (File/URL)] --> B{Agentic Dispatcher}
    
    subgraph "Local Sensory Cortex"
        B --> C[Decoder (Librosa)]
        C --> D[Wav2Vec2-XLSR-53]
        D --> E[Feature Extraction]
        E --> F[Neural Variance & Embedding Norm]
    end
    
    subgraph "Reasoning Neocortex (Groq LPU)"
        F --> G[Forensic Prompt Construction]
        G --> H[Llama-3.3-70b-Versatile]
        H --> I[Contextual Verdict Generation]
    end
    
    I --> J[Final JSON Response]
```

### 🧠 The Engines
1.  **The "Ears" (Sensation)**: `facebook/wav2vec2-large-xlsr-53`
    *   **Why**: It's the gold standard for cross-lingual speech representation. It perceives the *texture* of the sound.
    *   **What it detects**: Neural Uniformity. AI voices are "too perfect"—mathematically consistent in a way human vocal cords never are. VAK-AI measures this variance.

2.  **The "Brain" (Perception)**: `Groq LPU (Llama-3.3-70b)`
    *   **Why**: Speed and Logic. Groq's LPU provides near-instant inference.
    *   **Agentic Role**: It acts as a forensic pathologist, contextualizing raw acoustic numbers to conclude: *"Low variance confirms synthetic origin despite natural-sounding pitch."*

---

## 🇮🇳 The Indic Advantage

Most AI detectors fail on Indian languages because they misinterpret regional pitch modulations as "robotic."

**VAK-AI calculates language-agnostic neural signatures.**
*   A Telugu speaker's natural modulation has high variance. 
*   A Telugu *Deepfake* has low variance.
By focusing on the *consistency* of the signal rather than the content, VAK-AI works seamlessly across **Telugu, Hindi, Kannada, Tamil, and Malayalam** zero-shot.

---

## 🛠️ API Documentation

### Endpoint
`POST /detect` 

### Authentication
**Header**: `X-API-Key: indic-ai-voice-2026`
*Alternative*: `Authorization: Bearer indic-ai-voice-2026`

### Request Format
Accepts either Base64 string OR a direct URL.

**Option 1 (Base64):**
```json
{
  "audio_base64": "UklGRi..."
}
```

**Option 2 (URL - Great for Testers):**
```json
{
  "audio_url": "https://example.com/sample.mp3"
}
```

### Response Format
```json
{
  "label": "AI_GENERATED",
  "confidence": 0.985,
  "detected_language": "Telugu",
  "explanation": "Acoustic variance (0.124) is significantly below the human biological threshold (>0.18), indicating algorithmic pitch generation.",
  "details": {
    "neural_variance": 0.124,
    "embedding_norm": 5.21,
    "engine": "XLSR-53 + Groq (llama-3.3-70b-versatile)"
  }
}
```

---

## 🌍 Deployment Guide (Railway/Render)

VAK-AI is cloud-native and Docker-ready.

### 1. Prerequisites
*   **Groq API Key**: Essential for the reasoning engine.
*   **512MB+ RAM**: For the XLSR model.

### 2. Environment Variables
Set these in your cloud dashboard:
```bash
API_KEY=indic-ai-voice-2026
GROQ_API_KEY=your_groq_key_here
```

### 3. Deploy
*   **Docker**: The repo includes a production `Dockerfile` with `libsndfile1` pre-installed.
*   **Railway**: Connect GitHub repo -> Auto-detects Dockerfile -> Deploys.
*   **Render**: Create Web Service -> Environment: Docker -> Deploys.

---

## 🏆 Why VAK-AI Wins
1.  **Explainability**: We don't just say "Fake." We tell you *why* (e.g., "lack of breath artifacts").
2.  **Multilingual Supremacy**: Built for the next billion users in India.
3.  **Evaluator Ready**: Supports `audio_url` for automated testing suites.
4.  **Agentic Speed**: Powered by Groq/LPU for real-time protection.

***
*Built with ❤️ for a safer digital India.*
