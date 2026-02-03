# VAK-AI: The Agentic Indic Voice Sentinel 🛡️🇮🇳

> **Team**: RED-DRAGON  
> **Tagline**: *Unmasking the Synthetic Silence with Hybrid Intelligence.*

---

## 👥 Team Members (RED-DRAGON)

| Name | Role | Email |
| :--- | :--- | :--- |
| **V C Premchand Yadav** | **Admin / Lead Architect** | [Hidden for Privacy] |
| **EDUPULAPATI SAI PRANEETH** | **Member / Backend Developer** | [Hidden for Privacy] |
| **P R Kiran Kumar Reddy** | **Member / Model Engineer** | [Hidden for Privacy] |
| **Mohith Reddy** | **Member / Frontend Developer** | [Hidden for Privacy] |


---

## 🚀 Project Overview

**VAK-AI** is a high-performance **Agentic AI Voice Detection System** explicitly engineered to combat the rising threat of **Deepfake Audio** in the Indic digital ecosystem. 

Traditional detectors are biased towards English datasets and often fail to distinguish the complex tonal variations of Indian languages (Telugu, Hindi, Tamil) from robotic artifacts. **VAK-AI solves this problem.**

We utilize a **Hybrid Agentic Architecture** that combines:
1.  **Local Sensory Intelligence (The "Ears")**: A massive **1.2GB XLSR-53** multilingual model that dissects raw audio waveforms to detect **"Neural Uniformity"**—the mathematical consistency found in AI speech but absent in biological human voices.
2.  **Groq LPU™ Reasoning (The "Brain")**: A highly specialized **Llama-3.3-70b Agent** running on Groq's Language Processing Unit. It acts as a **Forensic Pathologist**, analyzing the acoustic features in millisecond real-time to generate a human-readable verdict explaining *why* a clip is fake.

---

## 🏗️ Architecture

The system operates on a dual-engine agentic workflow:

```mermaid
graph TD
    A[User Audio Upload (File/URL)] --> B{Agentic Dispatcher}
    
    subgraph "Local Sensory Cortex"
        B --> C[Decoder (Librosa)]
        C --> D[Wav2Vec2-XLSR-53]
        D --> E[Acoustic Feature Extraction]
        E --> F[Embedding Norm & Neural Variance]
    end
    
    subgraph "Reasoning Neocortex (Groq LPU)"
        F --> G[Forensic Prompt Constructor]
        G --> H[LLM Inference Engine (Llama-3.3)]
        H --> I[Anomaly Scoring & Explanation]
    end
    
    I --> J[Final Report + Confidence Score]
```

### 🧠 The Engines Explained
1.  **The "Ears" (Sensation)**: `facebook/wav2vec2-large-xlsr-53`
    *   **Why**: It is the gold standard for cross-lingual speech representation, trained on 53 languages. It doesn't analyze words; it analyzes the *physics* of the sound wave.
    *   **The Theory**: **Neural Uniformity**. AI models generate audio that is statistically "too perfect." Human voices have biological irregularities (stochasticity). VAK-AI measures this variance.

2.  **The "Brain" (Perception)**: `Groq LPU (Llama-3.3-70b)`
    *   **Why**: Speed and Logic. Standard LLMs are too slow for real-time security. Groq's LPU provides near-instant inference.
    *   **Agentic Role**: It takes the raw numbers (e.g., Variance: 0.12) and contextualizes them.
        *   *Scenario*: "This Telugu clip has natural pitch but suspicious uniformity in the high frequencies. Verdict: **AI Generated**."

---

## 🇮🇳 The Indic Advantage

Most AI detectors fail on Indian languages because they misinterpret regional pitch modulations as "robotic artifacts."

**VAK-AI calculates language-agnostic neural signatures.**
*   A Telugu speaker's natural modulation has **high variance** (biological).
*   A Telugu *Deepfake* has **low variance** (algorithmic).
By focusing on the *consistency* of the signal rather than the linguistic content, VAK-AI works seamlessly across **Telugu, Hindi, Kannada, Tamil, and Malayalam** in zero-shot scenarios.

---

## 🛠️ API Documentation

### Endpoint
`POST /detect` 

### Authentication
**Header**: `X-API-Key: indic-ai-voice-2026`
*Alternative*: `Authorization: Bearer indic-ai-voice-2026`

### Request Format
Accepts either Base64 string OR a direct URL (Evaluator Friendly).

**Option 1 (Base64 - Standard):**
```json
{
  "audio_base64": "UklGRi..."
}
```

**Option 2 (URL - Tester Friendly):**
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
*   **Docker**: The repo includes a production `Dockerfile` with `libsndfile1` pre-installed for universal audio decoding.
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
