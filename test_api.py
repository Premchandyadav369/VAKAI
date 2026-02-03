"""
Test script for VAK-AI Voice Detector API.
Run this after starting the server with: python run.py
"""
import requests
import base64
import wave
import struct
import os

# Configuration
API_URL = "http://localhost:8000/detect"
API_KEY = "indic-ai-voice-2026"
USER_MP3_PATH = r"C:\Users\PREMCHANDYADAV\Downloads\Text to Speech Ai Tool in Telugu - Create Telugu Voice Overs With Free AI Website  #shorts.mp3"


def create_test_audio(filename: str = "test_audio.wav", duration_sec: float = 1.0, freq: int = 440):
    """Create a test audio file with a sine wave tone."""
    if os.path.exists(filename):
        print(f"Using existing: {filename}")
        return filename
    
    print(f"Creating test audio: {filename}")
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    
    # Generate sine wave (more realistic than silence)
    import math
    samples = []
    for i in range(num_samples):
        # Add some variation to simulate speech-like audio
        t = i / sample_rate
        value = int(16000 * math.sin(2 * math.pi * freq * t) * (1 + 0.3 * math.sin(2 * math.pi * 3 * t)))
        samples.append(value)
    
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f'<{num_samples}h', *samples))
    
    return filename

def test_health():
    """Test health endpoint."""
    print("\n🔍 Testing Health Endpoint...")
    try:
        response = requests.get("http://localhost:8000/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_auth():
    """Test authentication."""
    print("\n🔐 Testing Authentication...")
    
    # Test without API key
    try:
        response = requests.post(API_URL, json={"audio_base64": "test"})
        print(f"Without key: {response.status_code} (expected 401)")
        
        # Test with wrong key
        response = requests.post(
            API_URL, 
            json={"audio_base64": "test"},
            headers={"X-API-Key": "wrong-key"}
        )
        print(f"Wrong key: {response.status_code} (expected 401)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_detection():
    """Test the main detection endpoint."""
    print("\n🎤 Testing Voice Detection...")
    
    # Use user MP3 if it exists, otherwise create dummy
    if os.path.exists(USER_MP3_PATH):
        audio_file = USER_MP3_PATH
        print(f"Using user provided MP3: {audio_file}")
    else:
        audio_file = create_test_audio()
    
    # Read and encode
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    print(f"Audio size: {len(audio_bytes)} bytes")
    print(f"Base64 size: {len(audio_base64)} chars")
    
    # Send request
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"audio_base64": audio_base64}
    
    try:
        print("Sending request...")
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"Label: {result.get('label')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Details: {result.get('details')}")
            print(f"Explanation: {result.get('explanation', 'N/A')[:100]}...")
            return True
        else:
            print(f"❌ Error Response: {response.json()}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. First request may take longer due to model loading.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("       VAK-AI Voice Detector - API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    health_ok = test_health()
    if not health_ok:
        print("\n⚠️  Server not running! Start with: python run.py")
        return
    
    # Run tests
    test_auth()
    detection_ok = test_detection()
    
    print("\n" + "=" * 60)
    if detection_ok:
        print("🎉 ALL TESTS PASSED! System is ready for submission.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
