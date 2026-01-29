"""
🧪 Quick Test Script for AI Voice Detection API
Run: python quick_test.py
"""
import base64
import json
import requests
import os

# API Configuration
API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "sk_test_123456789"

def test_with_audio_file(file_path: str, language: str = "English"):
    """
    Test API with an actual MP3 file
    
    Usage:
        test_with_audio_file("path/to/your/audio.mp3", "Tamil")
    """
    print(f"\n{'='*60}")
    print(f"🎤 Testing with: {file_path}")
    print(f"🌐 Language: {language}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    # Read and encode audio file
    with open(file_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    print(f"📦 Audio encoded: {len(audio_base64)} characters")
    
    # Make API request
    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY
        },
        json={
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        },
        timeout=120
    )
    
    result = response.json()
    
    print(f"\n📊 Response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get("status") == "success":
        print(f"\n{'='*60}")
        print(f"🎯 Classification: {result.get('classification')}")
        print(f"📈 Confidence: {result.get('confidenceScore', 0) * 100:.1f}%")
        print(f"💬 Explanation: {result.get('explanation')}")
        print(f"{'='*60}")
    
    return result


def test_api_health():
    """Test if API is running"""
    print("\n🏥 Checking API Health...")
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        result = response.json()
        print(f"✅ API Status: {result.get('status')}")
        print(f"📌 Version: {result.get('version')}")
        print(f"🌐 Languages: {', '.join(result.get('supported_languages', []))}")
        return True
    except Exception as e:
        print(f"❌ API not running: {e}")
        return False


def test_api_auth():
    """Test API authentication"""
    print("\n🔐 Testing Authentication...")
    
    # Test without API key
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={"language": "English", "audioFormat": "mp3", "audioBase64": "test"},
        timeout=5
    )
    
    if response.status_code == 401:
        print("✅ Auth check passed - API rejects requests without key")
    else:
        print(f"⚠️ Unexpected response: {response.status_code}")
    
    # Test with wrong API key
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json", "x-api-key": "wrong_key"},
        json={"language": "English", "audioFormat": "mp3", "audioBase64": "test"},
        timeout=5
    )
    
    if response.status_code == 403:
        print("✅ Auth check passed - API rejects invalid key")
    else:
        print(f"⚠️ Unexpected response: {response.status_code}")


def create_sample_audio_and_test():
    """Create a synthetic audio sample and test"""
    print("\n🎵 Creating synthetic audio for testing...")
    
    try:
        import numpy as np
        import tempfile
        import soundfile as sf
        from pydub import AudioSegment
        
        # Create synthetic audio (human-like)
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Generate voice-like signal with variations
        base_freq = 150 + 20 * np.sin(2 * np.pi * 0.5 * t)
        signal = np.sin(2 * np.pi * base_freq * t)
        signal += 0.5 * np.sin(2 * np.pi * 2 * base_freq * t)
        signal += 0.25 * np.sin(2 * np.pi * 3 * base_freq * t)
        
        # Add natural variations
        envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.random.randn(len(t))
        signal = signal * np.clip(envelope, 0.1, 1.0)
        signal += 0.02 * np.random.randn(len(signal))
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        # Save to temp files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
            sf.write(wav_path, signal, sr)
        
        # Convert to MP3
        audio = AudioSegment.from_wav(wav_path)
        mp3_path = wav_path.replace(".wav", ".mp3")
        audio.export(mp3_path, format="mp3")
        
        print(f"✅ Sample audio created: {mp3_path}")
        
        # Test with the created audio
        result = test_with_audio_file(mp3_path, "English")
        
        # Cleanup
        os.unlink(wav_path)
        os.unlink(mp3_path)
        
        return result
        
    except ImportError as e:
        print(f"⚠️ Missing library: {e}")
        print("   Run: pip install numpy soundfile pydub")
        return None


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════╗")
    print("║      🎤 AI Voice Detection API - Quick Test Suite          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Step 1: Health Check
    if not test_api_health():
        print("\n❌ Please start the API first:")
        print("   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        exit(1)
    
    # Step 2: Auth Check
    test_api_auth()
    
    # Step 3: Test with synthetic audio
    create_sample_audio_and_test()
    
    print("\n" + "="*60)
    print("📁 To test with your own audio file:")
    print("   python -c \"from quick_test import test_with_audio_file; test_with_audio_file('your_audio.mp3', 'Tamil')\"")
    print("="*60)
