"""
Test script for Voice Detection API
"""
import base64
import json
import requests
from pathlib import Path


API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "sk_test_123456789"


def encode_audio_file(file_path: str) -> str:
    """Encode an audio file to base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_voice_detection(audio_path: str, language: str = "English"):
    """
    Test the voice detection API with a local audio file.
    
    Args:
        audio_path: Path to the MP3 file
        language: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
    """
    print(f"\n{'='*60}")
    print(f"Testing Voice Detection API")
    print(f"Audio: {audio_path}")
    print(f"Language: {language}")
    print(f"{'='*60}\n")
    
    # Encode audio
    try:
        audio_base64 = encode_audio_file(audio_path)
        print(f"✓ Audio encoded: {len(audio_base64)} characters")
    except Exception as e:
        print(f"✗ Failed to encode audio: {e}")
        return
    
    # Prepare request
    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    # Send request
    try:
        print(f"\n→ Sending request to {API_URL}...")
        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=60
        )
        
        # Parse response
        result = response.json()
        
        print(f"\n📊 Response (Status: {response.status_code}):")
        print(json.dumps(result, indent=2))
        
        if result.get("status") == "success":
            print(f"\n🎯 Classification: {result.get('classification')}")
            print(f"📈 Confidence: {result.get('confidenceScore')}")
            print(f"💬 Explanation: {result.get('explanation')}")
        
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed. Make sure the API is running.")
    except Exception as e:
        print(f"✗ Request failed: {e}")


def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        print(f"Health Check: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")


def test_without_api_key():
    """Test request without API key (should fail)"""
    print("\n🔐 Testing without API key...")
    try:
        response = requests.post(
            API_URL,
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "test"
            },
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"Response: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def test_with_invalid_api_key():
    """Test request with invalid API key (should fail)"""
    print("\n🔐 Testing with invalid API key...")
    try:
        response = requests.post(
            API_URL,
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "test"
            },
            headers={
                "Content-Type": "application/json",
                "x-api-key": "invalid_key"
            },
            timeout=5
        )
        print(f"Response: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    print("╔═══════════════════════════════════════════════════════╗")
    print("║         AI Voice Detection API - Test Suite           ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    # Test health
    test_health()
    
    # Test authentication
    test_without_api_key()
    test_with_invalid_api_key()
    
    # Test with actual audio if provided
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else "English"
        test_voice_detection(audio_path, language)
    else:
        print("\n💡 To test with an audio file:")
        print("   python test_api.py <path_to_mp3> <language>")
        print("\n   Example:")
        print("   python test_api.py sample.mp3 Tamil")
