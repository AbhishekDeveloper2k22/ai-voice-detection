"""
Create a sample audio file for testing using librosa's example audio
"""
import base64
import numpy as np
import tempfile
import os
import soundfile as sf
import librosa
import json
import requests


def create_sample_human_audio():
    """
    Create a sample audio file that simulates human-like speech characteristics
    """
    # Generate synthetic speech-like audio with natural variations
    sr = 22050
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a complex signal with natural variations (simulating human voice)
    # Base frequency with natural variation
    base_freq = 150 + 20 * np.sin(2 * np.pi * 0.5 * t)  # Varying fundamental
    
    # Add harmonics (characteristic of human voice)
    signal = np.sin(2 * np.pi * base_freq * t)  # Fundamental
    signal += 0.5 * np.sin(2 * np.pi * 2 * base_freq * t)  # 2nd harmonic
    signal += 0.25 * np.sin(2 * np.pi * 3 * base_freq * t)  # 3rd harmonic
    signal += 0.125 * np.sin(2 * np.pi * 4 * base_freq * t)  # 4th harmonic
    
    # Add natural amplitude variation (like breathing)
    envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * np.random.randn(len(t))
    envelope = np.clip(envelope, 0.1, 1.0)
    signal = signal * envelope
    
    # Add some noise (natural microphone/environment noise)
    signal += 0.02 * np.random.randn(len(signal))
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save to WAV first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
        sf.write(wav_path, signal, sr)
    
    # Convert to MP3 using pydub
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(wav_path)
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
        mp3_path = mp3_file.name
        audio.export(mp3_path, format="mp3")
    
    # Read and encode to base64
    with open(mp3_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Cleanup
    os.unlink(wav_path)
    os.unlink(mp3_path)
    
    return audio_base64


def create_sample_ai_audio():
    """
    Create a sample audio file that simulates AI-generated speech characteristics
    """
    sr = 22050
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # AI-like: Very consistent pitch (robotic)
    base_freq = 200  # Constant fundamental frequency
    
    # Perfect harmonics (too perfect)
    signal = np.sin(2 * np.pi * base_freq * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * base_freq * t)
    signal += 0.25 * np.sin(2 * np.pi * 3 * base_freq * t)
    
    # Very consistent amplitude (robotic)
    envelope = 0.8 * np.ones(len(t))
    # Add slight fade in/out
    fade_samples = int(sr * 0.1)
    envelope[:fade_samples] = np.linspace(0, 0.8, fade_samples)
    envelope[-fade_samples:] = np.linspace(0.8, 0, fade_samples)
    signal = signal * envelope
    
    # Normalize (very clean, no noise)
    signal = signal / np.max(np.abs(signal)) * 0.95
    
    # Save to WAV first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
        sf.write(wav_path, signal, sr)
    
    # Convert to MP3 using pydub
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(wav_path)
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
        mp3_path = mp3_file.name
        audio.export(mp3_path, format="mp3")
    
    # Read and encode to base64
    with open(mp3_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Cleanup
    os.unlink(wav_path)
    os.unlink(mp3_path)
    
    return audio_base64


def test_api(audio_base64, language="English", description=""):
    """Test the API with the given audio"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    
    response = requests.post(
        "http://localhost:8000/api/voice-detection",
        headers={
            "Content-Type": "application/json",
            "x-api-key": "sk_test_123456789"
        },
        json={
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        },
        timeout=60
    )
    
    result = response.json()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print("Creating sample audio files...")
    
    try:
        # Create and test human-like audio
        print("\n📢 Creating simulated human voice...")
        human_audio = create_sample_human_audio()
        print(f"   Audio length: {len(human_audio)} base64 characters")
        test_api(human_audio, "English", "Simulated Human Voice")
        
        # Create and test AI-like audio
        print("\n🤖 Creating simulated AI voice...")
        ai_audio = create_sample_ai_audio()
        print(f"   Audio length: {len(ai_audio)} base64 characters")
        test_api(ai_audio, "English", "Simulated AI Voice")
        
        print("\n✅ Tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
