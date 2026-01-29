"""
Audio Processing Service
Handles audio decoding, preprocessing, and feature extraction
"""
import base64
import io
import tempfile
import os
from typing import Dict, Tuple, Optional
import numpy as np
from pydub import AudioSegment
import librosa
import scipy.stats as stats


class AudioProcessor:
    """
    Processes audio files and extracts features for voice detection.
    Supports MP3 format with base64 encoding.
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def decode_base64_audio(self, base64_data: str) -> bytes:
        """
        Decode base64-encoded audio data to raw bytes.
        
        Args:
            base64_data: Base64-encoded audio string
            
        Returns:
            Raw audio bytes
        """
        try:
            return base64.b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {str(e)}")
    
    def _detect_audio_format(self, audio_bytes: bytes) -> str:
        """
        Detect audio format from file signature (magic bytes).
        
        Returns:
            Format string: 'mp3', 'ogg', 'wav', 'flac', or 'unknown'
        """
        if len(audio_bytes) < 12:
            return 'unknown'
        
        # Check file signatures
        if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb' or audio_bytes[:2] == b'\xff\xfa':
            return 'mp3'
        elif audio_bytes[:4] == b'OggS':
            return 'ogg'  # Could be Opus, Vorbis, etc.
        elif audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
            return 'wav'
        elif audio_bytes[:4] == b'fLaC':
            return 'flac'
        elif audio_bytes[:4] == b'FORM':
            return 'aiff'
        else:
            return 'unknown'
    
    def load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes and convert to numpy array.
        Automatically detects format (MP3, OGG/Opus, WAV, FLAC, etc.)
        
        Args:
            audio_bytes: Raw audio bytes (any supported format)
            
        Returns:
            Tuple of (audio signal as numpy array, sample rate)
        """
        try:
            # Detect audio format
            audio_format = self._detect_audio_format(audio_bytes)
            
            # Choose appropriate file extension
            if audio_format == 'ogg':
                suffix = '.ogg'
            elif audio_format == 'wav':
                suffix = '.wav'
            elif audio_format == 'flac':
                suffix = '.flac'
            elif audio_format == 'aiff':
                suffix = '.aiff'
            else:
                suffix = '.mp3'  # Default to mp3
            
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Load with pydub - it auto-detects format
                if audio_format == 'ogg':
                    audio_segment = AudioSegment.from_ogg(tmp_path)
                elif audio_format == 'wav':
                    audio_segment = AudioSegment.from_wav(tmp_path)
                elif audio_format == 'flac':
                    audio_segment = AudioSegment.from_file(tmp_path, format='flac')
                else:
                    # Try to load as MP3, fallback to auto-detect
                    try:
                        audio_segment = AudioSegment.from_mp3(tmp_path)
                    except Exception:
                        # Fallback: let pydub auto-detect
                        audio_segment = AudioSegment.from_file(tmp_path)
                
                # Convert to mono and set sample rate
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                
                # Export to WAV for librosa
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                    wav_path = wav_file.name
                    audio_segment.export(wav_path, format="wav")
                
                # Load with librosa
                y, sr = librosa.load(wav_path, sr=self.sample_rate)
                
                # Clean up temp files
                os.unlink(wav_path)
            finally:
                os.unlink(tmp_path)
            
            return y, sr
            
        except Exception as e:
            raise ValueError(f"Failed to load audio: {str(e)}")
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, any]:
        """
        Extract comprehensive audio features for voice detection.
        
        Features extracted:
        - MFCC (Mel-frequency cepstral coefficients)
        - Spectral features (centroid, bandwidth, rolloff, contrast)
        - Pitch/F0 analysis
        - Energy and zero-crossing rate
        - Temporal features
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Ensure audio is not empty
        if len(audio) < sr:  # Less than 1 second
            raise ValueError("Audio is too short for analysis")
        
        # ============================================
        # 1. MFCC Features (crucial for voice analysis)
        # ============================================
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
        features['mfcc_delta_mean'] = np.mean(librosa.feature.delta(mfccs), axis=1).tolist()
        features['mfcc_delta_std'] = np.std(librosa.feature.delta(mfccs), axis=1).tolist()
        
        # ============================================
        # 2. Spectral Features
        # ============================================
        # Spectral Centroid - indicates brightness of sound
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        features['spectral_centroid_var'] = float(np.var(spectral_centroid))
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1).tolist()
        
        # Spectral Flatness - helps detect synthetic sounds
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = float(np.std(spectral_flatness))
        
        # ============================================
        # 3. Pitch/F0 Analysis (very important for AI detection)
        # ============================================
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            pitch_array = np.array(pitch_values)
            features['pitch_mean'] = float(np.mean(pitch_array))
            features['pitch_std'] = float(np.std(pitch_array))
            features['pitch_range'] = float(np.max(pitch_array) - np.min(pitch_array))
            features['pitch_variation'] = float(np.std(pitch_array) / (np.mean(pitch_array) + 1e-6))
            
            # Pitch consistency (AI voices tend to be more consistent)
            features['pitch_consistency'] = float(1.0 / (features['pitch_variation'] + 1e-6))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
            features['pitch_variation'] = 0.0
            features['pitch_consistency'] = 0.0
        
        # ============================================
        # 4. Energy Features
        # ============================================
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_variation'] = float(np.std(rms) / (np.mean(rms) + 1e-6))
        
        # ============================================
        # 5. Zero Crossing Rate
        # ============================================
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # ============================================
        # 6. Temporal Features
        # ============================================
        # Tempo and beat
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0]) if len(tempo) > 0 else 0.0
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_strength_mean'] = float(np.mean(onset_env))
        features['onset_strength_std'] = float(np.std(onset_env))
        
        # ============================================
        # 7. Statistical Features
        # ============================================
        features['audio_kurtosis'] = float(stats.kurtosis(audio))
        features['audio_skewness'] = float(stats.skew(audio))
        
        # ============================================
        # 8. Harmonic vs Noise Analysis
        # ============================================
        harmonic, percussive = librosa.effects.hpss(audio)
        features['harmonic_ratio'] = float(np.sum(np.abs(harmonic)) / (np.sum(np.abs(audio)) + 1e-6))
        features['percussive_ratio'] = float(np.sum(np.abs(percussive)) / (np.sum(np.abs(audio)) + 1e-6))
        
        # ============================================
        # 9. Formant-like Features (approximation)
        # ============================================
        # Using spectral peaks as formant approximation
        spectrum = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Find peaks in average spectrum
        avg_spectrum = np.mean(spectrum, axis=1)
        peak_indices = np.argsort(avg_spectrum)[-5:]  # Top 5 peaks
        features['formant_freqs'] = freqs[peak_indices].tolist()
        
        # Duration
        features['duration'] = float(len(audio) / sr)
        
        return features
    
    def process_base64_audio(self, base64_data: str) -> Dict[str, any]:
        """
        Complete pipeline: decode base64, load audio, extract features.
        
        Args:
            base64_data: Base64-encoded MP3 audio
            
        Returns:
            Dictionary of extracted audio features
        """
        # Decode base64
        audio_bytes = self.decode_base64_audio(base64_data)
        
        # Load audio
        audio, sr = self.load_audio(audio_bytes)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        return features
