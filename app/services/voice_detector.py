"""
Voice Detection Service
Analyzes audio features to determine if voice is AI-generated or Human
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Classification(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"


@dataclass
class DetectionResult:
    """Result of voice detection analysis"""
    classification: Classification
    confidence_score: float
    explanation: str
    feature_scores: Dict[str, float]


class VoiceDetector:
    """
    Multi-layered voice detection system that analyzes various audio characteristics
    to determine if a voice is AI-generated or human.
    
    Detection Strategy:
    1. Pitch Consistency Analysis - AI voices tend to have unnatural pitch patterns
    2. Spectral Anomaly Detection - Synthetic voices have distinct spectral signatures
    3. Temporal Pattern Analysis - Natural speech has micro-variations
    4. Harmonic Structure Analysis - Human voice has complex harmonics
    5. Statistical Anomaly Detection - AI-generated audio has different statistics
    """
    
    # Thresholds tuned for AI vs Human detection
    # These are based on empirical observations of AI-generated voices
    THRESHOLDS = {
        # Pitch analysis thresholds
        'pitch_consistency_high': 15.0,  # AI tends to be more consistent
        'pitch_variation_low': 0.05,  # AI has lower natural variation
        'pitch_range_narrow': 50.0,  # Hz, AI often has narrow pitch range
        
        # Spectral thresholds
        'spectral_flatness_high': 0.15,  # AI can have unusual flatness
        'spectral_variance_low': 100.0,  # AI has lower spectral variance
        
        # Energy thresholds
        'rms_variation_low': 0.1,  # AI has more consistent energy
        
        # Statistical thresholds
        'kurtosis_unusual_low': -1.0,
        'kurtosis_unusual_high': 5.0,
        
        # Duration and temporal
        'onset_regularity_high': 0.8,  # AI has more regular onsets
    }
    
    # Language-specific adjustments (multipliers for thresholds)
    LANGUAGE_ADJUSTMENTS = {
        'Tamil': {'pitch': 1.1, 'spectral': 1.0, 'temporal': 1.0},
        'English': {'pitch': 1.0, 'spectral': 1.0, 'temporal': 1.0},
        'Hindi': {'pitch': 1.05, 'spectral': 1.0, 'temporal': 1.0},
        'Malayalam': {'pitch': 1.1, 'spectral': 1.0, 'temporal': 1.0},
        'Telugu': {'pitch': 1.08, 'spectral': 1.0, 'temporal': 1.0},
    }
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def _analyze_pitch_patterns(self, features: Dict) -> Tuple[float, str]:
        """
        Analyze pitch patterns for AI detection indicators.
        AI-generated voices often have:
        - Unnaturally consistent pitch
        - Less micro-variation
        - More predictable F0 contours
        
        Returns: (ai_score, reason)
        """
        ai_indicators = 0
        total_checks = 4
        reasons = []
        
        # Check 1: Pitch consistency (high consistency = more likely AI)
        pitch_consistency = features.get('pitch_consistency', 0)
        if pitch_consistency > self.THRESHOLDS['pitch_consistency_high']:
            ai_indicators += 1.5
            reasons.append("unnatural pitch consistency")
        
        # Check 2: Low pitch variation (AI tends to have less natural variation)
        pitch_variation = features.get('pitch_variation', 0.5)
        if pitch_variation < self.THRESHOLDS['pitch_variation_low']:
            ai_indicators += 1
            reasons.append("limited pitch variation")
        
        # Check 3: Narrow pitch range
        pitch_range = features.get('pitch_range', 100)
        if pitch_range < self.THRESHOLDS['pitch_range_narrow']:
            ai_indicators += 0.5
            reasons.append("narrow pitch range")
        
        # Check 4: MFCC delta patterns
        mfcc_delta_std = features.get('mfcc_delta_std', [0.5] * 20)
        avg_mfcc_delta = np.mean(mfcc_delta_std)
        if avg_mfcc_delta < 0.3:  # Low delta suggests synthetic
            ai_indicators += 1
            reasons.append("robotic speech dynamics")
        
        score = min(ai_indicators / total_checks, 1.0)
        reason = ", ".join(reasons) if reasons else "natural pitch patterns"
        
        return score, reason
    
    def _analyze_spectral_features(self, features: Dict) -> Tuple[float, str]:
        """
        Analyze spectral characteristics for AI detection.
        AI voices often show:
        - Unusual spectral flatness
        - Less natural spectral variance
        - Synthetic harmonic structures
        
        Returns: (ai_score, reason)
        """
        ai_indicators = 0
        total_checks = 4
        reasons = []
        
        # Check 1: Spectral flatness
        spectral_flatness = features.get('spectral_flatness_mean', 0.05)
        if spectral_flatness > self.THRESHOLDS['spectral_flatness_high']:
            ai_indicators += 1
            reasons.append("synthetic spectral pattern")
        
        # Check 2: Low spectral centroid variance (AI is more consistent)
        spectral_var = features.get('spectral_centroid_var', 1000)
        if spectral_var < self.THRESHOLDS['spectral_variance_low']:
            ai_indicators += 1
            reasons.append("unnatural spectral consistency")
        
        # Check 3: Harmonic ratio analysis
        harmonic_ratio = features.get('harmonic_ratio', 0.5)
        if harmonic_ratio > 0.95 or harmonic_ratio < 0.3:
            ai_indicators += 1
            reasons.append("unusual harmonic structure")
        
        # Check 4: Spectral contrast analysis
        spectral_contrast = features.get('spectral_contrast_mean', [0] * 7)
        contrast_std = np.std(spectral_contrast)
        if contrast_std < 2.0:  # Very uniform contrast
            ai_indicators += 0.5
            reasons.append("flat spectral contrast")
        
        score = min(ai_indicators / total_checks, 1.0)
        reason = ", ".join(reasons) if reasons else "natural spectral characteristics"
        
        return score, reason
    
    def _analyze_temporal_patterns(self, features: Dict) -> Tuple[float, str]:
        """
        Analyze temporal patterns for AI detection.
        Human speech has natural:
        - Micro-hesitations
        - Breathing patterns
        - Variable rhythm
        
        Returns: (ai_score, reason)
        """
        ai_indicators = 0
        total_checks = 3
        reasons = []
        
        # Check 1: RMS variation (energy variation)
        rms_variation = features.get('rms_variation', 0.3)
        if rms_variation < self.THRESHOLDS['rms_variation_low']:
            ai_indicators += 1
            reasons.append("robotic energy consistency")
        
        # Check 2: Onset strength variation
        onset_std = features.get('onset_strength_std', 0.2)
        onset_mean = features.get('onset_strength_mean', 0.5)
        if onset_mean > 0 and onset_std / (onset_mean + 1e-6) < 0.3:
            ai_indicators += 1
            reasons.append("mechanical rhythm patterns")
        
        # Check 3: Zero crossing rate consistency
        zcr_std = features.get('zcr_std', 0.05)
        zcr_mean = features.get('zcr_mean', 0.1)
        if zcr_mean > 0 and zcr_std / (zcr_mean + 1e-6) < 0.2:
            ai_indicators += 1
            reasons.append("unnatural speech transitions")
        
        score = min(ai_indicators / total_checks, 1.0)
        reason = ", ".join(reasons) if reasons else "natural temporal patterns"
        
        return score, reason
    
    def _analyze_statistical_anomalies(self, features: Dict) -> Tuple[float, str]:
        """
        Analyze statistical properties for AI detection.
        Synthetic audio often has different statistical distributions.
        
        Returns: (ai_score, reason)
        """
        ai_indicators = 0
        total_checks = 3
        reasons = []
        
        # Check 1: Kurtosis (distribution shape)
        kurtosis = features.get('audio_kurtosis', 0)
        if kurtosis < self.THRESHOLDS['kurtosis_unusual_low'] or \
           kurtosis > self.THRESHOLDS['kurtosis_unusual_high']:
            ai_indicators += 1
            reasons.append("unusual amplitude distribution")
        
        # Check 2: Skewness analysis
        skewness = features.get('audio_skewness', 0)
        if abs(skewness) > 1.5:
            ai_indicators += 0.5
            reasons.append("asymmetric waveform")
        
        # Check 3: Formant analysis (if available)
        formant_freqs = features.get('formant_freqs', [])
        if len(formant_freqs) >= 3:
            # Check for unusual formant spacing
            formant_diffs = np.diff(sorted(formant_freqs))
            if len(formant_diffs) > 0 and np.std(formant_diffs) < 100:
                ai_indicators += 1
                reasons.append("synthetic formant patterns")
        
        score = min(ai_indicators / total_checks, 1.0)
        reason = ", ".join(reasons) if reasons else "natural statistical properties"
        
        return score, reason
    
    def _generate_comprehensive_explanation(
        self,
        classification: Classification,
        feature_scores: Dict[str, float],
        reasons: Dict[str, str]
    ) -> str:
        """
        Generate a human-readable explanation for the classification decision.
        """
        if classification == Classification.AI_GENERATED:
            # Find the strongest AI indicators
            top_indicators = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            explanations = []
            for indicator, _ in top_indicators:
                if indicator in reasons and reasons[indicator] != "natural pitch patterns":
                    explanations.append(reasons[indicator])
            
            if explanations:
                return f"Detected AI indicators: {', '.join(explanations)}"
            else:
                return "Synthetic voice patterns detected across multiple audio features"
        else:
            # Human classification
            natural_aspects = []
            for indicator, score in feature_scores.items():
                if score < 0.3:
                    if indicator == 'pitch':
                        natural_aspects.append("natural pitch dynamics")
                    elif indicator == 'spectral':
                        natural_aspects.append("organic spectral characteristics")
                    elif indicator == 'temporal':
                        natural_aspects.append("human-like rhythm")
                    elif indicator == 'statistical':
                        natural_aspects.append("natural voice properties")
            
            if natural_aspects:
                return f"Voice exhibits {', '.join(natural_aspects[:2])}"
            else:
                return "Voice shows characteristics consistent with human speech"
    
    def detect(self, features: Dict, language: str = "English") -> DetectionResult:
        """
        Main detection method that combines all analysis modules.
        
        Args:
            features: Extracted audio features from AudioProcessor
            language: Language of the audio for language-specific adjustments
            
        Returns:
            DetectionResult with classification, confidence, and explanation
        """
        # Get language adjustments
        lang_adj = self.LANGUAGE_ADJUSTMENTS.get(language, {'pitch': 1.0, 'spectral': 1.0, 'temporal': 1.0})
        
        # Run all analysis modules
        pitch_score, pitch_reason = self._analyze_pitch_patterns(features)
        spectral_score, spectral_reason = self._analyze_spectral_features(features)
        temporal_score, temporal_reason = self._analyze_temporal_patterns(features)
        statistical_score, statistical_reason = self._analyze_statistical_anomalies(features)
        
        # Apply language adjustments
        pitch_score *= lang_adj['pitch']
        spectral_score *= lang_adj['spectral']
        temporal_score *= lang_adj['temporal']
        
        # Weighted combination of scores
        # Pitch and spectral are most reliable for AI detection
        weights = {
            'pitch': 0.30,
            'spectral': 0.30,
            'temporal': 0.25,
            'statistical': 0.15
        }
        
        feature_scores = {
            'pitch': min(pitch_score, 1.0),
            'spectral': min(spectral_score, 1.0),
            'temporal': temporal_score,
            'statistical': statistical_score
        }
        
        reasons = {
            'pitch': pitch_reason,
            'spectral': spectral_reason,
            'temporal': temporal_reason,
            'statistical': statistical_reason
        }
        
        # Calculate weighted score
        ai_probability = sum(
            feature_scores[key] * weights[key]
            for key in weights
        )
        
        # Normalize to ensure it's between 0 and 1
        ai_probability = min(max(ai_probability, 0.0), 1.0)
        
        # Determine classification
        if ai_probability >= self.confidence_threshold:
            classification = Classification.AI_GENERATED
            confidence = ai_probability
        else:
            classification = Classification.HUMAN
            confidence = 1.0 - ai_probability
        
        # Generate explanation
        explanation = self._generate_comprehensive_explanation(
            classification, feature_scores, reasons
        )
        
        return DetectionResult(
            classification=classification,
            confidence_score=float(f"{confidence:.2f}"),
            explanation=explanation,
            feature_scores=feature_scores
        )
