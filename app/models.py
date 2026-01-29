"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from enum import Enum


class SupportedLanguage(str, Enum):
    """Supported languages for voice detection"""
    TAMIL = "Tamil"
    ENGLISH = "English"
    HINDI = "Hindi"
    MALAYALAM = "Malayalam"
    TELUGU = "Telugu"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    MP3 = "mp3"


class Classification(str, Enum):
    """Voice classification types"""
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"


class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection API"""
    language: SupportedLanguage = Field(
        ...,
        description="Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)"
    )
    audioFormat: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Format of the audio file (always mp3)"
    )
    audioBase64: str = Field(
        ...,
        description="Base64-encoded MP3 audio data",
        min_length=100  # Minimum length for valid audio
    )

    @field_validator('audioBase64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that the audio data is properly base64 encoded"""
        import base64
        try:
            # Try to decode to verify it's valid base64
            decoded = base64.b64decode(v)
            if len(decoded) < 100:
                raise ValueError("Audio data is too short to be valid")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")


class VoiceDetectionSuccessResponse(BaseModel):
    """Success response model for voice detection"""
    status: Literal["success"] = "success"
    language: str = Field(..., description="Language of the analyzed audio")
    classification: Classification = Field(
        ...,
        description="Classification result: AI_GENERATED or HUMAN"
    )
    confidenceScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        ...,
        description="Short reason for the classification decision"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.93,
                "explanation": "Detected AI indicators: unnatural pitch consistency"
            }
        }
    }
    
    def model_dump(self, **kwargs):
        """Override to format confidenceScore with 2 decimal places"""
        data = super().model_dump(**kwargs)
        data['confidenceScore'] = round(data['confidenceScore'], 2)
        return data


class VoiceDetectionErrorResponse(BaseModel):
    """Error response model for voice detection"""
    status: Literal["error"] = "error"
    message: str = Field(..., description="Error message describing the issue")


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    version: str
    supported_languages: list[str]
