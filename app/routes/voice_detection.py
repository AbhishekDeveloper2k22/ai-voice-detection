"""
Voice Detection API Routes
"""
from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
import logging
import time

from app.models import (
    VoiceDetectionRequest,
    VoiceDetectionSuccessResponse,
    VoiceDetectionErrorResponse,
    HealthCheckResponse,
    Classification
)
from app.middleware.auth import verify_api_key
from app.services.audio_processor import AudioProcessor
from app.services.voice_detector import VoiceDetector
from app.config import get_settings
from app import __version__


# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
audio_processor = AudioProcessor()
voice_detector = VoiceDetector()


@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running.
    Does not require authentication.
    """
    settings = get_settings()
    return HealthCheckResponse(
        status="healthy",
        version=__version__,
        supported_languages=settings.supported_languages
    )


@router.post(
    "/voice-detection",
    response_model=VoiceDetectionSuccessResponse,
    responses={
        400: {"model": VoiceDetectionErrorResponse},
        401: {"model": VoiceDetectionErrorResponse},
        403: {"model": VoiceDetectionErrorResponse},
        500: {"model": VoiceDetectionErrorResponse}
    },
    tags=["Voice Detection"],
    summary="Detect AI-Generated Voice",
    description="""
    Analyzes a voice sample to determine if it is AI-generated or spoken by a human.
    
    **Supported Languages:** Tamil, English, Hindi, Malayalam, Telugu
    
    **Input:** Base64-encoded MP3 audio
    
    **Output:** Classification (AI_GENERATED or HUMAN) with confidence score and explanation
    """
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main API endpoint for voice detection.
    
    Accepts a Base64-encoded MP3 audio file and returns whether the voice
    is AI-generated or human.
    """
    start_time = time.time()
    settings = get_settings()
    
    try:
        # Validate language
        language = request.language.value
        if language not in settings.supported_languages:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": f"Unsupported language: {language}. Supported: {settings.supported_languages}"
                }
            )
        
        logger.info(f"Processing voice detection request for language: {language}")
        logger.info(f"DEBUG: audioBase64 length = {len(request.audioBase64)} chars")
        logger.info(f"DEBUG: audioBase64 first 50 chars = {request.audioBase64[:50]}")
        logger.info(f"DEBUG: audioBase64 last 50 chars = {request.audioBase64[-50:]}")
        
        # Process audio and extract features
        try:
            features = audio_processor.process_base64_audio(request.audioBase64)
        except ValueError as e:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": f"Audio processing failed: {str(e)}"
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {str(e)}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "status": "error",
                    "message": "Failed to process audio file"
                }
            )
        
        # Detect voice type
        try:
            result = voice_detector.detect(features, language)
        except Exception as e:
            logger.error(f"Error in voice detection: {str(e)}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "status": "error",
                    "message": "Voice detection analysis failed"
                }
            )
        
        processing_time = time.time() - start_time
        logger.info(
            f"Voice detection complete: {result.classification}, "
            f"confidence: {result.confidence_score}, "
            f"time: {processing_time:.2f}s"
        )
        
        # Return success response
        return VoiceDetectionSuccessResponse(
            status="success",
            language=language,
            classification=Classification(result.classification.value),
            confidenceScore=result.confidence_score,
            explanation=result.explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in voice detection endpoint: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Internal server error"
            }
        )
