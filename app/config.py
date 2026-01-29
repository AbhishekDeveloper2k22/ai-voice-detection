"""
Configuration module for the Voice Detection API
"""
import os
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    api_secret_key: str = "sk_test_123456789"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Supported Languages
    supported_languages: List[str] = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    # Model Configuration
    model_confidence_threshold: float = 0.5
    
    # Debug Mode
    debug: bool = False
    
    # Audio Configuration
    sample_rate: int = 22050
    max_audio_duration: int = 300  # 5 minutes max
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
