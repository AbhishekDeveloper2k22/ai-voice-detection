"""
API Key Authentication Middleware
"""
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from app.config import get_settings


# API Key header configuration
API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Verify the API key from request headers.
    
    Args:
        api_key: API key from x-api-key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    settings = get_settings()
    
    if api_key is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "API key is missing. Please provide x-api-key header."
            }
        )
    
    if api_key != settings.api_secret_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )
    
    return api_key


async def verify_api_key_simple(request: Request) -> bool:
    """
    Simple API key verification for middleware use.
    
    Args:
        request: FastAPI request object
        
    Returns:
        True if valid, raises exception otherwise
    """
    settings = get_settings()
    api_key = request.headers.get("x-api-key")
    
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "API key is missing"
            }
        )
    
    if api_key != settings.api_secret_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail={
                "status": "error",
                "message": "Invalid API key"
            }
        )
    
    return True
