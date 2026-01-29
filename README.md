# 🎤 AI Voice Detection API

A secure REST API that detects whether a voice sample is **AI-generated** or **Human** across five Indian languages.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Multi-language Support**: Tamil, English, Hindi, Malayalam, Telugu
- **Base64 MP3 Input**: Easy integration with any frontend
- **Real-time Analysis**: Fast voice classification
- **Confidence Scoring**: 0.0 to 1.0 confidence scale
- **Detailed Explanations**: Human-readable reasoning for decisions
- **API Key Authentication**: Secure API access
- **Docker Ready**: Easy deployment with Docker

## 🔬 Detection Methodology

The API uses a multi-layered analysis approach:

1. **Pitch Pattern Analysis** - AI voices often have unnatural pitch consistency
2. **Spectral Feature Analysis** - Synthetic voices have distinct spectral signatures
3. **Temporal Pattern Analysis** - Natural speech has micro-variations and breathing patterns
4. **Statistical Anomaly Detection** - AI-generated audio has different statistical distributions
5. **Harmonic Structure Analysis** - Human voice has complex harmonics

## 📋 Requirements

- Python 3.11+
- FFmpeg (for audio processing)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
cd /path/to/project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install FFmpeg (Required for audio processing)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# API_SECRET_KEY=your_secret_key_here
```

### 4. Run the API

```bash
# Development mode
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the main module
python -m app.main
```

### 5. Access the API

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t voice-detection-api .
docker run -p 8000:8000 -e API_SECRET_KEY=your_key voice-detection-api
```

## 📡 API Usage

### Endpoint

```
POST /api/voice-detection
```

### Request Headers

| Header | Value | Required |
|--------|-------|----------|
| Content-Type | application/json | Yes |
| x-api-key | Your API key | Yes |

### Request Body

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| language | string | Tamil, English, Hindi, Malayalam, or Telugu |
| audioFormat | string | Always "mp3" |
| audioBase64 | string | Base64-encoded MP3 audio |

### cURL Example

```bash
curl -X POST https://your-domain.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_AUDIO_HERE"
  }'
```

### Success Response

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Detected AI indicators: unnatural pitch consistency, synthetic spectral pattern"
}
```

### Error Response

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## 🔑 Authentication

All requests require an API key in the `x-api-key` header.

```bash
# Request without API key
curl -X POST http://localhost:8000/api/voice-detection
# Response: 401 Unauthorized

# Request with invalid API key
curl -X POST http://localhost:8000/api/voice-detection \
  -H "x-api-key: wrong_key"
# Response: 403 Forbidden
```

## 📊 Classification Results

| Classification | Description |
|---------------|-------------|
| `AI_GENERATED` | Voice generated using AI or synthetic systems |
| `HUMAN` | Voice spoken by a real human |

## 🧪 Testing

```bash
# Run the test script
python test_api.py

# Test with a specific audio file
python test_api.py path/to/audio.mp3 Tamil
```

## 📁 Project Structure

```
ai-voice-detection/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── main.py             # FastAPI application
│   ├── models.py           # Pydantic models
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── auth.py         # API key authentication
│   ├── routes/
│   │   ├── __init__.py
│   │   └── voice_detection.py  # API endpoints
│   └── services/
│       ├── __init__.py
│       ├── audio_processor.py  # Audio feature extraction
│       └── voice_detector.py   # Voice classification
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── test_api.py
└── README.md
```

## ⚙️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| API_SECRET_KEY | API authentication key | sk_test_123456789 |
| API_HOST | Server host | 0.0.0.0 |
| API_PORT | Server port | 8000 |
| DEBUG | Debug mode | false |

## 🔒 Security Considerations

1. **Change the default API key** in production
2. Use HTTPS in production
3. Implement rate limiting for production use
4. Consider adding request logging for auditing

## 📈 Performance Tips

- Audio files should be under 5 minutes for optimal performance
- Higher quality audio provides better detection accuracy
- Clear speech samples without background noise work best

## 🌐 Deployment Options

### Railway
```bash
# railway.toml is included
railway deploy
```

### Render
Deploy using the Dockerfile with port 8000

### AWS/GCP/Azure
Use Docker container or deploy as a Python application

## 📝 License

MIT License - Feel free to use for hackathons and learning!

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

Built with ❤️ for GUV HCL Hackathon
