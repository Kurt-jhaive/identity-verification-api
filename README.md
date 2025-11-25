# Identity Verification API

A production-ready FastAPI backend for face matching between live selfies and ID card photos, specifically designed for Philippine Government ID verification.

## Features

✅ **Privacy-First**: Images are processed entirely in memory - never saved to disk  
✅ **Face Detection**: Automatically detects faces in both images  
✅ **Face Matching**: Uses Euclidean distance for accurate face comparison  
✅ **Detailed Error Codes**: Specific error responses for frontend handling  
✅ **Production-Ready**: Includes logging, error handling, and CORS support  
✅ **React Native Compatible**: Multipart form-data support with clear JSON responses  

## Requirements

- Python 3.8+
- CMake (required for dlib)
- C++ compiler (required for dlib)

### Windows Installation

1. Install Visual Studio Build Tools or Visual Studio with C++ support
2. Install CMake: `choco install cmake` (using Chocolatey) or download from [cmake.org](https://cmake.org/)

### macOS Installation

```bash
brew install cmake
```

### Linux Installation

```bash
sudo apt-get update
sudo apt-get install cmake build-essential
```

## Installation

1. Clone or navigate to this directory:
```bash
cd identity-verification-api
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - **Windows**: `venv\Scripts\activate`
   - **macOS/Linux**: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

### Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at: `http://localhost:8000`

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### `POST /api/verify`

Verify identity by comparing a live selfie with an ID card photo.

**Request:**
- Content-Type: `multipart/form-data`
- Parameters:
  - `selfie`: Image file (JPEG/PNG)
  - `id_card`: Image file (JPEG/PNG)

**Success Response (200):**
```json
{
  "match": true,
  "confidence_score": 98.5,
  "message": "Verification successful. The faces match."
}
```

**Error Response (400):**
```json
{
  "error": "Face Detection Failed",
  "error_code": "NO_FACE_IN_SELFIE",
  "message": "No face detected in selfie. Please ensure the face is clearly visible and well-lit."
}
```

### Error Codes

| Error Code | Description |
|------------|-------------|
| `NO_FACE_IN_SELFIE` | No face detected in the selfie image |
| `NO_FACE_IN_ID` | No face detected in the ID card image |
| `MULTIPLE_FACES_IN_SELFIE` | Multiple faces detected in selfie |
| `MULTIPLE_FACES_IN_ID` | Multiple faces detected in ID card |
| `ENCODING_FAILED_SELFIE` | Failed to encode face from selfie |
| `ENCODING_FAILED_ID` | Failed to encode face from ID card |
| `INVALID_SELFIE_TYPE` | Invalid file type for selfie |
| `INVALID_ID_TYPE` | Invalid file type for ID card |
| `FILE_TOO_LARGE` | File size exceeds 10MB limit |
| `INTERNAL_ERROR` | Unexpected server error |

### `GET /health`

Health check endpoint to verify the service is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "Identity Verification API",
  "face_recognition_available": true,
  "threshold": 0.6
}
```

## Configuration

You can adjust the face matching threshold in `main.py`:

```python
FACE_MATCH_THRESHOLD = 0.6  # Lower is more strict (recommended: 0.5-0.6)
```

- **0.6**: Recommended for most use cases (balanced)
- **0.5**: More strict (fewer false positives, but may reject valid matches)
- **0.7**: More lenient (more false positives)

## React Native Integration

### Example Request

```javascript
const verifyIdentity = async (selfieUri, idCardUri) => {
  const formData = new FormData();
  
  formData.append('selfie', {
    uri: selfieUri,
    type: 'image/jpeg',
    name: 'selfie.jpg',
  });
  
  formData.append('id_card', {
    uri: idCardUri,
    type: 'image/jpeg',
    name: 'id_card.jpg',
  });
  
  try {
    const response = await fetch('http://YOUR_SERVER_IP:8000/api/verify', {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    const result = await response.json();
    
    if (response.ok) {
      if (result.match) {
        console.log(`Verified! Confidence: ${result.confidence_score}%`);
      } else {
        console.log('Verification failed: Faces do not match');
      }
    } else {
      // Handle specific error codes
      console.error(`Error: ${result.error_code} - ${result.message}`);
    }
  } catch (error) {
    console.error('Network error:', error);
  }
};
```

## Testing with cURL

### Windows PowerShell:
```powershell
$selfie = Get-Item -Path "selfie.jpg"
$idCard = Get-Item -Path "id_card.jpg"

curl.exe -X POST "http://localhost:8000/api/verify" `
  -F "selfie=@selfie.jpg" `
  -F "id_card=@id_card.jpg"
```

### macOS/Linux:
```bash
curl -X POST "http://localhost:8000/api/verify" \
  -F "selfie=@selfie.jpg" \
  -F "id_card=@id_card.jpg"
```

## Security Considerations

1. **No Disk Storage**: Images are processed in memory only - never saved to disk
2. **File Size Limits**: Maximum 10MB per image to prevent DoS attacks
3. **File Type Validation**: Only JPEG and PNG files are accepted
4. **CORS**: Configure `allow_origins` in production to restrict access
5. **HTTPS**: Always use HTTPS in production (configure with reverse proxy like Nginx)
6. **Rate Limiting**: Consider adding rate limiting middleware for production

## Production Deployment

### Using Docker (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:
```bash
docker build -t identity-verification-api .
docker run -p 8000:8000 identity-verification-api
```

### Using Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

## Performance Optimization

1. **Workers**: Use multiple Uvicorn workers for better concurrency
2. **Image Optimization**: Consider resizing large images before processing
3. **Caching**: Implement Redis caching for repeated verifications
4. **GPU Acceleration**: Use `model="cnn"` in face_locations for GPU support (requires dlib with CUDA)

## Troubleshooting

### dlib installation fails
- Ensure CMake is installed
- Install C++ compiler (Visual Studio Build Tools on Windows)
- Try: `pip install dlib --no-cache-dir`

### "No face detected" errors
- Ensure images are well-lit
- Face should be clearly visible and frontal
- Minimum recommended resolution: 640x480

### Low confidence scores
- Check image quality
- Ensure good lighting conditions
- Adjust `FACE_MATCH_THRESHOLD` if needed

## License

MIT License

## Support

For issues or questions, please open an issue on GitHub.
