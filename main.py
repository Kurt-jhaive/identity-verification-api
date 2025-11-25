"""
Identity Verification API
A production-ready FastAPI service for face matching between selfie and ID photos.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
from PIL import Image, ImageStat, ExifTags
import io
from typing import Tuple, Optional, Dict, List
import logging
from pydantic import BaseModel
import cv2
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Identity Verification API",
    description="Face matching API for mobile identity verification",
    version="1.0.0"
)

# Configure CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class VerificationResponse(BaseModel):
    match: bool
    confidence_score: float
    message: str
    distance: float
    warning: Optional[str] = None
    fraud_risk: str  # "low", "medium", "high"
    fraud_indicators: List[str]

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    message: str

# Configuration
FACE_MATCH_THRESHOLD = 0.5  # Stricter for ID verification (0.5 for high security)
CONFIDENCE_MULTIPLIER = 100  # For converting distance to percentage
IDENTICAL_IMAGE_THRESHOLD = 0.02  # Distance below this means likely same image
MIN_FACE_SIZE = (50, 50)  # Minimum face dimensions in pixels

# Fraud Detection Thresholds
MIN_IMAGE_QUALITY_SCORE = 20  # Minimum average pixel brightness/quality
MAX_COMPRESSION_ARTIFACTS = 30  # Maximum JPEG artifacts score
MIN_EDGE_DENSITY = 0.05  # Minimum edge density for real documents
MAX_BLUR_SCORE = 100  # Maximum acceptable Laplacian variance (lower = blurrier)


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes into numpy array.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        numpy array in RGB format
        
    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise ValueError(f"Invalid image format: {str(e)}")


def analyze_image_forensics(image: Image.Image, image_array: np.ndarray, image_type: str) -> Dict:
    """
    Perform forensic analysis to detect potential fake/manipulated images.
    
    Args:
        image: PIL Image object
        image_array: Numpy array of the image
        image_type: 'selfie' or 'id_card'
        
    Returns:
        Dict with forensic analysis results and fraud indicators
    """
    fraud_indicators = []
    
    # 1. Check EXIF data (real photos should have metadata)
    has_exif = False
    try:
        exif_data = image._getexif()
        if exif_data:
            has_exif = True
            # Check for suspicious editing software
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ['Software', 'ProcessingSoftware']:
                    suspicious_software = ['photoshop', 'gimp', 'paint', 'editor']
                    if any(s in str(value).lower() for s in suspicious_software):
                        fraud_indicators.append(f"Image edited with {value}")
    except:
        pass
    
    if not has_exif and image_type == 'selfie':
        fraud_indicators.append("No EXIF data in selfie (may be screenshot or edited)")
    
    # 2. Check image quality and compression artifacts
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < MAX_BLUR_SCORE:
            fraud_indicators.append(f"Image is blurry (score: {laplacian_var:.2f})")
        
        # Edge detection (real IDs have sharp edges and text)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < MIN_EDGE_DENSITY and image_type == 'id_card':
            fraud_indicators.append(f"Low edge density (may be photo of screen/printout)")
        
    except Exception as e:
        logger.warning(f"OpenCV analysis failed: {e}")
    
    # 3. Check color distribution (printed/screen photos have different color profiles)
    stat = ImageStat.Stat(image)
    avg_brightness = sum(stat.mean) / len(stat.mean)
    if avg_brightness < MIN_IMAGE_QUALITY_SCORE:
        fraud_indicators.append("Image too dark (poor quality or screenshot)")
    
    # 4. Check for screen patterns (moiré effect from photographing screens)
    if image_type == 'id_card':
        # Check for regular patterns that indicate screen capture
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            # FFT to detect periodic patterns
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            
            # High frequency peaks indicate screen patterns
            threshold = magnitude_spectrum.mean() + 2 * magnitude_spectrum.std()
            high_freq_count = np.sum(magnitude_spectrum > threshold)
            
            if high_freq_count > 1000:  # Arbitrary threshold
                fraud_indicators.append("Moiré patterns detected (photo of screen/monitor)")
        except:
            pass
    
    # 5. Check aspect ratio and resolution (Philippine National ID is specific size)
    width, height = image.size
    aspect_ratio = width / height
    
    if image_type == 'id_card':
        # Philippine National ID is approximately 85.6mm × 53.98mm (1.585 ratio)
        # Allow some tolerance for cropping
        if aspect_ratio < 1.3 or aspect_ratio > 1.9:
            fraud_indicators.append(f"Unusual aspect ratio for ID card: {aspect_ratio:.2f}")
        
        # Check resolution (too low = printout, too perfect = digital forgery)
        total_pixels = width * height
        if total_pixels < 100000:  # Less than ~316x316
            fraud_indicators.append("Very low resolution for ID card")
    
    # 6. Check for digital artifacts (copy-paste, clone stamp)
    try:
        # Simple duplicate region detection
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        # This is a simplified check - professional systems use more complex algorithms
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Unusually uniform histogram can indicate manipulation
        hist_std = np.std(hist)
        if hist_std < 100 and image_type == 'id_card':
            fraud_indicators.append("Suspicious histogram uniformity")
    except:
        pass
    
    return {
        "has_exif": has_exif,
        "fraud_indicators": fraud_indicators,
        "laplacian_var": laplacian_var if 'laplacian_var' in locals() else 0,
        "edge_density": edge_density if 'edge_density' in locals() else 0,
        "avg_brightness": avg_brightness,
        "aspect_ratio": aspect_ratio
    }


def check_face_quality(face_location: tuple, image_shape: tuple) -> dict:
    """
    Check if detected face meets quality requirements.
    
    Args:
        face_location: (top, right, bottom, left) coordinates
        image_shape: Shape of the image array
        
    Returns:
        Dict with quality metrics
    """
    top, right, bottom, left = face_location
    face_width = right - left
    face_height = bottom - top
    image_height, image_width = image_shape[:2]
    
    # Calculate face size as percentage of image
    face_area_percent = (face_width * face_height) / (image_width * image_height) * 100
    
    return {
        "width": face_width,
        "height": face_height,
        "area_percent": round(face_area_percent, 2),
        "is_too_small": face_width < MIN_FACE_SIZE[0] or face_height < MIN_FACE_SIZE[1]
    }


def detect_and_encode_face(image_array: np.ndarray, image_type: str) -> Tuple[np.ndarray, list]:
    """
    Detect face and generate encoding from image array.
    
    Args:
        image_array: Image as numpy array
        image_type: Type of image ('selfie' or 'id_card')
        
    Returns:
        Tuple of (face_encoding, face_locations)
        
    Raises:
        HTTPException: If no face or multiple faces detected
    """
    # Detect face locations
    face_locations = face_recognition.face_locations(image_array, model="hog")
    
    if len(face_locations) == 0:
        error_code = "NO_FACE_IN_SELFIE" if image_type == "selfie" else "NO_FACE_IN_ID"
        message = f"No face detected in {image_type}. Please ensure the face is clearly visible and well-lit."
        logger.warning(f"{error_code}: {message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Face Detection Failed",
                "error_code": error_code,
                "message": message
            }
        )
    
    if len(face_locations) > 1:
        error_code = "MULTIPLE_FACES_IN_SELFIE" if image_type == "selfie" else "MULTIPLE_FACES_IN_ID"
        message = f"Multiple faces detected in {image_type}. Please ensure only one face is visible."
        logger.warning(f"{error_code}: {message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Multiple Faces Detected",
                "error_code": error_code,
                "message": message
            }
        )
    
    # Check face quality
    quality = check_face_quality(face_locations[0], image_array.shape)
    if quality["is_too_small"]:
        error_code = "FACE_TOO_SMALL_SELFIE" if image_type == "selfie" else "FACE_TOO_SMALL_ID"
        message = f"Face in {image_type} is too small or unclear. Please ensure the face is clearly visible and takes up more of the image."
        logger.warning(f"{error_code}: Face size {quality['width']}x{quality['height']}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Face Quality Too Low",
                "error_code": error_code,
                "message": message
            }
        )
    
    # Generate face encoding
    face_encodings = face_recognition.face_encodings(image_array, face_locations)
    
    if len(face_encodings) == 0:
        error_code = "ENCODING_FAILED_SELFIE" if image_type == "selfie" else "ENCODING_FAILED_ID"
        message = f"Failed to encode face from {image_type}. Please try again with a clearer image."
        logger.error(f"{error_code}: {message}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Face Encoding Failed",
                "error_code": error_code,
                "message": message
            }
        )
    
    return face_encodings[0], face_locations


def calculate_face_distance(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two face encodings.
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        
    Returns:
        Euclidean distance (lower means more similar)
    """
    return float(np.linalg.norm(encoding1 - encoding2))


def distance_to_confidence(distance: float) -> float:
    """
    Convert distance to confidence score (0-100%).
    
    Args:
        distance: Euclidean distance between faces
        
    Returns:
        Confidence score as percentage
    """
    # Invert and normalize the distance to a 0-100 scale
    # Distance of 0 = 100% confidence, Distance of 1.0 = 0% confidence
    if distance > 1.0:
        return 0.0
    confidence = max(0, (1.0 - distance) * CONFIDENCE_MULTIPLIER)
    return round(confidence, 2)


def assess_fraud_risk(all_indicators: List[str]) -> str:
    """
    Assess overall fraud risk based on indicators.
    
    Args:
        all_indicators: List of all fraud indicators found
        
    Returns:
        Risk level: "low", "medium", or "high"
    """
    indicator_count = len(all_indicators)
    
    # High-risk indicators that alone warrant high risk
    high_risk_keywords = ['edited', 'photoshop', 'moiré', 'screen', 'monitor']
    has_high_risk = any(any(keyword in ind.lower() for keyword in high_risk_keywords) 
                        for ind in all_indicators)
    
    if has_high_risk or indicator_count >= 4:
        return "high"
    elif indicator_count >= 2:
        return "medium"
    else:
        return "low"


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Identity Verification API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "service": "Identity Verification API",
        "face_recognition_available": True,
        "threshold": FACE_MATCH_THRESHOLD
    }


@app.post("/api/verify", response_model=VerificationResponse)
async def verify_identity(
    selfie: UploadFile = File(..., description="Live selfie image"),
    id_card: UploadFile = File(..., description="ID card photo")
):
    """
    Verify identity by comparing a live selfie with an ID card photo.
    
    **Process:**
    1. Loads both images into memory (never saves to disk)
    2. Detects faces in both images
    3. Encodes faces into 128-dimensional vectors
    4. Calculates Euclidean distance
    5. Determines match based on threshold
    
    **Returns:**
    - match: Boolean indicating if faces match
    - confidence_score: Percentage (0-100) representing similarity
    - message: Human-readable result message
    
    **Error Codes:**
    - NO_FACE_IN_SELFIE: No face detected in selfie
    - NO_FACE_IN_ID: No face detected in ID card
    - MULTIPLE_FACES_IN_SELFIE: Multiple faces in selfie
    - MULTIPLE_FACES_IN_ID: Multiple faces in ID card
    - ENCODING_FAILED_SELFIE: Failed to encode selfie
    - ENCODING_FAILED_ID: Failed to encode ID card
    """
    try:
        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        
        if selfie.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid File Type",
                    "error_code": "INVALID_SELFIE_TYPE",
                    "message": f"Selfie must be JPEG or PNG. Received: {selfie.content_type}"
                }
            )
        
        if id_card.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid File Type",
                    "error_code": "INVALID_ID_TYPE",
                    "message": f"ID card must be JPEG or PNG. Received: {id_card.content_type}"
                }
            )
        
        logger.info("Starting identity verification process")
        
        # Read images into memory (NO disk storage)
        selfie_bytes = await selfie.read()
        id_card_bytes = await id_card.read()
        
        logger.info(f"Received selfie: {len(selfie_bytes)} bytes, ID card: {len(id_card_bytes)} bytes")
        
        # Validate file sizes (prevent DoS attacks)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(selfie_bytes) > MAX_FILE_SIZE or len(id_card_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "File Too Large",
                    "error_code": "FILE_TOO_LARGE",
                    "message": "Image files must be less than 10MB"
                }
            )
        
        # Load images from bytes
        try:
            selfie_image = load_image_from_bytes(selfie_bytes)
            selfie_pil = Image.open(io.BytesIO(selfie_bytes))
            logger.info(f"Selfie loaded: shape {selfie_image.shape}")
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Image",
                    "error_code": "INVALID_SELFIE_FORMAT",
                    "message": str(e)
                }
            )
        
        try:
            id_card_image = load_image_from_bytes(id_card_bytes)
            id_card_pil = Image.open(io.BytesIO(id_card_bytes))
            logger.info(f"ID card loaded: shape {id_card_image.shape}")
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Image",
                    "error_code": "INVALID_ID_FORMAT",
                    "message": str(e)
                }
            )
        
        # Perform forensic analysis
        logger.info("Performing forensic analysis...")
        selfie_forensics = analyze_image_forensics(selfie_pil, selfie_image, "selfie")
        id_forensics = analyze_image_forensics(id_card_pil, id_card_image, "id_card")
        
        all_fraud_indicators = selfie_forensics["fraud_indicators"] + id_forensics["fraud_indicators"]
        
        logger.info(f"Fraud indicators found: {len(all_fraud_indicators)}")
        for indicator in all_fraud_indicators:
            logger.warning(f"⚠️ {indicator}")
        
        # Detect and encode faces
        logger.info("Detecting face in selfie...")
        selfie_encoding, selfie_locations = detect_and_encode_face(selfie_image, "selfie")
        logger.info(f"Selfie face detected at: {selfie_locations}")
        
        logger.info("Detecting face in ID card...")
        id_encoding, id_locations = detect_and_encode_face(id_card_image, "id_card")
        logger.info(f"ID card face detected at: {id_locations}")
        
        # Calculate face distance (Euclidean distance)
        distance = calculate_face_distance(selfie_encoding, id_encoding)
        logger.info(f"Face distance calculated: {distance}")
        
        # Check if images are suspiciously identical (potential fraud)
        if distance < IDENTICAL_IMAGE_THRESHOLD:
            logger.warning(f"⚠️ SUSPICIOUS: Identical images detected (distance: {distance:.4f})")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Identical Images Detected",
                    "error_code": "IDENTICAL_IMAGES",
                    "message": "The selfie and ID card photo appear to be the same image. Please upload a live selfie and a separate ID card photo."
                }
            )
        
        # Determine if faces match
        is_match = distance <= FACE_MATCH_THRESHOLD
        
        # Convert distance to confidence score
        confidence_score = distance_to_confidence(distance)
        
        # Assess fraud risk
        fraud_risk = assess_fraud_risk(all_fraud_indicators)
        
        # Generate response message and warnings
        warning = None
        if is_match:
            message = "Verification successful. The faces match."
            # Add warning for matches that are close to threshold
            if distance > (FACE_MATCH_THRESHOLD * 0.85):
                warning = "Match found but confidence is moderate. Consider manual review for high-security applications."
            # Override for high fraud risk
            if fraud_risk == "high":
                warning = "⚠️ ALERT: High fraud risk detected. Manual verification strongly recommended."
            logger.info(f"✓ MATCH - Distance: {distance:.4f}, Confidence: {confidence_score}%, Fraud Risk: {fraud_risk}")
        else:
            message = "Verification failed. The faces do not match."
            logger.info(f"✗ NO MATCH - Distance: {distance:.4f}, Confidence: {confidence_score}%, Fraud Risk: {fraud_risk}")
        
        return VerificationResponse(
            match=is_match,
            confidence_score=confidence_score,
            message=message,
            distance=round(distance, 4),
            warning=warning,
            fraud_risk=fraud_risk,
            fraud_indicators=all_fraud_indicators
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (these are handled errors)
        raise
    
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error during verification: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred during verification. Please try again."
            }
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for consistent error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {
            "error": "Error",
            "error_code": "UNKNOWN_ERROR",
            "message": str(exc.detail)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
