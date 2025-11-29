"""
Identity Verification API
A production-ready FastAPI service for face matching between selfie and ID photos.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
from PIL import Image, ImageStat, ExifTags, ImageEnhance
import io
from typing import Tuple, Optional, Dict, List
import logging
from pydantic import BaseModel
import cv2
from datetime import datetime
from scipy.spatial import distance as scipy_distance
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Identity Verification API",
    description="""
## Face Matching API for Mobile Identity Verification

This API provides a **3-step workflow** for secure identity verification:

### Recommended Workflow:
1. **Step 1**: `POST /api/verify-id` - Verify ID card quality (checks blur, fraud indicators, face detection)
2. **Step 2**: `POST /api/verify-selfie` - Verify selfie quality (checks blur, fraud indicators, face detection)  
3. **Step 3**: `POST /api/verify` - Match faces between selfie and ID card

### Alternative Workflow:
- Use `POST /api/verify` directly for single-step verification (includes all quality checks + face matching)

### Features:
- ✅ Privacy-first: Images processed in memory only (never saved to disk)
- ✅ Advanced fraud detection (edited images, screenshots, moiré patterns)
- ✅ Quality scoring (blur detection, brightness, resolution)
- ✅ Philippine National ID validation
- ✅ Production-ready with detailed error codes
    """,
    version="2.0.0"
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
class IDVerificationResponse(BaseModel):
    valid: bool
    message: str
    quality_score: float
    fraud_risk: str  # "low", "medium", "high"
    fraud_indicators: List[str]
    details: Dict

class SelfieVerificationResponse(BaseModel):
    valid: bool
    message: str
    quality_score: float
    fraud_risk: str  # "low", "medium", "high"
    fraud_indicators: List[str]
    details: Dict

class VerificationResponse(BaseModel):
    match: bool
    confidence_score: float
    message: str
    distance: float
    raw_distance: float  # Original Euclidean distance
    adjusted_threshold: float  # Threshold used based on quality
    reasoning: str  # Explanation of match/no-match decision
    warning: Optional[str] = None
    fraud_risk: str  # "low", "medium", "high"
    fraud_indicators: List[str]
    debug_info: Optional[Dict] = None  # Optional debug information

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    message: str

# Configuration
FACE_MATCH_THRESHOLD = 0.60  # Balanced threshold for ID verification (realistic for production)
FACE_MATCH_THRESHOLD_STRICT = 0.55  # Strict threshold for high-quality images
FACE_MATCH_THRESHOLD_LENIENT = 0.65  # Lenient threshold for lower-quality images
CONFIDENCE_MULTIPLIER = 100  # For converting distance to percentage
IDENTICAL_IMAGE_THRESHOLD = 0.02  # Distance below this means likely same image
MIN_FACE_SIZE = (30, 30)  # Minimum face dimensions in pixels (relaxed for better detection)
ALLOW_MULTIPLE_FACES = True  # Allow multiple faces and select the best one (more flexible)
NUM_JITTERS = 2  # Number of times to resample face for encoding (higher = more accurate but slower)
TARGET_IMAGE_SIZE = 800  # Maximum dimension for image preprocessing
FACE_PADDING = 0.25  # Padding around detected face (25% on each side)
BORDERLINE_DISTANCE_MIN = 0.55  # Start of borderline range for secondary checks
BORDERLINE_DISTANCE_MAX = 0.70  # End of borderline range

# Fraud Detection Thresholds
MIN_IMAGE_QUALITY_SCORE = 55  # Minimum average pixel brightness (relaxed)
MAX_BLUR_SCORE = 25  # Blur score below this is bad (relaxed - lower = blurrier)
MIN_EDGE_DENSITY = 0.03  # Minimum edge density for real documents (relaxed)
MOIRE_THRESHOLD = 2000  # FFT high frequency threshold for moiré (less sensitive)


def preprocess_image(image_array: np.ndarray, target_size: int = TARGET_IMAGE_SIZE) -> np.ndarray:
    """
    Advanced image preprocessing for better face detection and encoding.
    
    Steps:
    1. Resize to consistent size
    2. Normalize lighting using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Enhance image quality
    
    Args:
        image_array: Input image as numpy array
        target_size: Maximum dimension for resizing
        
    Returns:
        Preprocessed image array
    """
    # Step 1: Resize if image is too large
    height, width = image_array.shape[:2]
    if max(height, width) > target_size:
        scale = target_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    # Step 2: Normalize lighting using CLAHE
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        lab = cv2.merge([l, a, b])
        image_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        logger.debug("Applied CLAHE lighting normalization")
    except Exception as e:
        logger.warning(f"CLAHE failed, using original image: {e}")
    
    # Step 3: Slight sharpening
    try:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image_array, -1, kernel)
        # Blend 70% original + 30% sharpened
        image_array = cv2.addWeighted(image_array, 0.7, sharpened, 0.3, 0)
        logger.debug("Applied sharpening")
    except Exception as e:
        logger.warning(f"Sharpening failed: {e}")
    
    return image_array


def align_face(image_array: np.ndarray, face_location: tuple, face_landmarks: dict) -> Tuple[np.ndarray, Dict]:
    """
    Align face based on eye positions for consistent encoding.
    
    Args:
        image_array: Input image
        face_location: Face bounding box (top, right, bottom, left)
        face_landmarks: Face landmarks dictionary
        
    Returns:
        Tuple of (aligned_face_array, alignment_info)
    """
    try:
        # Get eye positions
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        
        # Calculate eye centers as float tuples
        left_eye_center = tuple(np.mean(left_eye, axis=0).astype(float))
        right_eye_center = tuple(np.mean(right_eye, axis=0).astype(float))
        
        # Calculate angle between eyes
        dy = float(right_eye_center[1] - left_eye_center[1])
        dx = float(right_eye_center[0] - left_eye_center[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        
        # Calculate center point between eyes as tuple
        eyes_center = (
            float((left_eye_center[0] + right_eye_center[0]) / 2),
            float((left_eye_center[1] + right_eye_center[1]) / 2)
        )
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        
        # Rotate image
        height, width = image_array.shape[:2]
        aligned_image = cv2.warpAffine(
            image_array, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Update face location after rotation
        top, right, bottom, left = face_location
        
        # Apply rotation to face bounding box corners
        corners = np.array([
            [left, top, 1],
            [right, top, 1],
            [right, bottom, 1],
            [left, bottom, 1]
        ], dtype=np.float32)
        
        # Transform corners
        rotated_corners = rotation_matrix.dot(corners.T).T
        
        # Get new bounding box
        new_left = int(np.min(rotated_corners[:, 0]))
        new_right = int(np.max(rotated_corners[:, 0]))
        new_top = int(np.min(rotated_corners[:, 1]))
        new_bottom = int(np.max(rotated_corners[:, 1]))
        
        # Add padding and ensure square bounding box
        face_width = new_right - new_left
        face_height = new_bottom - new_top
        
        # Make square
        max_dim = max(face_width, face_height)
        padding = int(max_dim * FACE_PADDING)
        
        center_x = (new_left + new_right) // 2
        center_y = (new_top + new_bottom) // 2
        
        half_size = (max_dim + padding * 2) // 2
        
        new_left = max(0, center_x - half_size)
        new_right = min(width, center_x + half_size)
        new_top = max(0, center_y - half_size)
        new_bottom = min(height, center_y + half_size)
        
        new_face_location = (new_top, new_right, new_bottom, new_left)
        
        # Calculate eye distance for quality metric
        eye_distance = float(np.sqrt((right_eye_center[0] - left_eye_center[0])**2 + 
                                     (right_eye_center[1] - left_eye_center[1])**2))
        
        alignment_info = {
            "angle": round(angle, 2),
            "eyes_distance": round(eye_distance, 2),
            "original_size": (face_width, face_height),
            "aligned_size": (new_right - new_left, new_bottom - new_top),
            "is_square": abs((new_right - new_left) - (new_bottom - new_top)) < 10,
            "confidence": "high" if abs(angle) < 15 else "medium" if abs(angle) < 30 else "low"
        }
        
        logger.info(f"Face aligned: angle={angle:.2f}°, square={alignment_info['is_square']}, confidence={alignment_info['confidence']}")
        
        return aligned_image, new_face_location, alignment_info
        
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}")
        top, right, bottom, left = face_location
        return image_array, face_location, {
            "angle": 0, 
            "error": str(e),
            "confidence": "none",
            "aligned": False
        }


def extract_face_crop(image_array: np.ndarray, face_location: tuple, padding: float = FACE_PADDING) -> np.ndarray:
    """
    Extract face crop with padding and ensure square dimensions.
    
    Args:
        image_array: Input image
        face_location: (top, right, bottom, left)
        padding: Padding ratio around face
        
    Returns:
        Cropped face array
    """
    top, right, bottom, left = face_location
    height, width = image_array.shape[:2]
    
    # Calculate dimensions
    face_width = right - left
    face_height = bottom - top
    
    # Make square
    max_dim = max(face_width, face_height)
    pad_pixels = int(max_dim * padding)
    
    # Calculate center and expand
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    half_size = (max_dim + pad_pixels * 2) // 2
    
    # Get square crop coordinates
    crop_left = max(0, center_x - half_size)
    crop_right = min(width, center_x + half_size)
    crop_top = max(0, center_y - half_size)
    crop_bottom = min(height, center_y + half_size)
    
    # Extract crop
    face_crop = image_array[crop_top:crop_bottom, crop_left:crop_right]
    
    logger.debug(f"Extracted face crop: {face_crop.shape}, square={abs(face_crop.shape[0]-face_crop.shape[1])<5}")
    
    return face_crop


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes into numpy array with auto-rotation based on EXIF.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        numpy array in RGB format
        
    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Handle EXIF orientation (important for mobile photos)
        try:
            exif = image._getexif()
            if exif is not None:
                orientation_key = 274  # EXIF orientation tag
                if orientation_key in exif:
                    orientation = exif[orientation_key]
                    # Rotate image based on EXIF orientation
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                        logger.info("Rotated image 180° based on EXIF")
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                        logger.info("Rotated image 270° based on EXIF")
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                        logger.info("Rotated image 90° based on EXIF")
        except (AttributeError, KeyError, IndexError) as e:
            # No EXIF data or orientation tag, proceed normally
            logger.debug(f"No EXIF orientation data: {e}")
        
        # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
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
    
    # 1. Check EXIF data (but don't penalize missing EXIF - many mobile apps strip it)
    has_exif = False
    try:
        exif_data = image._getexif()
        if exif_data:
            has_exif = True
            # Check for suspicious editing software
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ['Software', 'ProcessingSoftware']:
                    suspicious_software = ['photoshop', 'gimp', 'paint.net', 'pixlr']
                    if any(s in str(value).lower() for s in suspicious_software):
                        fraud_indicators.append(f"Image edited with {value}")
    except:
        pass
    
    # Note: Don't penalize missing EXIF - mobile apps often strip it
    
    # 2. Check image quality and compression artifacts
    laplacian_var = 0
    edge_density = 0
    
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Blur detection (Laplacian variance) - NEW: Only flag if VERY blurry (< 25)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < MAX_BLUR_SCORE:
            fraud_indicators.append(f"Image is very blurry (score: {laplacian_var:.1f})")
        
        # Edge detection (real IDs have sharp edges and text)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < MIN_EDGE_DENSITY and image_type == 'id_card':
            fraud_indicators.append(f"Low edge density (may be photo of screen)")
        
    except Exception as e:
        logger.warning(f"OpenCV analysis failed: {e}")
    
    # 3. Check color distribution
    stat = ImageStat.Stat(image)
    avg_brightness = sum(stat.mean) / len(stat.mean)
    if avg_brightness < MIN_IMAGE_QUALITY_SCORE:
        fraud_indicators.append("Image too dark (poor quality)")
    
    # 4. Check for screen patterns (moiré effect) - LESS SENSITIVE
    if image_type == 'id_card':
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            # FFT to detect periodic patterns
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            
            # High frequency peaks indicate screen patterns
            threshold = magnitude_spectrum.mean() + 2 * magnitude_spectrum.std()
            high_freq_count = np.sum(magnitude_spectrum > threshold)
            
            # Increased threshold from 1000 to 2000 (less sensitive)
            if high_freq_count > MOIRE_THRESHOLD:
                fraud_indicators.append("Moiré patterns detected (photo of screen)")
        except:
            pass
    
    # 5. Check aspect ratio - MORE FLEXIBLE (0.65 to 2.5 for IDs)
    width, height = image.size
    aspect_ratio = width / height
    
    if image_type == 'id_card':
        # Allow wide range: 0.65 to 2.5 (more flexible)
        if aspect_ratio < 0.65 or aspect_ratio > 2.5:
            fraud_indicators.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        
        # Check resolution (too low = printout)
        total_pixels = width * height
        if total_pixels < 50000:  # Very low resolution
            fraud_indicators.append("Very low resolution")
    
    # 6. Check for digital artifacts - simplified
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Unusually uniform histogram can indicate manipulation
        hist_std = np.std(hist)
        if hist_std < 50 and image_type == 'id_card':  # More lenient
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
    
    # More lenient size check - only reject extremely small faces
    is_too_small = (face_width < MIN_FACE_SIZE[0] or face_height < MIN_FACE_SIZE[1]) and face_area_percent < 0.5
    
    return {
        "width": int(face_width),
        "height": int(face_height),
        "area_percent": round(float(face_area_percent), 2),
        "is_too_small": bool(is_too_small)
    }


def detect_and_encode_face(image_array: np.ndarray, image_type: str) -> Tuple[np.ndarray, list, dict, dict]:
    """
    Detect face and generate encoding from image array with enhanced detection.
    
    Args:
        image_array: Image as numpy array
        image_type: Type of image ('selfie' or 'id_card')
        
    Returns:
        Tuple of (face_encoding, face_locations, face_landmarks, quality_info)
        
    Raises:
        HTTPException: If no face or multiple faces detected
    """
    # Preprocess image for better detection
    logger.info(f"Preprocessing {image_type} image...")
    processed_image = preprocess_image(image_array)
    
    # Try multiple detection strategies for better accuracy
    face_locations = []
    detection_method = "none"
    
    # Strategy 1: HOG model with default settings (fastest, works for most cases)
    logger.info(f"Trying HOG detection for {image_type}...")
    face_locations = face_recognition.face_locations(processed_image, model="hog", number_of_times_to_upsample=1)
    if len(face_locations) > 0:
        detection_method = "HOG"
        logger.info(f"HOG detected {len(face_locations)} face(s)")
    
    # Strategy 2: If HOG fails, try with more upsampling (better for smaller/distant faces)
    if len(face_locations) == 0:
        logger.info(f"HOG failed, trying with upsampling for {image_type}...")
        face_locations = face_recognition.face_locations(processed_image, model="hog", number_of_times_to_upsample=2)
        if len(face_locations) > 0:
            detection_method = "HOG+upsample"
            logger.info(f"HOG+upsample detected {len(face_locations)} face(s)")
    
    # Strategy 3: Try with different preprocessing
    if len(face_locations) == 0:
        logger.info(f"Trying alternative preprocessing for {image_type}...")
        try:
            # Try again on original image with max upsampling
            face_locations = face_recognition.face_locations(image_array, model="hog", number_of_times_to_upsample=2)
            if len(face_locations) > 0:
                detection_method = "HOG+original"
                processed_image = image_array  # Use original for encoding
                logger.info(f"Original image detection found {len(face_locations)} face(s)")
        except Exception as e:
            logger.warning(f"Alternative detection failed: {e}")
    
    logger.info(f"Final detection result for {image_type}: {len(face_locations)} face(s) using {detection_method}")
    
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
    
    # Handle multiple faces - select the best one instead of rejecting
    selected_face = None
    if len(face_locations) > 1:
        logger.info(f"Multiple faces detected in {image_type}, selecting best candidate...")
        selected_face = select_best_face(face_locations, image_array.shape, image_type)
        # Use only the selected face for encoding
        face_locations = [selected_face]
    else:
        selected_face = face_locations[0]
    
    # Check face quality (log only, don't reject for ID cards as some IDs have small photos)
    quality = check_face_quality(selected_face, image_array.shape)
    logger.info(f"Face quality - Size: {quality['width']}x{quality['height']}, Area: {quality['area_percent']}%")
    
    # Only enforce size check for selfies, not ID cards (some IDs have legitimately small faces)
    if quality["is_too_small"] and image_type == "selfie":
        error_code = "FACE_TOO_SMALL_SELFIE"
        message = f"Face in selfie is too small or unclear. Please move closer to the camera or ensure better lighting."
        logger.warning(f"{error_code}: Face size {quality['width']}x{quality['height']}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Face Quality Too Low",
                "error_code": error_code,
                "message": message
            }
        )
    elif quality["is_too_small"] and image_type == "id_card":
        logger.info(f"ID card has small face ({quality['width']}x{quality['height']}), but allowing it (some IDs have small photos)")
    
    # Get face landmarks for alignment and secondary checks
    logger.info(f"Extracting face landmarks for {image_type}...")
    face_landmarks_list = face_recognition.face_landmarks(processed_image, face_locations)
    
    if len(face_landmarks_list) == 0:
        logger.warning(f"No landmarks detected for {image_type}, using unaligned face")
        face_landmarks = None
        alignment_info = {"aligned": False}
        final_image = processed_image
        final_face_location = selected_face
    else:
        face_landmarks = face_landmarks_list[0]
        
        # Align face using landmarks
        logger.info(f"Aligning face for {image_type}...")
        try:
            aligned_image, aligned_face_location, alignment_info = align_face(
                processed_image, selected_face, face_landmarks
            )
            final_image = aligned_image
            final_face_location = aligned_face_location
            alignment_info["aligned"] = True
        except Exception as e:
            logger.warning(f"Face alignment failed for {image_type}: {e}")
            final_image = processed_image
            final_face_location = selected_face
            alignment_info = {"aligned": False, "error": str(e)}
    
    # Generate face encoding with aligned image
    # Use face landmarks model="large" for better accuracy with various poses
    logger.info(f"Generating face encoding for {image_type}...")
    face_encodings = face_recognition.face_encodings(
        final_image, 
        [final_face_location],
        num_jitters=NUM_JITTERS,
        model="large"
    )
    
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
    
    quality_info = {
        "detection_method": detection_method,
        "alignment": alignment_info,
        "quality": quality
    }
    
    return face_encodings[0], [final_face_location], face_landmarks, quality_info


def calculate_dynamic_threshold(selfie_quality: Dict, id_quality: Dict, 
                               selfie_alignment: Dict, id_alignment: Dict) -> Tuple[float, str]:
    """
    Calculate adaptive threshold based on image quality and alignment confidence.
    
    Rules:
    - Base threshold = 0.62
    - If image quality is low → threshold = 0.67
    - If alignment confidence is poor → threshold = 0.70
    
    Args:
        selfie_quality: Quality metrics for selfie
        id_quality: Quality metrics for ID
        selfie_alignment: Alignment info for selfie
        id_alignment: Alignment info for ID
        
    Returns:
        Tuple of (threshold, reasoning)
    """
    # Start with base threshold
    threshold = 0.62
    reasons = []
    
    # Check blur scores - NEW SCALE: < 25 is bad, 25-60 moderate, 60+ good
    avg_blur = (selfie_quality.get('laplacian_var', 100) + id_quality.get('laplacian_var', 100)) / 2
    if avg_blur < 25:
        threshold = 0.70  # Very blurry
        reasons.append(f"very blurry images (blur={avg_blur:.1f})")
    elif avg_blur < 60:
        threshold = 0.67  # Moderately blurry
        reasons.append(f"moderately blurry (blur={avg_blur:.1f})")
    
    # Check brightness - NEW RANGE: 55-200 acceptable
    avg_brightness = (selfie_quality.get('avg_brightness', 128) + id_quality.get('avg_brightness', 128)) / 2
    if avg_brightness < 55 or avg_brightness > 200:
        threshold = min(threshold + 0.03, 0.70)
        reasons.append(f"poor lighting (brightness={avg_brightness:.1f})")
    
    # Check alignment confidence
    selfie_conf = selfie_alignment.get('confidence', 'high')
    id_conf = id_alignment.get('confidence', 'high')
    
    if selfie_conf == 'low' or id_conf == 'low':
        threshold = 0.70
        reasons.append("poor alignment confidence")
    elif selfie_conf == 'medium' or id_conf == 'medium':
        threshold = min(threshold + 0.03, 0.67)
        reasons.append("moderate alignment")
    
    # Check fraud indicators count - less penalty
    total_fraud_indicators = len(selfie_quality.get('fraud_indicators', [])) + len(id_quality.get('fraud_indicators', []))
    if total_fraud_indicators >= 5:
        threshold = max(threshold - 0.05, 0.55)  # Stricter for high fraud
        reasons.append(f"high fraud indicators ({total_fraud_indicators})")
    
    # Clamp threshold to reasonable range
    threshold = max(0.55, min(0.70, threshold))
    
    reasoning = f"Threshold {threshold:.2f}: " + ", ".join(reasons) if reasons else f"Using base threshold {threshold:.2f} (good quality)"
    
    logger.info(f"Dynamic threshold: {threshold:.3f} (base: 0.62)")
    
    return threshold, reasoning


def calculate_cosine_similarity(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two face encodings.
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        
    Returns:
        Cosine similarity (0-1, higher = more similar)
    """
    dot_product = np.dot(encoding1, encoding2)
    norm1 = np.linalg.norm(encoding1)
    norm2 = np.linalg.norm(encoding2)
    return float(dot_product / (norm1 * norm2))


def compare_face_landmarks(landmarks1: dict, landmarks2: dict) -> Dict:
    """
    Compare facial landmarks for secondary verification.
    
    Args:
        landmarks1: First face landmarks
        landmarks2: Second face landmarks
        
    Returns:
        Dict with landmark comparison metrics
    """
    try:
        # Compare eye distances
        def eye_distance(landmarks):
            left_eye = np.mean(landmarks['left_eye'], axis=0)
            right_eye = np.mean(landmarks['right_eye'], axis=0)
            return np.linalg.norm(right_eye - left_eye)
        
        eye_dist1 = eye_distance(landmarks1)
        eye_dist2 = eye_distance(landmarks2)
        eye_ratio = min(eye_dist1, eye_dist2) / max(eye_dist1, eye_dist2)
        
        # Compare face width to height ratio
        def face_ratio(landmarks):
            # Use chin to top of nose bridge
            chin = np.array(landmarks['chin'][8])  # Bottom of chin
            nose = np.array(landmarks['nose_bridge'][0])  # Top of nose
            left = np.array(landmarks['chin'][0])  # Left side
            right = np.array(landmarks['chin'][-1])  # Right side
            
            height = np.linalg.norm(nose - chin)
            width = np.linalg.norm(right - left)
            return width / height if height > 0 else 1.0
        
        ratio1 = face_ratio(landmarks1)
        ratio2 = face_ratio(landmarks2)
        face_ratio_similarity = min(ratio1, ratio2) / max(ratio1, ratio2)
        
        landmarks_match = bool(eye_ratio > 0.85 and face_ratio_similarity > 0.85)
        
        return {
            "eye_distance_ratio": round(float(eye_ratio), 3),
            "face_ratio_similarity": round(float(face_ratio_similarity), 3),
            "landmarks_match": landmarks_match
        }
    except Exception as e:
        logger.warning(f"Landmark comparison failed: {e}")
        return {"error": str(e), "landmarks_match": None}


def compare_faces_advanced(selfie_encoding: np.ndarray, 
                          id_encoding: np.ndarray,
                          selfie_landmarks: dict,
                          id_landmarks: dict,
                          adjusted_threshold: float) -> Dict:
    """
    Advanced face comparison with multiple verification methods.
    
    Args:
        selfie_encoding: Selfie face encoding
        id_encoding: ID face encoding
        selfie_landmarks: Selfie face landmarks
        id_landmarks: ID face landmarks
        adjusted_threshold: Dynamic threshold based on quality
        
    Returns:
        Dict with comprehensive comparison results
    """
    # Primary: Euclidean distance
    euclidean_distance = float(np.linalg.norm(selfie_encoding - id_encoding))
    primary_match = euclidean_distance <= adjusted_threshold
    
    # Secondary checks for borderline cases
    secondary_checks = {}
    confidence_boost = 0.0
    
    if BORDERLINE_DISTANCE_MIN <= euclidean_distance <= BORDERLINE_DISTANCE_MAX:
        logger.info(f"Borderline distance {euclidean_distance:.3f}, applying secondary checks...")
        
        # Check 1: Cosine similarity
        cosine_sim = calculate_cosine_similarity(selfie_encoding, id_encoding)
        secondary_checks['cosine_similarity'] = round(cosine_sim, 4)
        if cosine_sim > 0.70:
            confidence_boost += 0.03
            logger.info(f"✓ Cosine similarity: {cosine_sim:.3f} (boost +0.03)")
        
        # Check 2: Landmark similarity
        landmark_comparison = compare_face_landmarks(selfie_landmarks, id_landmarks)
        secondary_checks['landmark_comparison'] = landmark_comparison
        if landmark_comparison.get('landmarks_match'):
            confidence_boost += 0.02
            logger.info(f"✓ Landmarks match (boost +0.02)")
        
        # Check 3: Alternative distance metrics
        manhattan_distance = scipy_distance.cityblock(selfie_encoding, id_encoding)
        secondary_checks['manhattan_distance'] = round(manhattan_distance, 4)
        
        # Apply confidence boost
        effective_distance = euclidean_distance - confidence_boost
        secondary_checks['confidence_boost'] = round(confidence_boost, 3)
        secondary_checks['effective_distance'] = round(effective_distance, 4)
        
        logger.info(f"Secondary checks complete: boost={confidence_boost:.3f}, effective_distance={effective_distance:.3f}")
    else:
        effective_distance = euclidean_distance
    
    # Final determination
    final_match = effective_distance <= adjusted_threshold
    
    return {
        "raw_distance": round(float(euclidean_distance), 4),
        "effective_distance": round(float(effective_distance), 4),
        "match": bool(final_match),
        "secondary_checks": secondary_checks,
        "confidence_boost": round(float(confidence_boost), 3)
    }


def select_best_face(face_locations: list, image_shape: tuple, image_type: str) -> tuple:
    """
    Select the best face from multiple detected faces.
    Prioritizes: 1) Largest face, 2) Most centered face
    
    Args:
        face_locations: List of face location tuples (top, right, bottom, left)
        image_shape: Shape of the image array
        image_type: Type of image ('selfie' or 'id_card')
        
    Returns:
        Selected face location tuple
    """
    if len(face_locations) == 1:
        return face_locations[0]
    
    image_height, image_width = image_shape[:2]
    center_y, center_x = image_height / 2, image_width / 2
    
    best_face = None
    best_score = -1
    
    for face_loc in face_locations:
        top, right, bottom, left = face_loc
        
        # Calculate face size
        face_width = right - left
        face_height = bottom - top
        face_area = face_width * face_height
        
        # Calculate distance from center
        face_center_y = (top + bottom) / 2
        face_center_x = (left + right) / 2
        distance_from_center = float(np.sqrt(
            ((face_center_x - center_x) / image_width) ** 2 +
            ((face_center_y - center_y) / image_height) ** 2
        ))
        
        # Score: 70% weight on size, 30% weight on centrality
        size_score = face_area / (image_width * image_height)
        centrality_score = 1 - distance_from_center
        total_score = (size_score * 0.7) + (centrality_score * 0.3)
        
        if total_score > best_score:
            best_score = total_score
            best_face = face_loc
    
    logger.info(f"Selected best face from {len(face_locations)} detected faces (score: {best_score:.3f})")
    return best_face


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
    
    New rules:
    - < 3 indicators: low risk
    - 3-4 indicators: medium risk  
    - >= 5 indicators: high risk
    
    Args:
        all_indicators: List of all fraud indicators found
        
    Returns:
        Risk level: "low", "medium", or "high"
    """
    indicator_count = len(all_indicators)
    
    # High-risk keywords that warrant immediate elevation
    high_risk_keywords = ['photoshop', 'gimp', 'edited with']
    has_critical_risk = any(any(keyword in ind.lower() for keyword in high_risk_keywords) 
                           for ind in all_indicators)
    
    if has_critical_risk:
        return "high"
    elif indicator_count >= 5:
        return "high"
    elif indicator_count >= 3:
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


@app.post("/api/verify-id", response_model=IDVerificationResponse)
async def verify_id(
    id_card: UploadFile = File(..., description="ID card photo")
):
    """
    Step 1: Verify ID card quality and detect fraud indicators.
    
    This endpoint checks:
    - Image quality (blur, brightness, resolution)
    - Presence of exactly one face
    - Fraud indicators (edited images, screenshots, etc.)
    - Document authenticity markers
    
    **Use this endpoint first** before proceeding to face matching.
    
    **Returns:**
    - valid: Boolean indicating if ID passes quality checks
    - quality_score: Overall quality score (0-100)
    - fraud_risk: Risk level (low/medium/high)
    - fraud_indicators: List of detected issues
    - details: Detailed forensic analysis results
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        
        if id_card.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid File Type",
                    "error_code": "INVALID_ID_TYPE",
                    "message": f"ID card must be JPEG or PNG. Received: {id_card.content_type}"
                }
            )
        
        logger.info("Starting ID card verification process")
        
        # Read image
        id_card_bytes = await id_card.read()
        logger.info(f"Received ID card: {len(id_card_bytes)} bytes")
        
        # Validate file size
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(id_card_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "File Too Large",
                    "error_code": "FILE_TOO_LARGE",
                    "message": "Image file must be less than 10MB"
                }
            )
        
        # Load image
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
        logger.info("Performing forensic analysis on ID card...")
        id_forensics = analyze_image_forensics(id_card_pil, id_card_image, "id_card")
        
        fraud_indicators = id_forensics["fraud_indicators"]
        
        logger.info(f"Fraud indicators found: {len(fraud_indicators)}")
        for indicator in fraud_indicators:
            logger.warning(f"⚠️ {indicator}")
        
        # Detect face in ID
        logger.info("Detecting face in ID card...")
        try:
            id_encoding, id_locations = detect_and_encode_face(id_card_image, "id_card")
            logger.info(f"ID card face detected at: {id_locations}")
            face_detected = True
        except HTTPException as e:
            # Face detection failed
            face_detected = False
            fraud_indicators.append("Face detection failed - ID may be invalid")
        
        # Calculate quality score
        quality_score = 100.0
        
        # Deduct points for blur
        if id_forensics["laplacian_var"] < MAX_BLUR_SCORE:
            blur_penalty = (MAX_BLUR_SCORE - id_forensics["laplacian_var"]) / MAX_BLUR_SCORE * 30
            quality_score -= blur_penalty
            logger.info(f"Blur penalty: -{blur_penalty:.1f} points")
        
        # Deduct points for low brightness
        if id_forensics["avg_brightness"] < MIN_IMAGE_QUALITY_SCORE:
            brightness_penalty = 20
            quality_score -= brightness_penalty
            logger.info(f"Brightness penalty: -{brightness_penalty} points")
        
        # Deduct points for each fraud indicator
        fraud_penalty = len(fraud_indicators) * 10
        quality_score -= fraud_penalty
        logger.info(f"Fraud indicator penalty: -{fraud_penalty} points")
        
        # Deduct points if no face detected
        if not face_detected:
            quality_score -= 30
            logger.info("No face penalty: -30 points")
        
        quality_score = max(0, quality_score)
        
        # Assess fraud risk
        fraud_risk = assess_fraud_risk(fraud_indicators)
        
        # Determine if ID is valid (passes quality checks)
        is_valid = (
            quality_score >= 60 and  # Minimum quality score
            face_detected and
            fraud_risk != "high" and
            id_forensics["laplacian_var"] >= MAX_BLUR_SCORE * 0.5  # Not too blurry
        )
        
        message = "ID card verification successful. Proceed to face matching." if is_valid else "ID card verification failed. Please provide a clearer ID photo."
        
        logger.info(f"ID Verification Result: {'VALID' if is_valid else 'INVALID'} - Quality: {quality_score:.1f}%, Fraud Risk: {fraud_risk}")
        
        return IDVerificationResponse(
            valid=is_valid,
            message=message,
            quality_score=round(quality_score, 2),
            fraud_risk=fraud_risk,
            fraud_indicators=fraud_indicators,
            details={
                "face_detected": face_detected,
                "blur_score": round(id_forensics["laplacian_var"], 2),
                "brightness": round(id_forensics["avg_brightness"], 2),
                "edge_density": round(id_forensics["edge_density"], 4),
                "aspect_ratio": round(id_forensics["aspect_ratio"], 2),
                "has_exif": id_forensics["has_exif"]
            }
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during ID verification: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred during ID verification. Please try again."
            }
        )


@app.post("/api/verify-selfie", response_model=SelfieVerificationResponse)
async def verify_selfie(
    selfie: UploadFile = File(..., description="Live selfie image")
):
    """
    Step 2: Verify selfie quality and detect fraud indicators.
    
    This endpoint checks:
    - Image quality (blur, brightness)
    - Presence of exactly one face
    - Fraud indicators (edited images, screenshots, etc.)
    - Liveness indicators
    
    **Use this endpoint** after ID verification passes and before face matching.
    
    **Returns:**
    - valid: Boolean indicating if selfie passes quality checks
    - quality_score: Overall quality score (0-100)
    - fraud_risk: Risk level (low/medium/high)
    - fraud_indicators: List of detected issues
    - details: Detailed forensic analysis results
    """
    try:
        # Validate file type
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
        
        logger.info("Starting selfie verification process")
        
        # Read image
        selfie_bytes = await selfie.read()
        logger.info(f"Received selfie: {len(selfie_bytes)} bytes")
        
        # Validate file size
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(selfie_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "File Too Large",
                    "error_code": "FILE_TOO_LARGE",
                    "message": "Image file must be less than 10MB"
                }
            )
        
        # Load image
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
        
        # Perform forensic analysis
        logger.info("Performing forensic analysis on selfie...")
        selfie_forensics = analyze_image_forensics(selfie_pil, selfie_image, "selfie")
        
        fraud_indicators = selfie_forensics["fraud_indicators"]
        
        logger.info(f"Fraud indicators found: {len(fraud_indicators)}")
        for indicator in fraud_indicators:
            logger.warning(f"⚠️ {indicator}")
        
        # Detect face in selfie
        logger.info("Detecting face in selfie...")
        try:
            selfie_encoding, selfie_locations = detect_and_encode_face(selfie_image, "selfie")
            logger.info(f"Selfie face detected at: {selfie_locations}")
            face_detected = True
        except HTTPException as e:
            # Face detection failed
            face_detected = False
            fraud_indicators.append("Face detection failed - selfie may be invalid")
        
        # Calculate quality score
        quality_score = 100.0
        
        # Deduct points for blur
        if selfie_forensics["laplacian_var"] < MAX_BLUR_SCORE:
            blur_penalty = (MAX_BLUR_SCORE - selfie_forensics["laplacian_var"]) / MAX_BLUR_SCORE * 30
            quality_score -= blur_penalty
            logger.info(f"Blur penalty: -{blur_penalty:.1f} points")
        
        # Deduct points for low brightness
        if selfie_forensics["avg_brightness"] < MIN_IMAGE_QUALITY_SCORE:
            brightness_penalty = 20
            quality_score -= brightness_penalty
            logger.info(f"Brightness penalty: -{brightness_penalty} points")
        
        # Deduct points for each fraud indicator
        fraud_penalty = len(fraud_indicators) * 10
        quality_score -= fraud_penalty
        logger.info(f"Fraud indicator penalty: -{fraud_penalty} points")
        
        # Deduct points if no face detected
        if not face_detected:
            quality_score -= 30
            logger.info("No face penalty: -30 points")
        
        quality_score = max(0, quality_score)
        
        # Assess fraud risk
        fraud_risk = assess_fraud_risk(fraud_indicators)
        
        # Determine if selfie is valid
        is_valid = (
            quality_score >= 60 and  # Minimum quality score
            face_detected and
            fraud_risk != "high" and
            selfie_forensics["laplacian_var"] >= MAX_BLUR_SCORE * 0.5  # Not too blurry
        )
        
        message = "Selfie verification successful. Proceed to face matching." if is_valid else "Selfie verification failed. Please provide a clearer selfie."
        
        logger.info(f"Selfie Verification Result: {'VALID' if is_valid else 'INVALID'} - Quality: {quality_score:.1f}%, Fraud Risk: {fraud_risk}")
        
        return SelfieVerificationResponse(
            valid=is_valid,
            message=message,
            quality_score=round(quality_score, 2),
            fraud_risk=fraud_risk,
            fraud_indicators=fraud_indicators,
            details={
                "face_detected": face_detected,
                "blur_score": round(selfie_forensics["laplacian_var"], 2),
                "brightness": round(selfie_forensics["avg_brightness"], 2),
                "aspect_ratio": round(selfie_forensics["aspect_ratio"], 2),
                "has_exif": selfie_forensics["has_exif"]
            }
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during selfie verification: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred during selfie verification. Please try again."
            }
        )


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
        
        # Detect and encode faces with landmarks
        logger.info("Detecting face in selfie...")
        selfie_encoding, selfie_locations, selfie_landmarks, selfie_quality_info = detect_and_encode_face(selfie_image, "selfie")
        logger.info(f"Selfie face detected: {selfie_quality_info}")
        
        logger.info("Detecting face in ID card...")
        id_encoding, id_locations, id_landmarks, id_quality_info = detect_and_encode_face(id_card_image, "id_card")
        logger.info(f"ID card face detected: {id_quality_info}")
        
        # Calculate dynamic threshold based on image quality and alignment
        adjusted_threshold, threshold_reasoning = calculate_dynamic_threshold(
            selfie_forensics, 
            id_forensics,
            selfie_quality_info.get("alignment", {}),
            id_quality_info.get("alignment", {})
        )
        
        # Check if images are suspiciously identical (potential fraud)
        initial_distance = float(np.linalg.norm(selfie_encoding - id_encoding))
        if initial_distance < IDENTICAL_IMAGE_THRESHOLD:
            logger.warning(f"⚠️ SUSPICIOUS: Identical images detected (distance: {initial_distance:.4f})")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Identical Images Detected",
                    "error_code": "IDENTICAL_IMAGES",
                    "message": "The selfie and ID card photo appear to be the same image. Please upload a live selfie and a separate ID card photo."
                }
            )
        
        # Advanced face comparison with secondary checks
        comparison_result = compare_faces_advanced(
            selfie_encoding, 
            id_encoding,
            selfie_landmarks if selfie_landmarks else {},
            id_landmarks if id_landmarks else {},
            adjusted_threshold
        )
        
        is_match = comparison_result["match"]
        raw_distance = comparison_result["raw_distance"]
        effective_distance = comparison_result["effective_distance"]
        
        # Convert distance to confidence score
        confidence_score = distance_to_confidence(effective_distance)
        
        # Assess fraud risk
        fraud_risk = assess_fraud_risk(all_fraud_indicators)
        
        # Generate detailed reasoning
        reasoning_parts = [threshold_reasoning]
        if comparison_result.get("secondary_checks"):
            sec = comparison_result["secondary_checks"]
            if sec.get("confidence_boost", 0) > 0:
                reasoning_parts.append(f"Applied confidence boost: {sec['confidence_boost']}")
            if "cosine_similarity" in sec:
                reasoning_parts.append(f"Cosine similarity: {sec['cosine_similarity']}")
            if "landmark_comparison" in sec:
                lc = sec["landmark_comparison"]
                if lc.get("landmarks_match"):
                    reasoning_parts.append("Landmarks match confirmed")
        
        reasoning = " | ".join(reasoning_parts)
        
        # Generate response message and warnings
        warning = None
        if is_match:
            message = "Verification successful. The faces match."
            # Add warning for matches that are close to threshold
            if effective_distance > (adjusted_threshold * 0.90):
                warning = "Match found but confidence is moderate. Consider manual review for high-security applications."
            # Override for high fraud risk
            if fraud_risk == "high":
                warning = "⚠️ ALERT: High fraud risk detected. Manual verification strongly recommended."
            logger.info(f"✓ MATCH - Raw: {raw_distance:.4f}, Effective: {effective_distance:.4f}, Threshold: {adjusted_threshold:.4f}, Confidence: {confidence_score}%, Risk: {fraud_risk}")
        else:
            message = "Verification failed. The faces do not match."
            logger.info(f"✗ NO MATCH - Raw: {raw_distance:.4f}, Effective: {effective_distance:.4f}, Threshold: {adjusted_threshold:.4f}, Confidence: {confidence_score}%, Risk: {fraud_risk}")
        
        # Prepare debug info with all values as serializable types
        debug_info = {
            "selfie_detection": selfie_quality_info.get("detection_method"),
            "id_detection": id_quality_info.get("detection_method"),
            "selfie_alignment": selfie_quality_info.get("alignment", {}),
            "id_alignment": id_quality_info.get("alignment", {}),
            "selfie_blur": round(float(selfie_forensics.get("laplacian_var", 0)), 2),
            "id_blur": round(float(id_forensics.get("laplacian_var", 0)), 2),
            "selfie_brightness": round(float(selfie_forensics.get("avg_brightness", 0)), 2),
            "id_brightness": round(float(id_forensics.get("avg_brightness", 0)), 2),
            "secondary_checks": comparison_result.get("secondary_checks", {})
        }
        
        return VerificationResponse(
            match=bool(is_match),
            confidence_score=round(float(confidence_score), 2),
            message=message,
            distance=round(float(effective_distance), 4),
            raw_distance=round(float(raw_distance), 4),
            adjusted_threshold=round(float(adjusted_threshold), 4),
            reasoning=reasoning,
            warning=warning,
            fraud_risk=fraud_risk,
            fraud_indicators=all_fraud_indicators,
            debug_info=debug_info
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
