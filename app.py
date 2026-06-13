import logging
import os
import json

MPLCONFIGDIR = os.path.join("/tmp", "buildwatch-matplotlib")
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

# Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static")
ANALYSIS_FOLDER = os.path.join(UPLOAD_FOLDER, "analyses")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Image processing constants
THRESHOLD_VALUE = 60
MIN_CONTOUR_AREA = 3000
YOLO_CONFIDENCE_THRESHOLD = 0.5
RISK_HIGH_THRESHOLD = 30
RISK_MEDIUM_THRESHOLD = 10
GAUSSIAN_BLUR_KERNEL = (9, 9)
MORPHOLOGY_KERNEL_SIZE = (5, 5)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if not present
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None


def allowed_file(filename):
    """Check if file has allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image(img, filename):
    """Validate image format and dimensions."""
    if img is None:
        logger.error(f"Failed to read image: {filename}")
        return False, "Failed to read image"
    
    if len(img.shape) < 2:
        logger.error(f"Invalid image format: {filename}")
        return False, "Invalid image format"
    
    if len(img.shape) == 3 and img.shape[2] != 3:
        logger.error(f"Image must be RGB/BGR: {filename}")
        return False, "Image must be RGB/BGR"
    
    return True, "Valid"


def process_images(before_path, after_path):
    """Process before and after images."""
    try:
        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)
        
        # Validate images
        is_valid, msg = validate_image(before_img, before_path)
        if not is_valid:
            return None, None, msg
        
        is_valid, msg = validate_image(after_img, after_path)
        if not is_valid:
            return None, None, msg
        
        # Resize images to same dimensions (use max to preserve detail)
        height = max(before_img.shape[0], after_img.shape[0])
        width = max(before_img.shape[1], after_img.shape[1])
        
        before_img = cv2.resize(before_img, (width, height))
        after_img = cv2.resize(after_img, (width, height))
        
        logger.info(f"Images processed: {width}x{height}")
        return before_img, after_img, "Success"
    
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        return None, None, f"Image processing error: {str(e)}"


def detect_changes(before_img, after_img):
    """Detect changes between before and after images."""
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
        
        # Reduce noise
        gray1 = cv2.GaussianBlur(gray1, GAUSSIAN_BLUR_KERNEL, 0)
        gray2 = cv2.GaussianBlur(gray2, GAUSSIAN_BLUR_KERNEL, 0)
        
        # Calculate difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold
        _, thresh = cv2.threshold(
            diff,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY
        )
        
        # Morphological operations
        kernel = np.ones(MORPHOLOGY_KERNEL_SIZE, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        logger.info("Change detection completed successfully")
        return thresh
    
    except Exception as e:
        logger.error(f"Error detecting changes: {str(e)}")
        return None


def save_output_images(thresh, after_img, before_img, clean_after_img, analysis_id):
    """Save mask, heatmap, and result images."""
    try:
        output_dir = os.path.join(ANALYSIS_FOLDER, analysis_id)
        os.makedirs(output_dir, exist_ok=True)

        # Save mask
        cv2.imwrite(os.path.join(output_dir, "mask.jpg"), thresh)
        
        # Create and save heatmap
        heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, "heatmap.jpg"), heatmap)
        
        # Save result image
        result_path = os.path.join(output_dir, "result.jpg")
        cv2.imwrite(result_path, after_img)

        # Save comparison images
        cv2.imwrite(os.path.join(output_dir, "before.jpg"), before_img)
        cv2.imwrite(os.path.join(output_dir, "after.jpg"), clean_after_img)
        
        logger.info("Output images saved successfully")
        public_base = f"/static/analyses/{analysis_id}"
        return {
            "result": f"{public_base}/result.jpg",
            "mask": f"{public_base}/mask.jpg",
            "heatmap": f"{public_base}/heatmap.jpg",
            "before": f"{public_base}/before.jpg",
            "after": f"{public_base}/after.jpg"
        }
    
    except Exception as e:
        logger.error(f"Error saving output images: {str(e)}")
        return None


def build_report_summary(confidence_score, detected_regions, risk, yolo_count):
    """Create a short result summary for the UI and API."""
    if detected_regions == 0:
        return "No significant construction changes were detected. Continue routine monitoring for this location."

    if risk == "High":
        return (
            f"{detected_regions} changed region(s) were detected with {confidence_score}% scene change. "
            "This should be prioritized for manual review and regulatory verification."
        )

    if risk == "Medium":
        return (
            f"{detected_regions} changed region(s) were detected with {confidence_score}% scene change. "
            "Review the highlighted zones to confirm whether the activity matches expected progress."
        )

    return (
        f"{detected_regions} changed region(s) were detected with limited scene impact. "
        f"YOLO identified {yolo_count} object(s) in the current image for added context."
    )


def detect_contours(thresh, after_img):
    """Detect contours and mark regions of change."""
    try:
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detected_regions = 0
        regions_data = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Draw rectangle
                cv2.rectangle(
                    after_img,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    3
                )
                
                # Add text label
                cv2.putText(
                    after_img,
                    "Change Detected",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
                
                regions_data.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": float(area)
                })
                
                detected_regions += 1
        
        logger.info(f"Detected {detected_regions} regions")
        return detected_regions, regions_data
    
    except Exception as e:
        logger.error(f"Error detecting contours: {str(e)}")
        return 0, []


def yolo_detection(after_path, after_img):
    """Perform YOLO object detection."""
    detected_objects = []
    
    if model is None:
        logger.warning("YOLO model not available, skipping detection")
        return detected_objects
    
    try:
        results = model(after_path)
        
        for r in results:
            boxes = r.boxes
            
            if boxes is None:
                continue
            
            for box in boxes:
                confidence = float(box.conf[0])
                
                if confidence < YOLO_CONFIDENCE_THRESHOLD:
                    continue
                
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]
                
                # Draw rectangle
                cv2.rectangle(
                    after_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                
                # Add text label
                cv2.putText(
                    after_img,
                    f"{label} {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                detected_objects.append({
                    "label": label,
                    "confidence": float(confidence),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                })
        
        logger.info(f"YOLO detection found {len(detected_objects)} objects")
    
    except Exception as e:
        logger.error(f"YOLO Detection Error: {str(e)}")
    
    return detected_objects


def calculate_risk_level(confidence_score):
    """Calculate risk level based on confidence score."""
    if confidence_score > RISK_HIGH_THRESHOLD:
        return "High"
    elif confidence_score > RISK_MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def calculate_confidence(thresh):
    """Calculate confidence score based on changed pixels."""
    try:
        change_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        
        confidence_score = round(
            (change_pixels / total_pixels) * 100,
            2
        )
        return confidence_score
    
    except Exception as e:
        logger.error(f"Error calculating confidence: {str(e)}")
        return 0.0


@app.route("/")
def home():
    """Render home page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload and processing."""
    logger.info("Processing upload request")
    
    try:
        # Validate files exist
        if "before" not in request.files or "after" not in request.files:
            logger.warning("Missing file in request")
            return render_template("index.html", error="Please upload both before and after images."), 400
        
        before_file = request.files["before"]
        after_file = request.files["after"]
        
        # Validate filenames
        if before_file.filename == "" or after_file.filename == "":
            logger.warning("Empty filename in request")
            return render_template("index.html", error="Please choose both image files before analyzing."), 400
        
        if not (allowed_file(before_file.filename) and allowed_file(after_file.filename)):
            logger.warning("Invalid file type in request")
            return render_template("index.html", error="Only JPG, JPEG, and PNG images are supported."), 400
        
        # Save uploaded files with secure filenames
        analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(3).hex()
        before_ext = before_file.filename.rsplit(".", 1)[1].lower()
        after_ext = after_file.filename.rsplit(".", 1)[1].lower()
        before_filename = secure_filename(f"before_{analysis_id}.{before_ext}")
        after_filename = secure_filename(f"after_{analysis_id}.{after_ext}")
        
        before_path = os.path.join(UPLOAD_FOLDER, before_filename)
        after_path = os.path.join(UPLOAD_FOLDER, after_filename)
        
        before_file.save(before_path)
        after_file.save(after_path)
        
        logger.info(f"Files saved: {before_filename}, {after_filename}")
        
        # Process images
        before_img, after_img, msg = process_images(before_path, after_path)
        if before_img is None:
            logger.error(f"Image processing failed: {msg}")
            return render_template("index.html", error=msg), 400
        
        # Make a copy for drawing
        after_img_annotated = after_img.copy()
        
        # Detect changes
        thresh = detect_changes(before_img, after_img)
        if thresh is None:
            logger.error("Change detection failed")
            return render_template("index.html", error="Change detection failed. Please try another image pair."), 500
        
        # Detect contours and get regions data
        detected_regions, regions_data = detect_contours(thresh, after_img_annotated)
        
        # YOLO Detection
        yolo_objects = yolo_detection(after_path, after_img_annotated)
        
        # Calculate confidence score
        confidence_score = calculate_confidence(thresh)
        
        # Calculate risk level
        risk = calculate_risk_level(confidence_score)
        
        # Save output images
        output_images = save_output_images(thresh, after_img_annotated, before_img, after_img, analysis_id)
        if output_images is None:
            logger.error("Failed to save output images")
            return render_template("index.html", error="Could not save analysis results. Please try again."), 500
        
        logger.info(f"Upload processed successfully: confidence={confidence_score}, risk={risk}, regions={detected_regions}")
        
        # Clean up temporary files
        try:
            os.remove(before_path)
            os.remove(after_path)
        except Exception as e:
            logger.warning(f"Could not clean up temporary files: {str(e)}")
        
        return render_template(
            "result.html",
            confidence=confidence_score,
            regions=detected_regions,
            risk=risk,
            images=output_images,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_pixels=before_img.shape[0] * before_img.shape[1],
            image_dimensions=f"{before_img.shape[1]}x{before_img.shape[0]}",
            yolo_detections=len(yolo_objects),
            changed_pixels=int(cv2.countNonZero(thresh)),
            summary=build_report_summary(confidence_score, detected_regions, risk, len(yolo_objects))
        )
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return render_template("index.html", error="Something went wrong during analysis. Please try again."), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """API endpoint for image upload and processing - returns JSON."""
    logger.info("Processing API upload request")
    
    try:
        # Validate files exist
        if "before" not in request.files or "after" not in request.files:
            logger.warning("Missing file in API request")
            return jsonify({"success": False, "error": "Missing before or after file"}), 400
        
        before_file = request.files["before"]
        after_file = request.files["after"]
        
        # Validate filenames
        if before_file.filename == "" or after_file.filename == "":
            logger.warning("Empty filename in API request")
            return jsonify({"success": False, "error": "No selected files"}), 400
        
        if not (allowed_file(before_file.filename) and allowed_file(after_file.filename)):
            logger.warning("Invalid file type in API request")
            return jsonify({"success": False, "error": "Invalid file type. Allowed: jpg, jpeg, png"}), 400
        
        # Save uploaded files
        analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(3).hex()
        before_ext = before_file.filename.rsplit(".", 1)[1].lower()
        after_ext = after_file.filename.rsplit(".", 1)[1].lower()
        before_filename = secure_filename(f"before_{analysis_id}.{before_ext}")
        after_filename = secure_filename(f"after_{analysis_id}.{after_ext}")
        
        before_path = os.path.join(UPLOAD_FOLDER, before_filename)
        after_path = os.path.join(UPLOAD_FOLDER, after_filename)
        
        before_file.save(before_path)
        after_file.save(after_path)
        
        logger.info(f"API Files saved: {before_filename}, {after_filename}")
        
        # Process images
        before_img, after_img, msg = process_images(before_path, after_path)
        if before_img is None:
            logger.error(f"Image processing failed: {msg}")
            return jsonify({"success": False, "error": msg}), 400
        
        after_img_annotated = after_img.copy()
        
        # Detect changes
        thresh = detect_changes(before_img, after_img)
        if thresh is None:
            logger.error("Change detection failed")
            return jsonify({"success": False, "error": "Change detection failed"}), 500
        
        # Get regions and data
        detected_regions, regions_data = detect_contours(thresh, after_img_annotated)
        
        # YOLO Detection
        yolo_objects = yolo_detection(after_path, after_img_annotated)
        
        # Calculate metrics
        confidence_score = calculate_confidence(thresh)
        risk = calculate_risk_level(confidence_score)
        
        # Save output images
        output_images = save_output_images(thresh, after_img_annotated, before_img, after_img, analysis_id)
        if output_images is None:
            logger.error("Failed to save output images")
            return jsonify({"success": False, "error": "Failed to save results"}), 500
        
        logger.info(f"API Analysis completed successfully")
        
        # Clean up
        try:
            os.remove(before_path)
            os.remove(after_path)
        except Exception as e:
            logger.warning(f"Could not clean up temporary files: {str(e)}")
        
        # Return comprehensive JSON response
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "confidence_score": confidence_score,
                "risk_level": risk,
                "regions_detected": detected_regions,
                "regions_data": regions_data,
                "yolo_detections": yolo_objects,
                "yolo_count": len(yolo_objects),
                "summary": build_report_summary(confidence_score, detected_regions, risk, len(yolo_objects))
            },
            "image_info": {
                "width": before_img.shape[1],
                "height": before_img.shape[0],
                "total_pixels": before_img.shape[0] * before_img.shape[1],
                "changed_pixels": int(cv2.countNonZero(thresh))
            },
            "output_images": output_images
        }), 200
    
    except Exception as e:
        logger.error(f"API Upload error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    logger.warning("File too large uploaded")
    return "File too large. Maximum size: 50MB", 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error."""
    logger.error(f"Internal server error: {str(error)}")
    return "Internal server error", 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "BuildWatch AI",
        "yolo_available": model is not None
    }), 200


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
