import logging
import os
import time
import json
import gc
from pathlib import Path
from uuid import uuid4

# Suppress matplotlib font cache (not used, but some deps reference it)
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import onnxruntime as ort
import cv2
import numpy as np
from datetime import datetime
from config import get_config

# Configuration
BASE_DIR = Path(__file__).resolve().parent
Config = get_config()
UPLOAD_FOLDER = Path(Config.UPLOAD_FOLDER)
if not UPLOAD_FOLDER.is_absolute():
    UPLOAD_FOLDER = BASE_DIR / UPLOAD_FOLDER
ANALYSIS_FOLDER = UPLOAD_FOLDER / "analyses"
MAX_FILE_SIZE = Config.MAX_FILE_SIZE
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS
MODEL_PATH = Path(Config.YOLO_MODEL)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH

# Image processing constants
THRESHOLD_VALUE = Config.THRESHOLD_VALUE
MIN_CONTOUR_AREA = Config.MIN_CONTOUR_AREA
YOLO_CONFIDENCE_THRESHOLD = Config.YOLO_CONFIDENCE_THRESHOLD
RISK_HIGH_THRESHOLD = Config.RISK_HIGH_THRESHOLD
RISK_MEDIUM_THRESHOLD = Config.RISK_MEDIUM_THRESHOLD
GAUSSIAN_BLUR_KERNEL = Config.GAUSSIAN_BLUR_KERNEL
MORPHOLOGY_KERNEL_SIZE = Config.MORPHOLOGY_KERNEL_SIZE

# Logging setup
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year cache for static files

# Create upload folder if not present
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ANALYSIS_FOLDER.mkdir(parents=True, exist_ok=True)

# COCO class names for YOLOv8
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# Lazy-loaded ONNX session — only allocates RAM when first analysis runs
_ort_session = None
_input_name = None

def _get_ort_session():
    """Lazy-load ONNX model on first use to minimize idle RAM."""
    global _ort_session, _input_name
    if _ort_session is None:
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1  # Minimize thread memory
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _ort_session = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            _input_name = _ort_session.get_inputs()[0].name
            logger.info("YOLO ONNX model loaded (lazy)")
        except Exception as e:
            logger.error(f"Failed to load YOLO ONNX model: {str(e)}")
    return _ort_session, _input_name


def allowed_file(filename):
    """Check if file has allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def new_analysis_id():
    """Create a unique id for uploaded files and generated report assets."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def public_analysis_url(analysis_id, filename):
    """Return the public URL for generated static analysis assets."""
    return f"/static/analyses/{analysis_id}/{filename}"


def save_cv_image(path, image):
    """Save an OpenCV image and raise a clear error if writing fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY] if str(path).lower().endswith(('.jpg', '.jpeg')) else []
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"Could not write image: {path}")


def cleanup_paths(paths):
    """Best-effort cleanup for temporary upload files."""
    for path in paths:
        try:
            if path and path.exists():
                path.unlink()
        except Exception as e:
            logger.warning(f"Could not clean up temporary file {path}: {str(e)}")


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
        before_img = cv2.imread(str(before_path))
        after_img = cv2.imread(str(after_path))
        
        # Validate images
        is_valid, msg = validate_image(before_img, before_path)
        if not is_valid:
            return None, None, msg
        
        is_valid, msg = validate_image(after_img, after_path)
        if not is_valid:
            return None, None, msg
        
        # Resize images to same dimensions, capped at MAX_IMAGE_DIMENSION
        height = min(before_img.shape[0], after_img.shape[0])
        width = min(before_img.shape[1], after_img.shape[1])
        
        # Cap to maximum dimension for performance
        max_dim = Config.MAX_IMAGE_DIMENSION
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            height = int(height * scale)
            width = int(width * scale)
        
        before_img = cv2.resize(before_img, (width, height), interpolation=cv2.INTER_AREA)
        after_img = cv2.resize(after_img, (width, height), interpolation=cv2.INTER_AREA)
        
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
        del gray1, gray2  # Free memory immediately
        
        # Threshold
        _, thresh = cv2.threshold(
            diff,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY
        )
        del diff  # Free diff immediately
        
        # Morphological operations
        kernel = np.ones(MORPHOLOGY_KERNEL_SIZE, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        del kernel
        
        logger.info("Change detection completed successfully")
        return thresh
    
    except Exception as e:
        logger.error(f"Error detecting changes: {str(e)}")
        return None


def save_output_images(thresh, after_img, before_img, after_img_clean, analysis_id):
    """Save mask, heatmap, and result images. Frees memory as it goes."""
    try:
        output_dir = ANALYSIS_FOLDER / analysis_id

        # Save mask
        save_cv_image(output_dir / "mask.jpg", thresh)
        
        # Create heatmap, save, and immediately free
        heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
        save_cv_image(output_dir / "heatmap.jpg", heatmap)
        del heatmap  # Free ~2MB
        
        # Save result (annotated) image
        save_cv_image(output_dir / "result.jpg", after_img)

        # Save before and after originals
        save_cv_image(output_dir / "before.jpg", before_img)
        save_cv_image(output_dir / "after.jpg", after_img_clean)
        
        logger.info("Output images saved successfully")
        return {
            "result": public_analysis_url(analysis_id, "result.jpg"),
            "mask": public_analysis_url(analysis_id, "mask.jpg"),
            "heatmap": public_analysis_url(analysis_id, "heatmap.jpg"),
            "before": public_analysis_url(analysis_id, "before.jpg"),
            "after": public_analysis_url(analysis_id, "after.jpg")
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


def _yolo_preprocess(img, input_size=640):
    """Preprocess image for YOLOv8 ONNX: letterbox resize, normalize, NCHW."""
    h, w = img.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Create letterboxed canvas
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    top, left = (input_size - nh) // 2, (input_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized

    # BGR→RGB, HWC→CHW, normalize 0-1, add batch dim
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob, scale, top, left


def _yolo_nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS (non-maximum suppression) using numpy."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def yolo_detection(after_path, after_img):
    """Perform YOLO object detection using raw ONNX Runtime (no torch)."""
    detected_objects = []

    session, input_name = _get_ort_session()
    if session is None:
        logger.warning("YOLO ONNX model not available, skipping detection")
        return detected_objects

    try:
        h_orig, w_orig = after_img.shape[:2]
        blob, scale, pad_top, pad_left = _yolo_preprocess(after_img)

        # Run inference
        outputs = session.run(None, {input_name: blob})
        preds = outputs[0]  # shape: (1, 84, 8400) for YOLOv8

        # Transpose to (8400, 84): each row = [cx, cy, w, h, class_scores...]
        preds = preds[0].T

        # Extract boxes and class scores
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        class_scores = preds[:, 4:]  # 80 COCO classes

        # Best class per detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask = confidences >= YOLO_CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return detected_objects

        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert center-wh to xyxy
        x1 = cx - bw / 2; y1 = cy - bh / 2
        x2 = cx + bw / 2; y2 = cy + bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = _yolo_nms(boxes, confidences)
        boxes = boxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # Scale boxes back to original image coordinates
        for i in range(len(boxes)):
            bx1 = int((boxes[i][0] - pad_left) / scale)
            by1 = int((boxes[i][1] - pad_top) / scale)
            bx2 = int((boxes[i][2] - pad_left) / scale)
            by2 = int((boxes[i][3] - pad_top) / scale)

            # Clamp to image bounds
            bx1 = max(0, min(bx1, w_orig - 1))
            by1 = max(0, min(by1, h_orig - 1))
            bx2 = max(0, min(bx2, w_orig - 1))
            by2 = max(0, min(by2, h_orig - 1))

            conf = float(confidences[i])
            cls_id = int(class_ids[i])
            label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}"

            # Draw rectangle
            cv2.rectangle(after_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            # Add text label
            cv2.putText(
                after_img,
                f"{label} {conf:.2f}",
                (bx1, max(by1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detected_objects.append({
                "label": label,
                "confidence": round(conf, 3),
                "x1": bx1, "y1": by1,
                "x2": bx2, "y2": by2
            })

        # Free inference memory
        del blob, preds, outputs

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


def analyze_image_pair(before_path, after_path, analysis_id=None):
    """Run the full analysis pipeline and return template/API-ready data.
    
    Memory-optimized: deletes arrays as soon as they're no longer needed
    to stay well under 512MB on Render free tier.
    """
    t_start = time.perf_counter()
    analysis_id = analysis_id or new_analysis_id()

    before_img, after_img, msg = process_images(before_path, after_path)
    if before_img is None:
        raise ValueError(msg)

    # Store dimensions before any processing
    img_h, img_w = before_img.shape[0], before_img.shape[1]
    total_pixels = img_h * img_w

    # Step 1: Change detection (grayscale diff → thresh)
    thresh = detect_changes(before_img, after_img)
    if thresh is None:
        raise RuntimeError("Change detection failed. Please try another image pair.")

    changed_pixels = int(cv2.countNonZero(thresh))
    confidence_score = round((changed_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
    risk = calculate_risk_level(confidence_score)

    # Step 2: Contour detection (draws on after_img directly — no .copy())
    detected_regions, regions_data = detect_contours(thresh, after_img)

    # Step 3: YOLO detection (draws on after_img directly)
    yolo_objects = yolo_detection(after_path, after_img)

    # Step 4: Save outputs — frees heatmap immediately inside
    output_images = save_output_images(
        thresh,
        after_img,       # now has contours + YOLO annotations
        before_img,
        after_img,       # same ref, saved as "after" too
        analysis_id
    )

    # Free the big arrays NOW
    del before_img, after_img, thresh
    gc.collect()

    if output_images is None:
        raise RuntimeError("Could not save analysis results. Please try again.")

    summary = build_report_summary(
        confidence_score,
        detected_regions,
        risk,
        len(yolo_objects)
    )

    processing_time_ms = round((time.perf_counter() - t_start) * 1000)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    res = {
        "analysis_id": analysis_id,
        "timestamp": timestamp_str,
        "confidence_score": confidence_score,
        "risk": risk,
        "detected_regions": detected_regions,
        "regions_data": regions_data,
        "yolo_objects": yolo_objects,
        "output_images": output_images,
        "summary": summary,
        "width": img_w,
        "height": img_h,
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "processing_time_ms": processing_time_ms
    }
    
    # Save to summary.json
    summary_path = BASE_DIR / "static" / "analyses" / analysis_id / "summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(res, f)
    except Exception as e:
        logger.error(f"Failed to save summary JSON: {e}")

    return res


def result_template_response(result):
    """Render the analysis result page."""
    return render_template(
        "result.html",
        confidence=result["confidence_score"],
        regions=result["detected_regions"],
        regions_data=result.get("regions_data", []),
        width=result.get("width", 1200),
        height=result.get("height", 800),
        risk=result["risk"],
        images=result["output_images"],
        timestamp=result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        total_pixels=result["total_pixels"],
        image_dimensions=f"{result['width']}x{result['height']}",
        yolo_detections=len(result["yolo_objects"]),
        yolo_objects=result["yolo_objects"],
        changed_pixels=result["changed_pixels"],
        summary=result["summary"],
        analysis_id=result["analysis_id"]
    )


def result_json_response(result):
    """Build the JSON API payload."""
    response = jsonify({
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "confidence_score": result["confidence_score"],
            "risk_level": result["risk"],
            "regions_detected": result["detected_regions"],
            "regions_data": result["regions_data"],
            "yolo_detections": result["yolo_objects"],
            "yolo_count": len(result["yolo_objects"]),
            "summary": result["summary"]
        },
        "image_info": {
            "width": result["width"],
            "height": result["height"],
            "total_pixels": result["total_pixels"],
            "changed_pixels": result["changed_pixels"]
        },
        "output_images": result["output_images"],
        "processing_time_ms": result["processing_time_ms"]
    })
    response.headers["X-Processing-Time"] = str(result["processing_time_ms"])
    return response


@app.route("/")
def home():
    """Render home page."""
    return render_template("index.html")


@app.route("/history")
def history():
    """List all previous analyses."""
    analyses_dir = BASE_DIR / "static" / "analyses"
    analyses = []
    
    if analyses_dir.exists():
        for d in analyses_dir.iterdir():
            if d.is_dir():
                summary_path = d / "summary.json"
                if summary_path.exists():
                    try:
                        with open(summary_path, "r") as f:
                            data = json.load(f)
                            analyses.append(data)
                    except Exception as e:
                        logger.error(f"Failed to read {summary_path}: {e}")
                        
    # Sort by timestamp descending
    analyses.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return render_template("history.html", analyses=analyses)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear all past analyses."""
    import shutil
    analyses_dir = BASE_DIR / "static" / "analyses"
    if analyses_dir.exists():
        for d in analyses_dir.iterdir():
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    logger.error(f"Failed to delete {d}: {e}")
    return redirect(url_for("history"))


@app.route("/analysis/<analysis_id>")
def view_analysis(analysis_id):
    """View a past analysis."""
    summary_path = BASE_DIR / "static" / "analyses" / secure_filename(analysis_id) / "summary.json"
    if not summary_path.exists():
        return render_template("404.html"), 404
    
    try:
        with open(summary_path, "r") as f:
            result = json.load(f)
        return result_template_response(result)
    except Exception as e:
        logger.error(f"Failed to read analysis {analysis_id}: {e}")
        return render_template("500.html"), 500


def upload_error_response(message):
    """Return to the upload screen with an error that client-side JS can show."""
    return redirect(url_for("home", error=message))





@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload and processing."""
    logger.info("Processing upload request")
    before_path = None
    after_path = None
    
    try:
        # Validate files exist
        if "before" not in request.files or "after" not in request.files:
            logger.warning("Missing file in request")
            return upload_error_response("Please upload both before and after images.")
        
        before_file = request.files["before"]
        after_file = request.files["after"]
        
        # Validate filenames
        if before_file.filename == "" or after_file.filename == "":
            logger.warning("Empty filename in request")
            return upload_error_response("Please choose both image files before analyzing.")
        
        if not (allowed_file(before_file.filename) and allowed_file(after_file.filename)):
            logger.warning("Invalid file type in request")
            return upload_error_response("Only JPG, JPEG, and PNG images are supported.")
        
        # Save uploaded files with secure filenames
        analysis_id = new_analysis_id()
        before_ext = before_file.filename.rsplit(".", 1)[1].lower()
        after_ext = after_file.filename.rsplit(".", 1)[1].lower()
        before_filename = secure_filename(f"before_{analysis_id}.{before_ext}")
        after_filename = secure_filename(f"after_{analysis_id}.{after_ext}")
        
        before_path = UPLOAD_FOLDER / before_filename
        after_path = UPLOAD_FOLDER / after_filename
        
        before_file.save(before_path)
        after_file.save(after_path)
        
        logger.info(f"Files saved: {before_filename}, {after_filename}")

        result = analyze_image_pair(before_path, after_path, analysis_id)
        logger.info(
            "Upload processed successfully: confidence=%s, risk=%s, regions=%s",
            result["confidence_score"],
            result["risk"],
            result["detected_regions"]
        )

        return result_template_response(result)
    except ValueError as e:
        logger.error(f"Upload validation error: {str(e)}")
        return upload_error_response(str(e))
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return upload_error_response("Something went wrong during analysis. Please try again.")
    finally:
        cleanup_paths([before_path, after_path])


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """API endpoint for image upload and processing - returns JSON."""
    logger.info("Processing API upload request")
    before_path = None
    after_path = None
    
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
        analysis_id = new_analysis_id()
        before_ext = before_file.filename.rsplit(".", 1)[1].lower()
        after_ext = after_file.filename.rsplit(".", 1)[1].lower()
        before_filename = secure_filename(f"before_{analysis_id}.{before_ext}")
        after_filename = secure_filename(f"after_{analysis_id}.{after_ext}")
        
        before_path = UPLOAD_FOLDER / before_filename
        after_path = UPLOAD_FOLDER / after_filename
        
        before_file.save(before_path)
        after_file.save(after_path)
        
        logger.info(f"API Files saved: {before_filename}, {after_filename}")

        result = analyze_image_pair(before_path, after_path, analysis_id)
        logger.info(f"API Analysis completed successfully")
        return result_json_response(result), 200
    except ValueError as e:
        logger.error(f"API upload validation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"API Upload error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        cleanup_paths([before_path, after_path])





@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors with a styled page."""
    return render_template("404.html"), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    logger.warning("File too large uploaded")
    return "File too large. Maximum size: 50MB", 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error with a styled page."""
    logger.error(f"Internal server error: {str(error)}")
    return render_template("500.html"), 500


@app.route("/favicon.ico")
def favicon():
    """Serve favicon to prevent console 404 errors."""
    return "", 204


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "BuildWatch AI",
        "yolo_available": model is not None
    }), 200


if __name__ == "__main__":
    host = Config.HOST
    port = Config.PORT
    debug = Config.DEBUG
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
