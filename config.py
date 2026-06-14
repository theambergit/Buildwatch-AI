"""
BuildWatch AI - Configuration Module
Manages application settings and environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    ENV = os.getenv("FLASK_ENV", "production")
    
    # File Upload
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    
    # Image Processing
    THRESHOLD_VALUE = int(os.getenv("THRESHOLD_VALUE", 60))
    MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", 3000))
    YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", 0.5))
    RISK_HIGH_THRESHOLD = int(os.getenv("RISK_HIGH_THRESHOLD", 30))
    RISK_MEDIUM_THRESHOLD = int(os.getenv("RISK_MEDIUM_THRESHOLD", 10))
    GAUSSIAN_BLUR_KERNEL = (
        int(os.getenv("GAUSSIAN_BLUR_KERNEL_SIZE", 9)),
        int(os.getenv("GAUSSIAN_BLUR_KERNEL_SIZE", 9))
    )
    MORPHOLOGY_KERNEL_SIZE = (
        int(os.getenv("MORPHOLOGY_KERNEL_SIZE", 5)),
        int(os.getenv("MORPHOLOGY_KERNEL_SIZE", 5))
    )
    MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", 640))
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 60))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "buildwatch.log")
    
    # Model
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.onnx")
    USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENV = "development"
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ENV = "production"
    LOG_LEVEL = "WARNING"


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    ENV = "testing"


def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv("FLASK_ENV", "production")
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    return config_map.get(env, ProductionConfig)
