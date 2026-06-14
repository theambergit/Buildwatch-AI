# 🏗️ BuildWatch AI - Intelligent Construction Monitoring Platform

[![GitHub](https://img.shields.io/badge/GitHub-theambergit-blue)](https://github.com/theambergit/Buildwatch-AI)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Latest-red)](https://flask.palletsprojects.com/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-yellow)](https://github.com/ultralytics/ultralytics)

## 📋 Overview

**BuildWatch AI** is an intelligent construction site monitoring platform that uses satellite/drone imagery and AI-powered computer vision to detect unauthorized construction, track progress, and assess changes in real-time.

### 🎯 Key Features

✅ **Image Comparison** - Compare before/after satellite or drone images  
✅ **Change Detection** - Intelligent pixel-level difference analysis  
✅ **AI Object Detection** - YOLO-powered construction element identification  
✅ **Risk Assessment** - Automatic risk level calculation (High/Medium/Low)  
✅ **Heatmap Generation** - Visual intensity mapping of changes  
✅ **Interactive Dashboard** - Beautiful, responsive results interface  
✅ **Before/After Slider** - Smooth image comparison tool  
✅ **JSON API** - RESTful endpoints for integration  
✅ **Batch Processing** - Process multiple image pairs  
✅ **Drag & Drop Upload** - Modern, intuitive file upload  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- 2GB+ RAM (for YOLO model)
- 1GB+ disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/theambergit/Buildwatch-AI.git
cd Buildwatch-AI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python3 app.py
```

Visit `http://localhost:5000` in your browser.

If you are using the included virtual environment on macOS/Linux:

```bash
source .venv/bin/activate
python app.py
```

You can also run it directly without activation:

```bash
.venv/bin/python app.py
```

### Quick Verification

Run the smoke test:

```bash
.venv/bin/python tests/smoke_test.py
```

Then open `http://localhost:5000/demo` to run the bundled sample images without uploading anything.

---

## 📸 Usage

### Web Interface

1. Open the application in your browser
2. Upload two images:
   - **Historical Image (Before)** - Previous state
   - **Current Image (After)** - Current state
3. Click **"Launch Analysis"**
4. View results with:
   - Confidence score
   - Detected regions count
   - Risk level assessment
   - Interactive before/after slider
   - Detailed analysis images (mask, heatmap, detections)

### API Usage

#### Upload and Analyze (JSON Response)

```bash
curl -X POST http://localhost:5000/api/upload \
  -F "before=@before.jpg" \
  -F "after=@after.jpg"
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:45.123456",
  "analysis": {
    "confidence_score": 42.5,
    "risk_level": "High",
    "regions_detected": 3,
    "regions_data": [
      {
        "x": 100,
        "y": 150,
        "width": 200,
        "height": 180,
        "area": 36000
      }
    ],
    "yolo_detections": [
      {
        "label": "truck",
        "confidence": 0.92,
        "x1": 50,
        "y1": 60,
        "x2": 250,
        "y2": 320
      }
    ]
  },
  "image_info": {
    "width": 1920,
    "height": 1440,
    "total_pixels": 2764800,
    "changed_pixels": 125000
  },
  "output_images": {
    "result": "/static/result.jpg",
    "mask": "/static/mask.jpg",
    "heatmap": "/static/heatmap.jpg",
    "before": "/static/before.jpg",
    "after": "/static/after.jpg"
  }
}
```

#### Health Check

```bash
curl http://localhost:5000/health
```

---

## 🏗️ Architecture

### Processing Pipeline

```
Upload Images
    ↓
Image Validation & Resizing
    ↓
Grayscale Conversion
    ↓
Gaussian Blur (Noise Reduction)
    ↓
Absolute Difference Calculation
    ↓
Binary Thresholding
    ↓
Morphological Operations
    ↓
Contour Detection & Region Analysis
    ↓
YOLO Object Detection
    ↓
Risk Assessment & Heatmap Generation
    ↓
Results Visualization
```

### File Structure

```
Buildwatch-AI/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── static/
│   ├── style.css              # Modern CSS with animations
│   └── result.jpg             # Generated output images
├── templates/
│   ├── index.html             # Upload interface
│   └── result.html            # Results dashboard
├── IMPROVEMENT_ROADMAP.md     # Future enhancements
└── README.md                  # This file
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0           # Server host
PORT=5000              # Server port
FLASK_DEBUG=False      # Debug mode

# Storage
UPLOAD_FOLDER=static   # Upload directory

# Image Processing
THRESHOLD_VALUE=60
MIN_CONTOUR_AREA=3000
YOLO_CONFIDENCE_THRESHOLD=0.5
RISK_HIGH_THRESHOLD=30
RISK_MEDIUM_THRESHOLD=10
```

### Customization

Edit these constants in `app.py` to adjust behavior:

```python
# Image processing constants
THRESHOLD_VALUE = 60              # Binary threshold
MIN_CONTOUR_AREA = 3000           # Minimum region size
YOLO_CONFIDENCE_THRESHOLD = 0.5   # Object detection threshold
RISK_HIGH_THRESHOLD = 30          # High risk % threshold
RISK_MEDIUM_THRESHOLD = 10        # Medium risk % threshold
```

---

## 📊 Analysis Results

### Metrics Explained

- **Confidence Score** - Percentage of changed pixels (0-100%)
- **Regions Detected** - Number of significant change areas
- **Risk Level** - High/Medium/Low based on confidence score
- **Changed Pixels** - Absolute count of different pixels

### Output Files

1. **result.jpg** - Image with detected changes highlighted
2. **mask.jpg** - Binary mask of changed areas
3. **heatmap.jpg** - Color-coded intensity map
4. **before.jpg** - Original historical image
5. **after.jpg** - Current image

---

## 🎨 UI/UX Features

### Modern Design
- Glassmorphic cards with backdrop blur
- Gradient animations and smooth transitions
- Dark theme with cyan/blue accents
- Responsive grid layouts

### Interactive Elements
- **Drag & Drop Upload** - Intuitive file selection
- **Before/After Slider** - Smooth image comparison
- **Animated Metrics** - Staggered entrance animations
- **Loading States** - Visual feedback during processing
- **Error Messages** - Clear, actionable error notifications

### Mobile Responsive
- Adapts to all screen sizes (mobile, tablet, desktop)
- Touch-optimized buttons and sliders
- Flexible grid layouts

---

## 🔧 API Endpoints

### Web Interface
- `GET /` - Home page
- `POST /upload` - Process images (returns HTML)

### REST API
- `POST /api/upload` - Process images (returns JSON)
- `GET /health` - Health check

### Status Codes
- `200 OK` - Successful processing
- `400 Bad Request` - Invalid input
- `413 Payload Too Large` - File exceeds 50MB
- `500 Server Error` - Internal error

---

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

```bash
docker build -t buildwatch-ai .
docker run -p 5000:5000 buildwatch-ai
```

### Cloud Deployment
- **Heroku** - Add Procfile (included)
- **AWS** - EC2 + S3 for image storage
- **Google Cloud** - App Engine or Cloud Run
- **Azure** - App Service

---

## 📈 Performance Metrics

- **Processing Time** - 5-15 seconds per image pair
- **Memory Usage** - 2-4GB with YOLO model
- **Max File Size** - 50MB per image
- **Supported Formats** - JPG, JPEG, PNG
- **Output Quality** - High-resolution analysis images

---

## 🔒 Security Features

✅ Secure filename generation (random hex)  
✅ File type validation  
✅ File size limits (50MB)  
✅ Input sanitization  
✅ Automatic temp file cleanup  
✅ Error handling and logging  

---

## 🐛 Troubleshooting

### YOLO Model Not Loading
```
Solution: Download yolov8n.pt manually or ensure internet connection
```

### Out of Memory
```
Solution: Reduce image resolution or use a smaller YOLO model (yolov8s.pt)
```

### Slow Processing
```
Solution: Enable GPU acceleration with CUDA if available
```

### File Not Found Errors
```
Solution: Ensure static/ and templates/ directories exist
```

---

## 🚀 Future Roadmap

See [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) for planned features:

- Historical data tracking
- Advanced analytics dashboard
- PDF report export
- Batch processing
- Database integration
- User authentication
- API rate limiting
- Mobile application
- Cloud deployment

---

## 📚 Technologies Used

- **Backend** - Flask (Python)
- **Computer Vision** - OpenCV
- **AI/ML** - YOLOv8 (Ultralytics)
- **Deep Learning** - PyTorch
- **Frontend** - HTML5, CSS3, Vanilla JavaScript
- **UI Framework** - Custom modern CSS with glassmorphism

---

## 👨‍💻 Development

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Test thoroughly before submitting

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- 📧 Email: amberdinesh1976@gmail.com
- 🐙 GitHub Issues: [Create an issue](https://github.com/theambergit/Buildwatch-AI/issues)
- 💬 Discussions: [Join discussions](https://github.com/theambergit/Buildwatch-AI/discussions)

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics for object detection
- **OpenCV** for computer vision processing
- **Flask** for web framework
- **PyTorch** for deep learning capabilities

---

## ⭐ Show Your Support

If you found this project helpful, please star the repository! ⭐

---

**Last Updated:** 2024-06-13  
**Version:** 2.0 (Major UI/UX Overhaul)  
**Status:** Active Development
