# Use official slim Python runtime as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    PORT=5000

# Set work directory
WORKDIR /app

# Install system dependencies required for OpenCV headless and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Ensure necessary directories exist for file uploads and analyses
RUN mkdir -p static/analyses uploads

# Expose server port
EXPOSE 5000

# Health check to ensure service is responding
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, json; \
                   res = urllib.request.urlopen('http://localhost:5000/health'); \
                   data = json.loads(res.read().decode()); \
                   exit(0 if data['status'] == 'healthy' else 1)"

# Start Flask application using Gunicorn WSGI server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "--max-requests", "500", "app:app"]
