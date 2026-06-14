# Use official slim Python runtime as base
FROM python:3.9-slim

# Set environment variables — aggressive memory reduction
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    PORT=5000 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=131072 \
    MALLOC_TOP_PAD_=131072 \
    MPLCONFIGDIR=/tmp \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Set work directory
WORKDIR /app

# Install only essential system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt .

# Install python dependencies with no cache to save disk
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

# Start with 1 worker, preload to share memory, max-requests to prevent leaks
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "--max-requests", "100", "--max-requests-jitter", "20", "--preload", "app:app"]
