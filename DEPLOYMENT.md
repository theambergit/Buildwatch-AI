# 🚀 BuildWatch AI - Deployment Guide

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Platforms](#cloud-platforms)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring & Logging](#monitoring--logging)

---

## Local Development

### Setup

```bash
# Clone repository
git clone https://github.com/theambergit/Buildwatch-AI.git
cd Buildwatch-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp .env.example .env
# Edit .env with your settings

# Run application
python app.py
```

Visit `http://localhost:5000`

### Development with Hot Reload

```bash
export FLASK_DEBUG=True
export FLASK_ENV=development
python app.py
```

---

## Docker Deployment

### Build Docker Image

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create uploads directory
RUN mkdir -p static uploads

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Run application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

### Build and Run

```bash
# Build image
docker build -t buildwatch-ai:latest .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/static:/app/static \
  -e FLASK_ENV=production \
  buildwatch-ai:latest

# With environment file
docker run -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/static:/app/static \
  buildwatch-ai:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  buildwatch-ai:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./static:/app/static
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
      - HOST=0.0.0.0
      - PORT=5000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Run with Docker Compose:
```bash
docker-compose up -d
```

---

## Cloud Platforms

### Heroku Deployment

1. **Setup Heroku CLI**
```bash
# Install Heroku CLI
curl https://cli.heroku.com/install.sh | sh

# Login
heroku login
```

2. **Create and Deploy**
```bash
# Create app
heroku create your-app-name

# Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set HOST=0.0.0.0

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

3. **Procfile** (included)
```
web: gunicorn -w 4 -b 0.0.0.0:$PORT --timeout 120 app:app
```

### AWS Deployment

#### Using EC2

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance.ip

# Install dependencies
sudo yum update -y
sudo yum install python3 python3-pip git -y

# Clone and setup
git clone https://github.com/theambergit/Buildwatch-AI.git
cd Buildwatch-AI
pip install -r requirements.txt

# Run with systemd
sudo tee /etc/systemd/system/buildwatch.service > /dev/null <<EOF
[Unit]
Description=BuildWatch AI
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/Buildwatch-AI
ExecStart=/usr/local/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl start buildwatch
sudo systemctl enable buildwatch
```

#### Using Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.9 buildwatch-ai

# Create environment
eb create production

# Deploy
eb deploy

# Monitor
eb logs
```

### Google Cloud Deployment

#### Using Cloud Run

```bash
# Authenticate
gcloud auth login

# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/buildwatch-ai

# Deploy
gcloud run deploy buildwatch-ai \
  --image gcr.io/PROJECT_ID/buildwatch-ai \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 120
```

#### Using App Engine

```bash
# Create app.yaml
cat > app.yaml <<EOF
runtime: python39
entrypoint: gunicorn -w 4 -b 0.0.0.0:8080 app:app

env_variables:
  FLASK_ENV: "production"

automatic_scaling:
  min_instances: 1
  max_instances: 10
EOF

# Deploy
gcloud app deploy
```

### Azure Deployment

```bash
# Create resource group
az group create --name buildwatch-rg --location eastus

# Create App Service Plan
az appservice plan create \
  --name buildwatch-plan \
  --resource-group buildwatch-rg \
  --sku B2 --is-linux

# Create Web App
az webapp create \
  --resource-group buildwatch-rg \
  --plan buildwatch-plan \
  --name buildwatch-app \
  --runtime "PYTHON|3.9"

# Deploy from GitHub
az webapp up --repo-url https://github.com/theambergit/Buildwatch-AI.git
```

---

## Performance Optimization

### 1. Use Gunicorn with Multiple Workers

```bash
# 4 workers (adjust based on CPU cores)
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app

# With gevent workers (better for I/O)
pip install gunicorn gevent
gunicorn -w 4 -k gevent -b 0.0.0.0:5000 app:app
```

### 2. Enable GPU Support

```bash
# Check for GPU
nvidia-smi

# In config.py
USE_GPU = True

# In requirements.txt (replace torch line)
torch==2.0.0+cu118
```

### 3. Use Smaller YOLO Model

```bash
# Faster but less accurate
YOLO_MODEL=yolov8s.pt  # Small model
YOLO_MODEL=yolov8n.pt  # Nano model (default)

# More accurate but slower
YOLO_MODEL=yolov8m.pt  # Medium model
```

### 4. Image Caching

```python
# Add to app.py for static files
@app.after_request
def set_cache_headers(response):
    response.cache_control.max_age = 3600
    return response
```

### 5. Load Balancing (Nginx)

```nginx
# nginx.conf
upstream buildwatch {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name buildwatch.example.com;

    location / {
        proxy_pass http://buildwatch;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files
    location /static {
        alias /var/www/buildwatch/static;
        expires 7d;
    }
}
```

---

## Monitoring & Logging

### Application Logging

```python
# Configure logging
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('buildwatch.log', 
                                       maxBytes=10485760, 
                                       backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
```

### Health Monitoring

```bash
# Using curl
watch -n 5 'curl http://localhost:5000/health'

# Using healthcheck tools
docker healthcheck stats buildwatch-ai
```

### Log Aggregation

#### ELK Stack (Elasticsearch, Logstash, Kibana)

```yaml
# docker-compose with ELK
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.0.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000/udp"

  kibana:
    image: docker.elastic.co/kibana/kibana:8.0.0
    ports:
      - "5601:5601"
```

### Monitoring Tools

- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **New Relic** - APM monitoring
- **Datadog** - Full-stack monitoring

---

## Security Checklist

- [ ] Use HTTPS/SSL certificate
- [ ] Enable rate limiting
- [ ] Add authentication/API keys
- [ ] Validate and sanitize all inputs
- [ ] Use environment variables for secrets
- [ ] Run behind reverse proxy (Nginx)
- [ ] Keep dependencies updated
- [ ] Regular security audits
- [ ] Implement CORS properly
- [ ] Monitor access logs

---

## Troubleshooting

### High Memory Usage
```bash
# Reduce max workers
gunicorn -w 2 -b 0.0.0.0:5000 app:app

# Use smaller YOLO model
YOLO_MODEL=yolov8n.pt
```

### Slow Processing
```bash
# Enable GPU
USE_GPU=True

# Increase timeouts
gunicorn --timeout 300 app:app
```

### Connection Timeouts
```bash
# Increase timeout
gunicorn --timeout 120 app:app

# Use connection pooling
```

### Out of Disk Space
```bash
# Clean old uploads
find static/ -type f -mtime +7 -delete

# Archive logs
gzip buildwatch.log
```

---

## Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Set `FLASK_DEBUG=False`
- [ ] Use strong secret keys
- [ ] Enable logging
- [ ] Setup monitoring
- [ ] Configure backups
- [ ] Setup SSL/TLS
- [ ] Use reverse proxy
- [ ] Rate limiting enabled
- [ ] Load balancer configured
- [ ] Health checks working
- [ ] Error pages customized
- [ ] CORS configured
- [ ] Dependencies pinned

---

**Last Updated:** June 2024  
**Version:** 2.0
