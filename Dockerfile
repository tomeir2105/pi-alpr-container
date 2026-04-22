FROM python:3.10-slim

# System dependencies (OpenCV, RTSP, video handling)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure runtime directories exist (used with read-only containers)
RUN mkdir -p /data /tmp /data/watcher-home

# Environment defaults (overridden by docker-compose + .env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/data/watcher-home \
    TMPDIR=/tmp

# Default entrypoint (watcher process)
CMD ["python", "alpr_watcher.py"]
