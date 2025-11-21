# ================================
# Backend Dockerfile (Clean + Fixed)
# ================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Working directory for your backend
WORKDIR /app/backend

# Install dependencies needed for HF + SentenceTransformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Add Zscaler certificate
COPY zscalar-root.crt /usr/local/share/ca-certificates/zscalar-root.crt
RUN update-ca-certificates

# Copy requirements first (for Docker caching)
COPY requirements.txt /app/backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy your backend source code ONLY
COPY . /app/backend

# Environment variable for HF SSL fix
ENV HF_HUB_DISABLE_SSL_VERIFICATION=1
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Run backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]