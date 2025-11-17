FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/zscalar-root.crt \
    SSL_CERT_FILE=/usr/local/share/ca-certificates/zscalar-root.crt \
    CURL_CA_BUNDLE=/usr/local/share/ca-certificates/zscalar-root.crt

WORKDIR /app

# Install CA utilities and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates gnupg \
 && rm -rf /var/lib/apt/lists/*

# --- 🔒 Add and register Zscaler Root Certificate ---
COPY zscalar-root.crt /usr/local/share/ca-certificates/zscalar-root.crt
RUN chmod 644 /usr/local/share/ca-certificates/zscalar-root.crt \
 && update-ca-certificates

# Make sure pip trusts the custom cert
RUN mkdir -p /root/.config/pip && \
    echo "[global]\ncert = /usr/local/share/ca-certificates/zscalar-root.crt" > /root/.config/pip/pip.conf

# Verify connectivity before pip install
RUN curl -I https://pypi.org || echo "⚠️ curl test failed, but continuing..."

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host=files.pythonhosted.org \
        -r requirements.txt

# Copy code
COPY . /app

EXPOSE 9100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9100", "--reload"]