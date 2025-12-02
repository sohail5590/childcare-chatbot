FROM python:3.11-slim

# ------------------------------------------------------------
# ENVIRONMENT SETTINGS
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/zscalar-root.crt \
    SSL_CERT_FILE=/usr/local/share/ca-certificates/zscalar-root.crt \
    CURL_CA_BUNDLE=/usr/local/share/ca-certificates/zscalar-root.crt

WORKDIR /app

# ------------------------------------------------------------
# INSTALL SYSTEM DEPS + CA CERTS
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
#  🔒 ADD & REGISTER ZSCALER ROOT CERTIFICATE
# ------------------------------------------------------------
COPY zscalar-root.crt /usr/local/share/ca-certificates/zscalar-root.crt
RUN chmod 644 /usr/local/share/ca-certificates/zscalar-root.crt \
    && update-ca-certificates

# Make pip trust the cert
RUN mkdir -p /root/.config/pip && \
    echo "[global]\ncert = /usr/local/share/ca-certificates/zscalar-root.crt" \
        > /root/.config/pip/pip.conf

# ------------------------------------------------------------
#  🌐 VERIFY NETWORK ACCESS BEFORE INSTALLING PACKAGES
# ------------------------------------------------------------
RUN curl -I https://pypi.org || echo "⚠️ WARNING: curl test to PyPI failed, continuing..."

# ------------------------------------------------------------
# INSTALL PYTHON DEPENDENCIES
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install \
      --trusted-host pypi.org \
      --trusted-host pypi.python.org \
      --trusted-host files.pythonhosted.org \
      -r requirements.txt

# ------------------------------------------------------------
# COPY APPLICATION CODE
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# EXPOSE PORT FOR chat-agents SERVICE
# ------------------------------------------------------------
EXPOSE 9300

# ------------------------------------------------------------
# START FASTAPI SERVICE
# ------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9300"]
