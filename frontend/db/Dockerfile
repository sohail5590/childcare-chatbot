# Use official lightweight Python
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Working directory
WORKDIR /app

# -----------------------------------------------------------
# 🔐 Install CA certs + add Zscaler corporate root CA
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        curl \
        gcc \
        libpq-dev \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Add Zscaler CA
COPY zscalar-root.crt /usr/local/share/ca-certificates/zscalar-root.crt
RUN update-ca-certificates

# -----------------------------------------------------------
# 📦 Install Python dependencies
# -----------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --trusted-host pypi.org \
        --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org \
        -r /app/requirements.txt

# -----------------------------------------------------------
# 📁 Copy source code (including seed.py & start.sh)
# -----------------------------------------------------------
COPY . /app

# Ensure scripts are executable
RUN chmod +x /app/start.sh

# -----------------------------------------------------------
# 🏁 ENTRYPOINT — auto-seed + start Uvicorn
# -----------------------------------------------------------
ENTRYPOINT ["/app/start.sh"]

# -----------------------------------------------------------
# 🔌 Expose API port
# -----------------------------------------------------------
EXPOSE 9200

# -----------------------------------------------------------
# ❤️ Healthcheck
# -----------------------------------------------------------
HEALTHCHECK CMD curl --fail http://localhost:9200/docs || exit 1
