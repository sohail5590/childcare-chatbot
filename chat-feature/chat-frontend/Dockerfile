FROM python:3.11-slim

# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------------------------------------------
# SYSTEM DEPENDENCIES + STANDARD CA CERTS
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    gnupg \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# PYTHON DEPENDENCIES
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ------------------------------------------------------------
# APPLICATION CODE
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
