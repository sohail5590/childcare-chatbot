FROM python:3.11-slim

# ------------------------------------------------------------
# ENVIRONMENT SETTINGS
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------------------------------------------
# INSTALL SYSTEM DEPS + STANDARD CA CERTS
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        gnupg \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# INSTALL PYTHON DEPENDENCIES
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ------------------------------------------------------------
# COPY APPLICATION CODE
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# EXPOSE PORT (confirm this matches your app)
# ------------------------------------------------------------
EXPOSE 9300

# ------------------------------------------------------------
# START SERVICE
# ------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9300"]
