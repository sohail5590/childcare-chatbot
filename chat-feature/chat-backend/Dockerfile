FROM python:3.11-slim

# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------------------------------------------
# SYSTEM DEPS + CA CERTS
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    gnupg \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# PYTHON DEPS
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ------------------------------------------------------------
# APP CODE
# ------------------------------------------------------------
COPY . /app

EXPOSE 9100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9100"]
