# Use an official lightweight Python image as a base
FROM python:3.10-slim

# Disable buffering and noninteractive apt prompts
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# -----------------------------------------------------------
# 🧠 Install CA certificates + add Zscaler or corporate root CA
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

# Copy your Zscaler/corporate root certificate into container
# Make sure this file exists in your frontend folder (like in backend)
COPY zscalar-root.crt /usr/local/share/ca-certificates/zscalar-root.crt

# Update trust store inside image
RUN update-ca-certificates

# -----------------------------------------------------------
# 🧩 Copy requirements and install dependencies
# -----------------------------------------------------------
COPY requirements.txt .

# Upgrade pip and install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --trusted-host pypi.python.org \
        -r requirements.txt

# -----------------------------------------------------------
# Copy app (if not mounted as volume)
# COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck for container monitoring
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# -----------------------------------------------------------
# Run Streamlit
# -----------------------------------------------------------
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]