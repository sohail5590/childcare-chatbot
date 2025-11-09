# Use an official lightweight Python image as a base
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container first
# This leverages Docker's layer caching, speeding up future builds if dependencies don't change
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY zscalar-root.crt /usr/local/share/ca-certificates/zscalar-root.crt
RUN update-ca-certificates
COPY requirements.txt .

RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy the rest of the application's source code into the container
COPY . .

# Expose port 8000 to allow communication with the API server from other containers
# Port will be 9000
# EXPOSE 8000

# The command to run when the container starts
# This starts the Uvicorn server, which runs the FastAPI application
# --host 0.0.0.0: Makes the server accessible from outside the container
# --reload: Automatically restarts the server when code changes are detected (great for development)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]