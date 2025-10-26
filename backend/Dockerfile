# Use an official lightweight Python image as a base
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container first
# This leverages Docker's layer caching, speeding up future builds if dependencies don't change
COPY requirements.txt .

# Install the Python dependencies listed in the requirements.txt file
# The --no-cache-dir flag keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code into the container
COPY . .

# Expose port 8000 to allow communication with the API server from other containers
EXPOSE 8000

# The command to run when the container starts
# This starts the Uvicorn server, which runs the FastAPI application
# --host 0.0.0.0: Makes the server accessible from outside the container
# --reload: Automatically restarts the server when code changes are detected (great for development)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]