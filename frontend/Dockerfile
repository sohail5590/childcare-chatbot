# Use an official lightweight Python image as a base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the Streamlit application code into the working directory
# Taken care in the main docker-compose file as volume
# COPY app.py .


# Expose the port that Streamlit runs on
EXPOSE 8501

# The command to run when the container starts
# CMD ["streamlit", "run", "app.py"]

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]