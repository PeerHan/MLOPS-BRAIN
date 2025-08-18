# Use official Python image
FROM python:3.9-slim

# Install git properly and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set git executable path
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

# Set working directory
WORKDIR /app

# Accept build argument with default value
ARG COMMIT_TAG=unknown

# Accept build argument
ARG COMMIT_TAG

# Set environment variable
ENV GIT_COMMIT_TAG=${COMMIT_TAG}

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/main.py .
COPY src/signal_toolkit.py .

# Command to start FastAPI
CMD ["python", "main.py"]