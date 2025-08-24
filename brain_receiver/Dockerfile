# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install python-multipart

# Copy application code
COPY main.py .

# Expose the FastAPI default port
EXPOSE 8000

# Command to start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
