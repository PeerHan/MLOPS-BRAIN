FROM python:3.11-alpine as builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache gcc musl-dev python3-dev

# Install Python packages
COPY requirements_trigger.txt .
RUN pip install --no-cache-dir -r requirements_trigger.txt

FROM python:3.11-alpine

WORKDIR /app

# Copy only the necessary files from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY additional_scripts/model_trigger.py .

ENTRYPOINT ["python", "model_trigger.py"]
