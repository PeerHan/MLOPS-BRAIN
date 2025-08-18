FROM python:3.13-slim

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    rsync && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY src/drift_detection.py /app/drift_detection.py

EXPOSE 8000

CMD ["uvicorn", "drift_detection:app", "--host", "0.0.0.0", "--port", "8000"]