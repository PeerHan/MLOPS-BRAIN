FROM registry.code.fbi.h-da.de/mlops-brain/brain_inference/base

RUN apt-get update && \
    apt-get install -y

COPY requirements.txt src /app/

RUN pip install -r /app/requirements.txt

WORKDIR /app

CMD ["python", "main.py"]