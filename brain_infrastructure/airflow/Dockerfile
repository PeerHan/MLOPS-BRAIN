FROM apache/airflow:2.10.4
USER root
RUN apt-get update && apt-get install -y docker.io \
    tzdata
ENV TZ=Europe/Berlin
USER airflow
RUN pip install --no-cache-dir \
        python-dotenv
