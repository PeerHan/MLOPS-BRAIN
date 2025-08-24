# Use the official MLflow image as the base
FROM ghcr.io/mlflow/mlflow:v2.20.0

# Install mlflow[pipelines]
RUN pip install mlflow[pipelines]
RUN pip install mlflow[extras]
RUN pip install mlflow-torchserve
RUN pip install pymysql