FROM registry.code.fbi.h-da.de/mlops-brain/brain_model/base:latest as base

FROM python:3.12-slim
WORKDIR /app

ARG COMMIT_TAG=unknown
ENV GIT_COMMIT_TAG=${COMMIT_TAG}

# Copy Python packages and DVC executable
COPY --from=base /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=base /usr/local/bin/dvc /usr/local/bin/
COPY src /app
COPY requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        openssh-client \
        python3-pip \
        curl \
    && update-ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mv /app/entrypoint.sh /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# Add DNS resolution retry
RUN echo "options timeout:1 attempts:3 rotate" >> /etc/resolv.conf

ENTRYPOINT ["entrypoint.sh"]