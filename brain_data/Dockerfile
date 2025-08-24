FROM python:3.12-slim

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    rsync && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirement.txt /app/requirement.txt

RUN pip install -r requirement.txt

COPY entrypoint.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]