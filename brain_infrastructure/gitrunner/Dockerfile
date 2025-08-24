FROM gitlab/gitlab-runner:alpine3.21

ENV CI_SERVER_URL=https://code.fbi.h-da.de
ENV RUNNER_EXECUTOR=docker
ENV DOCKER_IMAGE=alpine:latest
ENV DOCKER_NETWORK_MODE=host

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]