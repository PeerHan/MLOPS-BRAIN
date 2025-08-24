#!/bin/sh

RUNNER_DESCRIPTION=$RUNNER_TAG_LIST

if [ ! -f /etc/gitlab-runner/config.toml ]; then
  echo "Registering GitLab Runner..."
  gitlab-runner register --non-interactive \
    --url "$CI_SERVER_URL" \
    --registration-token "$REGISTRATION_TOKEN" \
    --executor "$RUNNER_EXECUTOR" \
    --docker-image "$DOCKER_IMAGE" \
    --description "$RUNNER_DESCRIPTION" \
    --tag-list "$RUNNER_TAG_LIST" \
    --docker-network-mode "$DOCKER_NETWORK_MODE"
else
  echo "GitLab Runner already registered."
fi

gitlab-runner run