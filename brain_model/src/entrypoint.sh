#!/bin/bash

set -e
# set -x  # Enable debugging
shopt -s nullglob

echo "Starting script..."

# Ensure the PATH includes the directory where DVC is installed
export PATH="/usr/local/bin:${PATH}"

# pulling feature store data
rm -rf repo
DVCFOLDER=feature_store
echo "Cloning repository..."
git clone --depth=1 https://brain_repo_token:${BRAIN_REPO_TOKEN}@code.fbi.h-da.de/mlops-brain/brain_${DVCFOLDER}.git repo
cd repo
mkdir -p "${DVCFOLDER}"
# git reset --hard origin/master -q

# Store the git commit hash of the cloned DVC repo as an environment variable
export GIT_COMMIT_HASH_DATASET=$(git rev-parse HEAD || echo "unknown")

echo "Pulling ${DVCFOLDER} data..."
dvc pull "${DVCFOLDER}" || echo "DVC pull failed, assuming empty folder."

cp -r ${DVCFOLDER} ../
cd ../

python main.py