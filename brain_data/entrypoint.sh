#!/bin/bash

set -e
# set -x  # Enable debugging
shopt -s nullglob

echo "Starting script..."

# Add retry function
clone_with_retry() {
    local max_attempts=3
    local delay=5
    local attempt=1
    
    while true; do
        echo "Clone attempt $attempt of $max_attempts..."
        if git clone --depth=1 "https://brain_repo_token:${BRAIN_REPO_TOKEN}@code.fbi.h-da.de/mlops-brain/brain_${DVCFOLDER}.git" repo; then
            echo "Clone successful!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -gt $max_attempts ]; then
            echo "Failed to clone after $max_attempts attempts"
            return 1
        fi
        
        echo "Clone failed, retrying in $delay seconds..."
        sleep $delay
    done
}

# Remove the repository if it exists
rm -rf repo

# Clone with retry
echo "Cloning repository..."
if ! clone_with_retry; then
    echo "Failed to clone repository after all attempts. Exiting."
    exit 1
fi

# Enter the repository
cd repo

# Create the DVC folder if it doesn't exist
mkdir -p "${DVCFOLDER}"

# Configure Git
git config --global user.email "brain_${DVCFOLDER}"
git config --global user.name "brain_${DVCFOLDER}"

# Pull DVC data
# echo "Pulling DVC data..."
dvc pull "${DVCFOLDER}" || echo "DVC pull failed, assuming empty folder."

# Sync new files into the DVC folder
files=(/app/collect_point/*)
if [ ${#files[@]} -eq 0 ]; then
    echo "No new files to sync."
    exit 0
fi

echo "Syncing new files..."
rsync -a "${files[@]}" "${DVCFOLDER}/"

# Add all changes to DVC and push
echo "Adding and pushing changes to DVC..."
dvc add "${DVCFOLDER}" && dvc push

# Check if there are any changes to commit
if git diff-index --quiet HEAD --; then
    echo "No changes to commit."
else
    # Commit the changes to Git
    echo "Committing changes to Git..."
    git add .
    git commit -m "Update ${DVCFOLDER} with new files"
    git push
fi

# Clean up the collection point
echo "Cleaning up collection point..."
rm -r /app/collect_point/*

# Remove the repository
cd ..
rm -rf repo

echo "Script completed."
