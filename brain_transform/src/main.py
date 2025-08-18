import os
from os.path import basename
import logging
import sys
import shutil
import json
from datetime import datetime
# Add these lines before importing matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import pandas as pd
import git
from dvc.repo import Repo
from dvc.api import DVCFileSystem
from signal_toolkit import (apply_butter_bandpass_filter,
                            apply_moving_average,
                            apply_stationary,
                            generate_img)
# Read environment variables
commit_tag = os.getenv('GIT_COMMIT_TAG', 'unknown')
brain_repo_token = os.getenv('BRAIN_REPO_TOKEN', 'yYWxX4QBFmwijeGRj7Sh')
dvc_folder = os.getenv('DVCFOLDER', 'data_raw')
git_repo_url = f"https://brain_repo_token:{brain_repo_token}@code.fbi.h-da.de/mlops-brain/brain_{dvc_folder}.git"
feature_store_url = f"https://brain_repo_token:{brain_repo_token}@code.fbi.h-da.de/mlops-brain/brain_feature_store.git"
aws_creds_dir = "./.aws"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Starting data processing pipeline with commit tag: {commit_tag}")

# Directory setup
repo_dir = "./repo"
input_dir = os.path.join(repo_dir, dvc_folder)
processed_dir = "./processed"
viz_dir = "./viz"

# Create base directories
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

# Create commit tag specific directories
tag_processed_dir = os.path.join(processed_dir, commit_tag)
tag_viz_dir = os.path.join(viz_dir, commit_tag)

def get_subdirectories():
    return ['test', 'trainval']

def setup_directories(subdirs):
    for subdir in subdirs:
        # Create subdirectories for true/false within each subdir
        for label in ('true', 'false'):
            os.makedirs(os.path.join(tag_processed_dir, subdir, label), exist_ok=True)
            os.makedirs(os.path.join(tag_viz_dir, subdir, label), exist_ok=True)

def setup_git_and_dvc():
    # Clean up existing repo directory
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    
    # Clone git repository
    logging.info(f"Cloning repository from {git_repo_url}")
    #repo = git.Repo.clone_from(git_repo_url, repo_dir)
    
    # Initialize DVC and pull data
    logging.info("Pulling DVC data")
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = os.path.join(aws_creds_dir, 'credentials')
    dvc_repo = Repo(repo_dir)
    dvc_repo.pull()

def find_complete_samples(directory):
    file_groups = {}
    for subdir in get_subdirectories():
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            continue
        for file in os.listdir(subdir_path):
            if file.endswith(".csv") or file.endswith(".json"):
                prefix = file[:15]
                if prefix not in file_groups:
                    file_groups[prefix] = {'files': [], 'subdir': subdir}
                file_groups[prefix]['files'].append(file)
    
    complete_samples = {
        prefix: info['subdir']
        for prefix, info in file_groups.items()
        if sum(f.endswith(".csv") for f in info['files']) == 3 
        and sum(f.endswith(".json") for f in info['files']) == 1
    }
    return complete_samples

def process_csv(file_path, time_division, label_target, phase_values, subdir, tag_viz_dir):
    logging.info(f"Processing CSV file: {file_path}")
    df = pd.read_csv(file_path)
    time_stemps = np.arange(0, len(df), len(df) // time_division)
    parts = []
    for i, stemp in enumerate(time_stemps):
        next_stemp = time_stemps[i+1] if i < len(time_stemps)-1 else len(df)
        part = df.iloc[stemp:next_stemp, :]
        parts.append(part)
    col_names = ["Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7", "Channel_8"]
    for idx, part in enumerate(parts):
        label = 1 if idx + 1 == label_target else 0
        phase_value = phase_values[idx]
        
        # Update output directories to include subdir
        output_dir = os.path.join(tag_processed_dir, subdir, 'true' if label == 1 else 'false')
        viz_output_dir = os.path.join(tag_viz_dir, subdir, 'true' if label == 1 else 'false')
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(viz_output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_part_{idx + 1}_{commit_tag}.csv")

        part.loc[:, "label"] = label
        part.loc[:, "Task_Phase_Value"] = phase_value
        part = apply_butter_bandpass_filter(part, col_names)
        part = apply_moving_average(part, col_names, 5)
        part = apply_stationary(part, col_names)
        part.dropna(ignore_index=True, inplace=True)
        part.to_csv(output_file, index=False)

        logging.info(f"Generated processed CSV file: {output_file}")
        generate_img(
            part, 
            label, 
            f"{os.path.basename(file_path).split('.')[0]}_part_{idx + 1}_{commit_tag}",
            subdir,
            tag_viz_dir
        )

def get_already_processed_files():
    feature_store_dir = "./feature_store_repo"
    all_files = get_dvc_filenames_in_folder(
        git_repo_url=feature_store_url,
        repo_dir=feature_store_dir,
        dvc_folder="feature_store"
    )
    # Filter for files ending with commit_tag and extract first 15 digits
    processed_prefixes = set()
    for filename in all_files:
        postfix=f"_{commit_tag}.png"
        if filename.endswith(postfix):
            prefix = filename[:15]  # Get first 15 digits
            processed_prefixes.add(prefix)

    return processed_prefixes

def run_pipeline():
    try:
        logging.info("Setting up Git repository and DVC data")
        setup_git_and_dvc()
        
        # Setup all required directories
        setup_directories(get_subdirectories())
        
        # Get set of already processed sample prefixes
        processed_prefixes = get_already_processed_files()
        logging.info(f"Found {len(processed_prefixes)} already processed sample prefixes")
        
        logging.info("Checking for complete samples...")
        complete_samples = find_complete_samples(input_dir)
        for sample_prefix, subdir in complete_samples.items():
            # Check if this sample prefix has already been processed
            if sample_prefix in processed_prefixes:
                logging.info(f"Skipping {sample_prefix} - already processed")
                continue
            
            logging.info(f"Processing sample: {sample_prefix} in {subdir}")
            
            metadata_file = os.path.join(input_dir, subdir, f"{sample_prefix}_metadata.json")

            with open(metadata_file, "r") as file:
                metadata = json.load(file)

            start_year = metadata["start_year"]
            end_year = metadata["end_year"]
            subject_dob = metadata["subject_date_of_birth"]

            time_divisions = {"days": 31, "months": 12, "years": end_year - start_year}
            dob = datetime.strptime(subject_dob, "%d-%B-%Y")
            label_targets = {
                "days": dob.day,
                "months": dob.month,
                "years": dob.year - start_year + 1
            }
            phase_values = {
                "days": list(range(1, 32)),
                "months": list(range(1, 13)),
                "years": list(range(start_year, end_year + 1))
            }

            for interval_type in ("days", "months", "years"):
                csv_file = os.path.join(input_dir, subdir, f"{sample_prefix}_{interval_type}.csv")
                if os.path.exists(csv_file):
                    process_csv(csv_file, time_divisions[interval_type], label_targets[interval_type], phase_values[interval_type], subdir, tag_viz_dir)

            for file in os.listdir(os.path.join(input_dir, subdir)):
                if file.startswith(sample_prefix):
                    os.remove(os.path.join(input_dir, subdir, file))
                    logging.info(f"Removed file: {file}")

        # Cleanup repo directory after processing
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
            
    except Exception as e:
        logging.error(f"Error: {e}")

def get_dvc_filenames_in_folder(git_repo_url:str, repo_dir:str, dvc_folder:str) -> list:
    """
    1) Clones the Git+DVC repo (metadata only, no data download).
    2) Uses DVCFileSystem to list DVC-tracked files
       in the folder specified by DVCFOLDER.
    3) Returns just the filenames (no paths).
    """
    # Clean up any old clone
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    
    # Clone Git repo
    logging.info(f"Cloning repository from {git_repo_url}")
    git.Repo.clone_from(git_repo_url, repo_dir)

    # Create a DVCFileSystem from the local repo
    fs = DVCFileSystem(url=repo_dir)

    # "/myfolder" is the path inside the repo. Use leading slash.
    folder_path = f"/{dvc_folder}"
    logging.info(f"Listing DVC-tracked files in '{folder_path}'")

    # Recursively find DVC-tracked files in the given folder
    dvc_paths = fs.find(folder_path, detail=False, dvc_only=True)

    # Extract just the filenames from each path
    # Example: /data_raw/foo/bar.csv -> bar.csv
    filenames = [basename(path) for path in dvc_paths]
    return filenames
# Main Method
def main():
    try:
        logging.info("Starting brain transform pipeline")
        run_pipeline()
        logging.info("Pipeline completed successfully")
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
