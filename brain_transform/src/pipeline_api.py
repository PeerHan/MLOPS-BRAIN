import os
import json
import shutil
from pathlib import Path
import sys
import logging
import git
from dvc.repo import Repo
from dvc.api import DVCFileSystem
import requests

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("dvc").setLevel(logging.ERROR)  # Set DVC logging level to ERROR
logging.getLogger("botocore").setLevel(logging.ERROR)  # Set botocore logging level to ERROR

# Read environment variables
brain_repo_token = os.getenv('BRAIN_REPO_TOKEN')
dvc_folder = os.getenv('DVCFOLDER')
api_base_url = os.getenv('API_URL')
git_repo_url = f"https://brain_repo_token:{brain_repo_token}@code.fbi.h-da.de/mlops-brain/brain_{dvc_folder}.git"
feature_store_url = f"https://brain_repo_token:{brain_repo_token}@code.fbi.h-da.de/mlops-brain/brain_feature_store.git"
aws_creds_dir = "./.aws"

logging.debug(f"brain_repo_token: {brain_repo_token}")
logging.debug(f"dvc_folder: {dvc_folder}")
logging.debug(f"api_base_url: {api_base_url}")

# Get API version/commit tag
for attempt in range(2):
    try:
        response = requests.get(f"{api_base_url}/version")
        response.raise_for_status()
        commit_tag = response.json()["commit_tag"]
        logging.info(f"Using API commit tag: {commit_tag}")
        break
    except Exception as e:
        if attempt == 1:
            raise
        logging.warning(f"Retrying to get initial API version after error: {str(e)}")

# Directory setup
repo_dir = "./repo"
processed_dir = "./processed"
viz_dir = "./viz"

def setup_directories(subdirs):
    for subdir in subdirs:
        for label in ('true', 'false'):
            os.makedirs(os.path.join(processed_dir, commit_tag, subdir, label), exist_ok=True)
            os.makedirs(os.path.join(viz_dir, commit_tag, subdir, label), exist_ok=True)

def transform_via_api(csv_path: str, metadata_path: str, output_dir: str, viz_output_dir: str, filename_stem: str):
    logging.debug(f"Transform call for CSV: {csv_path}, Metadata: {metadata_path}")
    """Transform data using the API service (no interval_type in pipeline)"""
    try:
        # Prepare files for upload
        files = {
            'csv_file': open(csv_path, 'rb'),
            'metadata_file': open(metadata_path, 'rb')
        }
        
        # Make API request
        for attempt in range(2):
            try:
                response = requests.post(f"{api_base_url}/transform/", files=files)
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == 1:
                    raise
                logging.warning(f"Retrying transform API call after error: {str(e)}")
        
        # Parse response
        data = response.json()
        
        # Process each part
        for part in data['parts']:
            logging.debug(f"API Response part: {part}")
            part_suffix = f"_part_{part['part_index']}_{commit_tag}"
            label_dir = 'true' if part['label'] == 1 else 'false'
            
            # Save CSV data
            csv_output_path = os.path.join(output_dir, label_dir, f"{filename_stem}{part_suffix}.csv")
            with open(csv_output_path, 'w') as f:
                f.write(part['csv_data'])
            
            # Download and save visualization
            viz_url = f"{api_base_url}{part['visualization']}"
            for attempt in range(2):
                try:
                    viz_response = requests.get(viz_url)
                    viz_response.raise_for_status()
                    break
                except Exception as e:
                    if attempt == 1:
                        raise
                    logging.warning(f"Retrying visualization download after error: {str(e)}")
            
            viz_output_path = os.path.join(viz_output_dir, label_dir, f"{filename_stem}{part_suffix}.png")
            logging.debug(f"Downloading visualization from {viz_url}")
            logging.debug(f"Response status code: {viz_response.status_code}")
            logging.debug(f"Saving visualization to: {viz_output_path}")
            
            with open(viz_output_path, 'wb') as f:
                f.write(viz_response.content)
                
            logging.info(f"Generated files: {csv_output_path}, {viz_output_path}")
        
        if not data.get('parts'):
            logging.debug("No parts returned from API response.")
        
        return True
        
    except Exception as e:
        logging.error(f"Error transforming file {csv_path}: {str(e)}")
        return False
    finally:
        for f in files.values():
            f.close()

def process_csv(file_path, subdir):
    logging.debug(f"process_csv arguments: file_path={file_path}, subdir={subdir}")
    """Process CSV file using the API service"""
    logging.info(f"Processing CSV file: {file_path}")
    
    # Get metadata file path
    filename_stem = Path(file_path).stem
    metadata_path = os.path.join(os.path.dirname(file_path), f"{filename_stem[:15]}_metadata.json")
    
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found: {metadata_path}")
        return False
    
    # Load metadata and add interval_type
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if "days" in filename_stem:
        metadata["interval_type"] = "days"
    elif "months" in filename_stem:
        metadata["interval_type"] = "months"
    elif "years" in filename_stem:
        metadata["interval_type"] = "years"
    else:
        logging.error(f"Unknown interval type for file: {file_path}")
        return False
    
    # Save updated metadata back to file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    output_dir = os.path.join(processed_dir, commit_tag, subdir)
    viz_output_dir = os.path.join(viz_dir, commit_tag, subdir)
    
    return transform_via_api(
        file_path,
        metadata_path,
        output_dir,
        viz_output_dir,
        filename_stem
    )

def find_complete_samples(directory):
    file_groups = {}
    for subdir in ('test', 'trainval'):
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

def get_api_version():
    """Get the version/commit tag of the API service"""
    for attempt in range(2):
        try:
            response = requests.get(f"{api_base_url}/version")
            response.raise_for_status()
            return response.json()["commit_tag"]
        except Exception as e:
            if attempt == 1:
                raise
            logging.warning(f"Retrying get_api_version after error: {str(e)}")

def get_already_processed_files(api_commit_tag):
    """Check for already processed files using the API's commit tag"""
    feature_store_dir = "./feature_store_repo"
    try:
        # Clean up any old clone
        if os.path.exists(feature_store_dir):
            shutil.rmtree(feature_store_dir)
        
        # Clone feature store repo
        for attempt in range(2):
            try:
                logging.info(f"Cloning feature store from {feature_store_url}")
                git.Repo.clone_from(feature_store_url, feature_store_dir)
                break
            except Exception as e:
                if attempt == 1:
                    raise
                logging.warning(f"Retrying feature store clone after error: {str(e)}")

        # Use DVCFileSystem to list files
        fs = DVCFileSystem(url=feature_store_dir)
        dvc_paths = fs.find("/feature_store", detail=False, dvc_only=True)
        
        # Extract prefixes from processed files
        processed_prefixes = set()
        postfix = f"_{api_commit_tag}.png"  # Use API's commit tag instead of our own
        for path in dvc_paths:
            filename = os.path.basename(path)
            if filename.endswith(postfix):
                prefix = filename[:15]
                processed_prefixes.add(prefix)
                
        return processed_prefixes
    finally:
        if os.path.exists(feature_store_dir):
            shutil.rmtree(feature_store_dir)

def run_pipeline():
    logging.debug("Starting run_pipeline debugging phase")
    """Main pipeline function (no interval loop)"""
    try:
        # Get API version first
        api_commit_tag = get_api_version()
        if not api_commit_tag:
            raise Exception("Could not get API version, aborting pipeline")
        logging.info(f"API version: {api_commit_tag}")
        
        # Setup Git and DVC
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        
        for attempt in range(2):
            try:
                logging.info(f"Cloning repository from {git_repo_url}")
                repo = git.Repo.clone_from(git_repo_url, repo_dir)
                break
            except Exception as e:
                if attempt == 1:
                    raise
                logging.warning(f"Retrying repository clone after error: {str(e)}")
        
        logging.info("Pulling DVC data")
        os.environ['AWS_SHARED_CREDENTIALS_FILE'] = os.path.join(aws_creds_dir, 'credentials')
        dvc_repo = Repo(repo_dir)
        dvc_repo.pull()
        
        # Get already processed samples using API's commit tag
        processed_prefixes = get_already_processed_files(api_commit_tag)
        logging.info(f"Found {len(processed_prefixes)} already processed sample prefixes")
        
        # Process files
        subdirs = ['test', 'trainval']
        setup_directories(subdirs)
        
        input_dir = os.path.join(repo_dir, dvc_folder)
        complete_samples = find_complete_samples(input_dir)
        
        processed_any_data = False
        
        for sample_prefix, subdir in complete_samples.items():
            # Skip if already processed
            if sample_prefix in processed_prefixes:
                logging.info(f"Skipping {sample_prefix} - already processed")
                continue
                
            logging.info(f"Processing sample: {sample_prefix} in {subdir}")
            
            metadata_file = os.path.join(input_dir, subdir, f"{sample_prefix}_metadata.json")
            with open(metadata_file, "r") as file:
                metadata = json.load(file)

            for file in os.listdir(os.path.join(input_dir, subdir)):
                # Instead of checking for “_full.csv”, call process_csv on any CSV that matches prefix
                if file.startswith(sample_prefix) and file.endswith(".csv"):
                    csv_file = os.path.join(input_dir, subdir, file)
                    process_csv(csv_file, subdir)
            
            # Clean up processed files
            for file in os.listdir(os.path.join(input_dir, subdir)):
                if file.startswith(sample_prefix):
                    os.remove(os.path.join(input_dir, subdir, file))
                    logging.debug(f"Removing file: {file}")
                    logging.info(f"Removed file: {file}")
                    
            processed_any_data = True
        
        return processed_any_data
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)

def main():
    try:
        logging.info("Starting brain transform pipeline using API")
        processed_any_data = run_pipeline()
        if processed_any_data:
            logging.info("New data was transformed.")
        else:
            logging.info("No new data was available for transformation.")
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
