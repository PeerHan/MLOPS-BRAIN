import requests
import mlflow
import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables
mlflow_server = os.getenv("MLFLOW_SERVER")
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
transform_api_url = os.getenv("TRANSFORM_API_URL")
mlflow.set_tracking_uri(uri=mlflow_server)

os.environ['GIT_USER'] = os.getenv('GIT_USER')
os.environ['GIT_PASS'] = os.getenv('GIT_PASS')

def get_version_info():
    """
    Fetch the commit tag and commit hash from the version endpoint.
    """
    url = f"{transform_api_url}/version"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("commit_tag"), data.get("commit_hash")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch version info: {e}")
        raise

def get_latest_model_version(commit_tag):
    """
    Fetch the latest model version for the given commit tag.
    Returns None if no versions are found.
    """
    model_name = f"model_{commit_tag}"
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.info(f"No versions found for model '{model_name}'")
            return None
        latest_version = max(versions, key=lambda v: int(v.version))
        return latest_version
    except Exception as e:
        logger.error(f"Failed to fetch latest model version: {e}")
        raise

def get_git_dataset_hash(latest_version):
    """
    Access the source run of the latest model version and get the parameter `git_dataset_hash`.
    """
    client = mlflow.tracking.MlflowClient()
    run_id = latest_version.run_id
    try:
        run = client.get_run(run_id)
        git_dataset_hash = run.data.params.get("git_dataset_hash")
        if git_dataset_hash is None:
            raise Exception("Parameter 'git_dataset_hash' not found in the run")
        return git_dataset_hash
    except Exception as e:
        logger.error(f"Failed to fetch git dataset hash: {e}")
        raise

def get_nfiles(commit_hash):
    """
    Fetch the number of files from the .dvc file for the given commit hash.
    """
    token = os.getenv('GIT_PASS')
    repo_file_url = f"https://code.fbi.h-da.de/api/v4/projects/38757/repository/files/feature_store.dvc/raw?ref={commit_hash}"
    headers = {"Private-Token": token}
    try:
        response = requests.get(repo_file_url, headers=headers, timeout=10)
        response.raise_for_status()
        if "<html" in response.text.lower():
            raise Exception("Received HTML instead of expected YAML. Check token permissions.")
        dvc_data = yaml.safe_load(response.text)
        return dvc_data["outs"][0]["nfiles"]
    except requests.RequestException as e:
        logger.error(f"Failed to fetch nfiles: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML: {e}")
        raise

def compare_nfiles_with_master():
    """
    Compare the nfiles value of the dataset hash with the nfiles number of the latest version in master.
    Always reports a difference if no model version is found.
    """
    try:
        commit_tag, commit_hash = get_version_info()
        logger.info(f"Commit Tag: {commit_tag}")
        logger.info(f"Commit Hash: {commit_hash}")

        latest_version = get_latest_model_version(commit_tag)
        
        if latest_version is None:
            logger.info("No model versions found, treating as different from master")
            logger.info("File counts differ between the dataset commit and master.")
            return

        logger.info(f"Latest Model Version: {latest_version.version}")
        git_dataset_hash = get_git_dataset_hash(latest_version)
        logger.info(f"Git Dataset Hash: {git_dataset_hash}")
        nfiles_dataset = get_nfiles(git_dataset_hash)
            
        nfiles_master = get_nfiles("master")
        logger.info(f"Nfiles (dataset): {nfiles_dataset}")
        logger.info(f"Nfiles (master): {nfiles_master}")

        if nfiles_dataset == nfiles_master:
            logger.info("Both commits have the same number of files.")
        else:
            logger.info("File counts differ between the dataset commit and master.")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    compare_nfiles_with_master()
    # Ensure the script exits with a status code indicating success
    exit(0)
