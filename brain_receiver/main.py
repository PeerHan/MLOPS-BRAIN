from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import os
import pandas as pd
import json
from pathlib import Path
import random
import requests
from datetime import datetime, timedelta
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()
infer_api_url = os.getenv('INFER_API_URL')
drift_api_url = os.getenv('DRIFT_API_URL')

# Set up Basic Auth
security = HTTPBasic()
USER_CREDENTIALS = {
    "username": os.getenv("BASIC_AUTH_USERNAME", "default_user"),
    "password": os.getenv("BASIC_AUTH_PASSWORD", "default_password")
}
# print(f"Basic Auth: {USER_CREDENTIALS}")

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if USER_CREDENTIALS.get("username") != credentials.username or USER_CREDENTIALS.get("password") != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return credentials.username

# Define the upload directory
UPLOAD_DIR = "/app/uploads/output"
TRAINVAL_DIR = os.path.join(UPLOAD_DIR, "trainval")
os.makedirs(TRAINVAL_DIR, exist_ok=True)

# Mockup process_experiment_data function
def process_experiment_data(csv_file, task_phase):
    try:
        # Load the CSV file
        data = pd.read_csv(csv_file)

        # Ensure required columns are present
        if "Timestamp" not in data.columns or task_phase not in data["Task_Phase"].unique():
            raise ValueError("Invalid data format or task phase mismatch.")

        # Determine the number of chunks based on the task phase
        if task_phase == "days":
            num_chunks = 31
        elif task_phase == "months":
            num_chunks = 12
        elif task_phase == "years":
            num_chunks = 8
        elif task_phase == "meta":
            return {"chunk_index": None}
        else:
            raise ValueError(f"Unknown task phase: {task_phase}")

        # Divide the data into chunks
        chunk_size = len(data) // num_chunks
        chunks = [data.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks - 1)]
        chunks.append(data.iloc[(num_chunks - 1) * chunk_size:])

        if len(chunks) != num_chunks:
            raise ValueError(f"Mismatch in expected chunks: {len(chunks)} vs {num_chunks}")

        # Mockup: Select a random chunk as the "outlier"
        chosen_chunk = random.randint(0, num_chunks - 1)
        return {"chunk_index": chosen_chunk}

    except Exception as e:
        return {"error": str(e)}

def send_to_infer_api(csv_path, metadata_path=None):
    url = f"{infer_api_url}/predict/"
    files = {
        "csv_file": ("csv_file", open(csv_path, "rb"), "text/plain")
    }
    if metadata_path:
        files["metadata_file"] = ("metadata_file", open(metadata_path, "rb"), "text/plain")
    response = requests.post(url, files=files)
    response.raise_for_status()
    return response.json()

def send_to_drift_api(file_prefix):
    """Send complete sample to drift detection API."""
    url = f"{drift_api_url}/update"
    
    # Find all CSV files for this prefix
    files = {
        "days_file": None,
        "months_file": None,
        "years_file": None
    }
    
    for file in os.listdir(TRAINVAL_DIR):
        if file.startswith(file_prefix):
            if file.endswith('.csv'):
                if 'days' in file:
                    files["days_file"] = open(os.path.join(TRAINVAL_DIR, file), 'rb')
                elif 'months' in file:
                    files["months_file"] = open(os.path.join(TRAINVAL_DIR, file), 'rb')
                elif 'years' in file:
                    files["years_file"] = open(os.path.join(TRAINVAL_DIR, file), 'rb')
    
    # Add signature
    data = {"signature": file_prefix}
    
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    finally:
        # Close all opened files
        for f in files.values():
            if f:
                f.close()

def is_valid_metadata(metadata_path):
    """Validate that metadata JSON contains a valid YYYY-MM-DD subject_date_of_birth."""
    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
        dob = data.get("subject_date_of_birth")
        datetime.strptime(dob, "%Y-%m-%d")
        return True
    except Exception as e:
        return False

@app.post("/upload/")
async def upload_data(
    csv_file: UploadFile,
    metadata_file: UploadFile = None,
    task_phase: str = Form(...),
    username: str = Depends(authenticate)
):
    """
    Uploads a CSV file and optionally a JSON metadata file. Processes the data and stores it.
    """
    if csv_file.filename == "placeholder":
        return JSONResponse({
            "message": "Placeholder file detected, no processing needed.",
            "processing_result": None,
            "inference_result": None
        })

    try:
        # Ensure directory exists
        os.makedirs(TRAINVAL_DIR, exist_ok=True)
        
        # Get the first 11 characters as prefix
        file_prefix = csv_file.filename[:11]
        
        # Save files with proper error handling
        csv_filename = os.path.basename(csv_file.filename)
        csv_path = Path(TRAINVAL_DIR) / csv_filename
        
        logger.info(f"Saving CSV file to {csv_path}")
        csv_content = await csv_file.read()
        with open(csv_path, "wb") as f:
            f.write(csv_content)

        metadata_path = None
        if metadata_file:
            metadata_filename = os.path.basename(metadata_file.filename)
            metadata_path = Path(TRAINVAL_DIR) / metadata_filename
            logger.info(f"Saving metadata file to {metadata_path}")
            metadata_content = await metadata_file.read()
            with open(metadata_path, "wb") as f:
                f.write(metadata_content)

        # Verify files exist before processing
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        # Run prediction
        logger.info(f"Processing experiment data for {task_phase}")
        processing_result = process_experiment_data(csv_path, task_phase)
        if "error" in processing_result:
            raise ValueError(processing_result["error"])

        logger.info("Sending to inference API")
        inference_result = send_to_infer_api(
            str(csv_path),
            str(metadata_path) if metadata_path else None
        )

        drift_result = None

        return JSONResponse({
            "message": "Prediction run successfully.",
            "processing_result": processing_result,
            "inference_result": inference_result,
            "drift_result": drift_result,
            "file_path": str(csv_path)
        })

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return JSONResponse(
            {"error": str(e)}, status_code=500
        )
    finally:
        # Clean up uploaded files if needed
        if 'csv_content' in locals():
            del csv_content
        if 'metadata_content' in locals():
            del metadata_content

def is_file_older_than_one_day(filename):
    """Check if file is older than 1 day based on filename timestamp (YYYYMMDD_HHM)"""
    try:
        # Extract timestamp from first 11 characters (YYYYMMDD_HHM)
        timestamp_str = filename[:11]
        file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
        one_day_ago = datetime.now() - timedelta(days=3)
        return file_date < one_day_ago
    except (ValueError, IndexError):
        # If filename doesn't match expected format, consider it old
        return True
