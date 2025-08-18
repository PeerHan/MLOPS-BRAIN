import git
import pandas as pd
from dvc.repo import Repo
import shutil
from scipy.stats import ks_2samp
import hashlib
import logging
import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, List
import io
# Read environment variables

commit_tag = os.getenv('GIT_COMMIT_TAG', 'unknown')
brain_repo_token = os.getenv('BRAIN_REPO_TOKEN', 'your-token-here')
dvc_folder = os.getenv('DVCFOLDER', 'data_raw')
git_repo_url = f"https://brain_repo_token:{brain_repo_token}@code.fbi.h-da.de/mlops-brain/brain_{dvc_folder}.git"
aws_creds_dir = "./.aws"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Starting data processing pipeline with commit tag: {commit_tag}")

# Directory setup
repo_dir = "./data"

app = FastAPI()

# Global variable to store latest drift status
current_drift_status = {}

def setup_git_and_dvc():
    # Clean up existing repo directory
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    
    # Clone git repository
    logging.info(f"Cloning repository from {git_repo_url}")
    repo = git.Repo.clone_from(git_repo_url, repo_dir)
    
    # Initialize DVC and pull data
    logging.info("Pulling DVC data")
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = os.path.join(aws_creds_dir, 'credentials')
    dvc_repo = Repo(repo_dir)
    dvc_repo.pull()

def hash_dataframe(df):
    df = df.sort_index(axis=0).sort_index(axis=1)
    df_bytes = df.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(df_bytes).hexdigest()

def detect_datadrift(path, eeg_signatur, window=5):
    cols = [
        "Channel_1", "Channel_2", "Channel_3",	
        "Channel_4", "Channel_5", "Channel_6",
        "Channel_7", "Channel_8"
    ]

    new_data = {
        "days" : pd.read_csv(f"{eeg_signatur}_days.csv"),
        "months" : pd.read_csv(f"{eeg_signatur}_months.csv"),
        "years" : pd.read_csv(f"{eeg_signatur}_years.csv") 
    }

    state_data = {
        "days" : {},
        "months" : {},
        "years" : {},
        "metadata" : {}
    }

    hash_to_time = {}

    for dataset in ["trainval", "test"]:
        for file in os.listdir(f"{path}/{dataset}"):
            file_split = file.split("_")[-1]
            state = file_split.split(".")[0]
            if state == "placeholder" or file.split(".")[-1] != "csv":
                continue
            state_df = pd.read_csv(f"{path}/{dataset}/{file}")
            state_hash = hash_dataframe(state_df)
            state_data[state][state_hash] = state_df
            date = file.split("_")[0]
            date = f"{date[-2:]}.{date[4:-2]}.{date[:4]}"
            date = pd.to_datetime(date, dayfirst=True)
            hash_to_time[state_hash] = date
            new_df_hash = hash_dataframe(new_data[state])
            if state_hash == new_df_hash:
                return {col : 0 for col in cols}

    alpha_val = 0.05 / len(state_data["days"])

    pval_res = {}
    for state in new_data.keys():
        pval_res[state] = {}
        for df_hash in state_data[state].keys():
            pval_res[state][df_hash] = {}
            for channel in cols:
                data = state_data[state][df_hash][channel]
                new_state_data = new_data[state][channel]
                res = ks_2samp(data, new_state_data)
                pval_res[state][df_hash][channel] = res.pvalue

    state_hash_dict = {}
    alpha_counts = {}
    for state in pval_res.keys():
        state_hash_dict[state] = {}
        for hash, feature_dict in pval_res[state].items():
            state_hash_dict[state][hash] = {}
            for feats, pvals in feature_dict.items():
                state_hash_dict[state][hash][feats] = pvals
        state_df = pd.DataFrame(state_hash_dict[state]).T
        melted_df = state_df.melt(var_name="Feature",
                                  value_name="Pval",
                                  ignore_index=False)
        melted_df["Drift Detected"] = melted_df["Pval"] < alpha_val
        melted_df = melted_df.reset_index(names="DF Hash")
        melted_df["Date"] = melted_df["DF Hash"].map(hash_to_time)
        melted_df["Date"]
        melted_df.sort_values('Date').to_csv(f"{state}_datadrift_res.csv", index=False)
        
        alpha_count = {col : 0 for col in cols}

        for feature in cols:
            feature_df = melted_df[melted_df.Feature == feature].sort_values("Date").reset_index(drop=True)
            n = len(feature_df)
            for p in range(0, n - window + 1):
                slide = feature_df.Pval[p:p+window]
                alpha_count[feature] += slide.apply(lambda pval : pval < alpha_val).all()
        alpha_counts[state] = alpha_count
    return alpha_counts

# Initialize the service
setup_git_and_dvc()

@app.get("/status")
async def get_drift_status() -> JSONResponse:
    """Return the current drift status for each channel"""
    if not current_drift_status:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "No drift detection performed yet"}
        )
    
    # Check if any channel has drift
    has_drift = any(status == "Drift Detected" for status in current_drift_status.values())
    
    if has_drift:
        return JSONResponse(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            content=current_drift_status
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "No drift detected"}
        )

async def process_files_background(temp_dir: str, signature: str):
    try:
        channel_drift = detect_datadrift(
            path=f"{repo_dir}/data_raw/",
            eeg_signatur=temp_dir + f"/{signature}"
        )
        
        # Update global status
        global current_drift_status
        current_drift_status = {
            channel: "Drift Detected" if count > 0 else "No Drift"
            for state, count_dict in channel_drift.items()
            for channel, count in count_dict.items()
        }

    except Exception as e:
        logging.error(f"Error in background processing: {str(e)}")
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

@app.post("/update")
async def update_data(
    background_tasks: BackgroundTasks,
    days_file: UploadFile = File(...),
    months_file: UploadFile = File(...),
    years_file: UploadFile = File(...),
    signature: str = "latest"
) -> JSONResponse:
    """Accept new CSV files and start async drift detection"""
    # Save uploaded files
    temp_dir = f"./temp_{signature}"
    os.makedirs(temp_dir, exist_ok=True)
    
    files = {
        "days": days_file,
        "months": months_file,
        "years": years_file
    }
    
    # Save files
    try:
        for file_type, file in files.items():
            content = await file.read()
            filepath = f"{temp_dir}/{signature}_{file_type}.csv"
            with open(filepath, "wb") as f:
                f.write(content)
        
        # Add task to background
        background_tasks.add_task(
            process_files_background,
            temp_dir,
            signature
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "processing",
                "message": "Files received and processing started. Check /status endpoint for results."
            }
        )
    
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(temp_dir)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": str(e)}
        )
