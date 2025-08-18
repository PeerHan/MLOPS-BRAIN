import os
import logging
import random
import json
import uuid
from datetime import datetime
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from starlette.background import BackgroundTask
import numpy as np
from signal_toolkit import (
    apply_butter_bandpass_filter,
    apply_moving_average,
    apply_stationary
    )

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Get commit tag from environment variable
COMMIT_TAG = os.getenv('GIT_COMMIT_TAG', 'unknown')

# Get long commit hash from environment variable
COMMIT_HASH = os.getenv('GIT_COMMIT_HASH', 'unknown')

# Log the commit tag and hash
logging.info(f"Commit tag: {COMMIT_TAG}")
logging.info(f"Commit hash: {COMMIT_HASH}")

def process_data(df: pd.DataFrame):
    """Process the data using core transformation logic"""
    col_names = ["Channel_1", "Channel_2", "Channel_3", "Channel_4", 
                 "Channel_5", "Channel_6", "Channel_7", "Channel_8"]
    
    df = apply_butter_bandpass_filter(df, col_names)
    df = apply_moving_average(df, col_names, 5)
    df = apply_stationary(df, col_names)
    df.dropna(ignore_index=True, inplace=True)
    
    return df

def split_and_process_data(df: pd.DataFrame, metadata: dict):
    """Split data into parts based on metadata and process each part"""
    interval_type = df["Task_Phase"].iloc[0]
    start_year = metadata["start_year"]
    end_year = metadata["end_year"]
    
    # Generate random date if subject_date_of_birth is missing, invalid, or placeholder
    try:
        if ("subject_date_of_birth" in metadata and 
            metadata["subject_date_of_birth"] != "placeholder" and 
            metadata["subject_date_of_birth"]):
            subject_dob = metadata["subject_date_of_birth"]
            dob = datetime.strptime(subject_dob, "%d-%B-%Y")
        else:
            raise ValueError("Invalid or missing subject_date_of_birth")
    except (ValueError, TypeError):
        logging.info("Using random date due to missing, invalid or placeholder subject_date_of_birth")
        random_year = random.randint(start_year, end_year)
        random_month = random.randint(1, 12)
        random_day = random.randint(1, 28)  # Using 28 to avoid month-specific day limits
        dob = datetime(random_year, random_month, random_day)
        logging.info(f"Generated random date: {dob.strftime('%d-%B-%Y')}")

    time_divisions = {
        "days": 31,
        "months": 12,
        "years": end_year - start_year + 1
    }
    
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

    time_division = time_divisions[interval_type]
    label_target = label_targets[interval_type]
    time_stemps = np.linspace(0, len(df), time_division + 1, dtype=int)
    
    results = []
    for idx in range(len(time_stemps) - 1):
        stemp = time_stemps[idx]
        next_stemp = time_stemps[idx + 1]
            
        # Create explicit copy of the slice
        part = df.iloc[stemp:next_stemp, :].copy()
        # Set label based on whether we have DOB
        label = 1 if (label_targets[interval_type] is not None and idx + 1 == label_targets[interval_type]) else 0
        phase_value = phase_values[interval_type][idx]
        
        # Add metadata columns to the copy
        part["label"] = label
        part["Task_Phase_Value"] = phase_value
        
        # Process the part
        processed_part = process_data(part)
        
        # Generate visualization
        viz_path = generate_visualization(processed_part)
        
        # Prepare CSV output
        csv_output = io.StringIO()
        processed_part.to_csv(csv_output, index=False)
        
        results.append({
            "part_index": idx + 1,
            "label": label,
            "phase_value": phase_value,
            "csv_data": csv_output.getvalue(),
            "visualization": f"/download-viz/{os.path.basename(viz_path)}"
        })
    
    return results

def generate_visualization(data: pd.DataFrame) -> str:
    """Generate visualization and return the file path"""
    y_lim_min = -10
    y_lim_max = 10
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    
    col_names = ["Channel_1", "Channel_2", "Channel_3", "Channel_4", 
                 "Channel_5", "Channel_6", "Channel_7", "Channel_8"]
    
    for col_name in col_names:
        axs.plot(data[col_name].values, color="black")
    
    axs.set_ylim(y_lim_min, y_lim_max)
    axs.grid(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    
    temp_viz_path = f"/tmp/viz_{uuid.uuid4()}.png"
    plt.savefig(temp_viz_path)
    plt.close()
    
    return temp_viz_path

@app.post("/transform/")
async def transform_data(
    csv_file: UploadFile = File(...),
    metadata_file: UploadFile = File(...)
):
    """Transform complete dataset with metadata"""
    try:
        logging.info("Processing data...")
        csv_content = await csv_file.read()
        df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
        metadata_content = await metadata_file.read()
        metadata = json.loads(metadata_content.decode('utf-8'))

        # Log the metadata file content
        logging.info(f"Metadata content: {metadata}")

        parts = split_and_process_data(df, metadata)

        return {
            "commit_tag": COMMIT_TAG,
            "parts": parts
        }
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await csv_file.close()
        await metadata_file.close()

@app.get("/download-viz/{filename}")
async def download_visualization(filename: str):
    file_path = f"/tmp/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="image/png",
            filename="visualization.png",
            headers={"Content-Disposition": "attachment"},
            background=BackgroundTask(lambda: os.remove(file_path))  # Delete after sending
        )
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/version")
async def get_version():
    return {
        "commit_tag": COMMIT_TAG,
        "commit_hash": COMMIT_HASH
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
