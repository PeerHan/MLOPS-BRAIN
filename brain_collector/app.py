import streamlit as st
from pylsl import StreamInlet, resolve_stream
import csv
import json
from datetime import datetime
import time
import threading
import logging
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import os
import pandas as pd

# Setup logging
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("eeg_logger.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

# Retrieve configuration from environment
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/upload/")
API_USERNAME = os.getenv("API_USERNAME", "user")
API_PASSWORD = os.getenv("API_PASSWORD", "password")
time_interval = float(os.getenv("TIME_INTERVAL", "0.1"))
start_year = int(os.getenv("START_YEAR", "1999"))
end_year = int(os.getenv("END_YEAR", "2007"))
# Utility function to save metadata to a JSON file
def save_metadata(metadata, metadata_file):
    try:
        with open(metadata_file, "w") as json_file:
            json.dump(metadata, json_file)
        logging.info(f"Metadata saved to {metadata_file}.")
    except Exception as e:
        logging.error(f"Error saving metadata: {e}")

pred_year=1999
pred_month='March'
pred_day=1

# General function to upload files to an API
def send_files_to_api(csv_file, metadata_file="placeholder.json", task_phase="meta", api_url=API_URL):
    """
    Uploads files and metadata to a specified API endpoint.
    
    :param csv_file: Path to the CSV file to upload.
    :param metadata_file: Path to the JSON metadata file (if not provided, sends a placeholder).
    :param task_phase: Task phase identifier (e.g., 'days', 'months', 'years') (optional for metadata).
    :param api_url: API endpoint URL.
    :return: API response as a dictionary or None if an error occurs.
    """
    try:
        # Create default metadata if using placeholder
        if metadata_file == "placeholder.json":
            default_metadata = {
                "start_year": start_year,
                "end_year": end_year,
                "time_interval": time_interval,
                "device_name": "Neurosity Crown"
            }
            with open(metadata_file, 'w') as f:
                json.dump(default_metadata, f)

        files = {
            "csv_file": (csv_file, open(csv_file, 'rb'), "text/csv"),
            "metadata_file": (metadata_file, open(metadata_file, 'rb'), "application/json")
        }
        data = {}
        
        if task_phase:
            data["task_phase"] = task_phase

        response = requests.post(
            api_url,
            files=files,
            data=data,
            auth=HTTPBasicAuth(API_USERNAME, API_PASSWORD)  # Replace with your credentials
        )

        if response.status_code == 200:
            return response.json()
        else:
            if task_phase == "meta":
                logging.info(f"Ignoring metadata submission error: status code {response.status_code}: {response.text}")
                return None
            else:
                logging.error(f"API request failed with status code {response.status_code}: {response.text}")
                st.error(f"API request failed with status code {response.status_code}: {response.text}")
                return None
    except Exception as e:
        if task_phase == "meta":
            logging.info(f"Ignoring metadata submission exception: {e}")
            return None
        else:
            logging.error(f"Error sending files to API: {e}")
            st.error(f"Error sending files to API: {e}")
            return None


# Function to process experiment data
def get_max_prediction_index(predictions):
    """Returns the index of the maximum value in the predictions list"""
    return predictions.index(max(predictions))

def map_day_index(index):
    """Maps index to day (1-31)"""
    return index + 1  # Since days are 1-based

def map_month_index(index):
    """Maps index to month name"""
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return months[index]

def map_year_index(index):
    """Maps index to year based on start_year"""
    return start_year + index

def process_experiment_data(csv_file, task_phase, api_url=API_URL):
    """
    Processes EEG experiment data by sending it to the API.
    """
    global pred_day, pred_month, pred_year
    result = send_files_to_api(csv_file, task_phase=task_phase, api_url=api_url)
    
    # Log and guard against a None or unexpected result
    if result is None:
        logging.error("No response received from send_files_to_api. Aborting process_experiment_data.")
        st.error("No response received from the API.")
        return None
    if not isinstance(result, dict):
        logging.error(f"API response is not a dictionary: {result}")
        st.error("Unexpected API response structure received.")
        return None

    # Validate inference_result is present and is a dictionary
    inference = result.get("inference_result")
    if not inference or not isinstance(inference, dict):
        logging.error(f"Invalid or missing 'inference_result' in API response: {result}")
        st.error("Unexpected API response structure received.")
        return None
    if "predictions" not in inference:
        logging.error(f"Missing 'predictions' in inference_result: {result}")
        st.error("Unexpected API response structure received.")
        return None

    predictions = inference["predictions"]
    logging.info(f"Received predictions: {predictions}")
    
    # Validate predictions are a non-empty list
    if not isinstance(predictions, list) or not predictions:
        logging.error(f"Invalid predictions received: {predictions}")
        st.error("Invalid predictions received from the API.")
        return None

    # Wrap get_max_prediction_index to catch any errors (optional)
    try:
        max_index = get_max_prediction_index(predictions)
    except Exception as e:
        logging.error(f"Error computing maximum prediction index: {e}")
        st.error("Error processing predictions from the API.")
        return None

    if task_phase == "days":
        pred_day = map_day_index(max_index)
    elif task_phase == "months":
        pred_month = map_month_index(max_index)
    elif task_phase == "years":
        pred_year = map_year_index(max_index)
    
    display_predictions(task_phase, predictions)
    logging.info(f"Updated predictions: Day={pred_day}, Month={pred_month}, Year={pred_year}")
    return max_index

def display_predictions(task_phase, predictions):
    # Use a persistent placeholder to replace previous plots rather than appending.
    if "predictions_placeholder" not in st.session_state:
        st.session_state["predictions_placeholder"] = st.empty()
    placeholder = st.session_state["predictions_placeholder"]
    placeholder.empty()
    with placeholder.container():
        st.subheader("Current Predictions")
        if task_phase == "days":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Day", pred_day)
        elif task_phase == "months":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Day", pred_day)
            with col2:
                st.metric("Predicted Month", pred_month)
        elif task_phase == "years":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Day", pred_day)
            with col2:
                st.metric("Predicted Month", pred_month)
            with col3:
                st.metric("Predicted Year", pred_year)

        chart_data = pd.DataFrame(predictions)
        st.line_chart(chart_data)
        
        max_index = get_max_prediction_index(predictions)
        st.info(f"Strongest prediction at index {max_index} with value {predictions[max_index]:.4f}")

# Function to record EEG data
def record_eeg_data(stop_event, csv_file, task_phase):
    try:
        for _ in range(5):  # Retry logic
            streams = resolve_stream('type', 'EEG')
            if streams:
                break
            time.sleep(2)
        else:
            st.error("No EEG streams found. Ensure the EEG device is connected and try again.")
            logging.error("No EEG streams found.")
            return

        inlet = StreamInlet(streams[0])
        sample, _ = inlet.pull_sample()
        channel_count = len(sample)

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Task_Phase", "Timestamp"] + [f"Channel_{i+1}" for i in range(channel_count)])

            while not stop_event.is_set():
                sample, timestamp = inlet.pull_sample()
                if len(sample) == channel_count:
                    writer.writerow([task_phase, timestamp] + sample)
        logging.info(f"EEG data recording completed for {task_phase}.")
    except Exception as e:
        st.error(f"An error occurred during EEG recording: {e}")
        logging.error(f"EEG recording error: {e}")

# Function to run animations
def run_animation(task_phase, stop_event, csv_file, placeholder, iterable):
    try:
        stop_event.clear()
        eeg_thread = threading.Thread(target=record_eeg_data, args=(stop_event, csv_file, task_phase))
        eeg_thread.start()

        for item in iterable:
            placeholder.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; font-size: 50px; font-weight: bold;">
                    <div style="border: 3px solid #000; padding: 10px; margin: 5px; width: 100px; text-align: center;">{item[0]}</div>
                    <div style="font-size: 40px; margin: 0 10px;">-</div>
                    <div style="border: 3px solid #000; padding: 10px; margin: 5px; width: 275px; text-align: center;">{item[1]}</div>
                    <div style="font-size: 40px; margin: 0 10px;">-</div>
                    <div style="border: 3px solid #000; padding: 10px; margin: 5px; width: 130px; text-align: center;">{item[2]}</div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(time_interval)

        stop_event.set()
        eeg_thread.join()
    except Exception as e:
        st.error(f"Error during animation: {e}")
        logging.error(f"Animation error: {e}")

# Animation manager
def animate_dates():
    global pred_day, pred_month, pred_year  # Explicitly use global variables
    placeholder = st.empty()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_files = {
        "days": f"output/{timestamp}_days.csv",
        "months": f"output/{timestamp}_months.csv",
        "years": f"output/{timestamp}_years.csv",
    }
    stop_event = threading.Event()

    metadata = {
        "start_year": start_year,
        "end_year": end_year,
        "time_interval": time_interval,
        "files": csv_files,
        "device_name": "Neurosity Crown",
        "notes": "",
    }
    metadata_file = f"output/{timestamp}_metadata.json"
    save_metadata(metadata, metadata_file)
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    # Days animation
    days_iterable = [(str(day).zfill(2), "January", str(start_year)) for day in range(1, 32)]
    run_animation("days", stop_event, csv_files["days"], placeholder, days_iterable)
    process_experiment_data(csv_files["days"], "days")

    # Months animation using predicted day
    months_iterable = [(str(pred_day).zfill(2), month, str(start_year)) for month in months]
    run_animation("months", stop_event, csv_files["months"], placeholder, months_iterable)
    process_experiment_data(csv_files["months"], "months")

    # Years animation using predicted day and month
    years_iterable = [(str(pred_day).zfill(2), pred_month, str(year)) for year in range(start_year, end_year + 1)]
    run_animation("years", stop_event, csv_files["years"], placeholder, years_iterable)
    process_experiment_data(csv_files["years"], "years")

    # Ensure final display shows all predictions correctly
    # display_predictions("years", [0] * (end_year - start_year + 1))  # Dummy predictions to trigger final display

    return csv_files, metadata_file, metadata

# Form for entering date of birth
def render_dob_form(metadata_file, metadata):
    with st.form("dob_form"):
        st.write("Please enter the subject's date of birth:")
        dob = st.date_input("Date of Birth", value=datetime(2000, 1, 1))
        submitted = st.form_submit_button("Submit")

        if submitted:
            metadata["subject_date_of_birth"] = dob.strftime("%d-%B-%Y")
            save_metadata(metadata, metadata_file)
            send_files_to_api("placeholder.csv", metadata_file=metadata_file, task_phase='meta')
            # No response is expected after form submission.

# Main Streamlit app
def main():
    st.title("EEG Data Logger with Date Animation")
    st.info("Welcome to the EEG Data Logger! Follow the steps below to record your data.")
    st.sidebar.title("Session Progress")
    st.sidebar.info("Start by capturing EEG data with animations.")

    if "experiment_in_progress" not in st.session_state:
        st.session_state["experiment_in_progress"] = False
    if "metadata" not in st.session_state:
        st.session_state["metadata"] = None

    if st.button("Start Animation and Data Capture", disabled=st.session_state["experiment_in_progress"]):
        if not st.session_state["experiment_in_progress"]:
            st.session_state["experiment_in_progress"] = True
            try:
                csv_files, metadata_file, metadata = animate_dates()
                st.session_state["metadata_file"] = metadata_file
                st.session_state["metadata"] = metadata
                st.sidebar.success("EEG data capture completed.")
                st.success("EEG data capture completed.")
            finally:
                st.session_state["experiment_in_progress"] = False

    if st.session_state.get("metadata") is not None:
        render_dob_form(st.session_state["metadata_file"], st.session_state["metadata"])
        st.sidebar.success("Metadata updated with date of birth.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Critical application error:")
        st.error(f"A critical error occurred: {e}")
