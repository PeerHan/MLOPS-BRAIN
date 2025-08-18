# Brain Transform
This repository contains a data processing pipeline for signal data. The main objective is to process raw data files, apply signal processing techniques, and generate visualizations.

# Repository Structure
main.py: The main script that orchestrates the data processing pipeline.
signal_toolkit.py: Contains functions for signal processing, such as filtering and scaling.
data: Directory containing the raw input data files.
etl_processed: Directory where processed data files are stored.
etl_finished: Directory where visualization outputs are saved.
Dockerfile: Instructions to build the Docker image for the application.
compose.yml: Docker Compose file for setting up the application services.
.gitlab-ci.yml: Configuration for the GitLab CI/CD pipeline.
requirements.txt: Python dependencies for the project.
## main.py Overview
The main.py script performs the following tasks:

Environment Setup: Reads the commit tag from the environment variable GIT_COMMIT_TAG, sets up logging, and defines directories for input, processed data, and visualizations.

Directory Preparation: Ensures that all necessary directories exist for storing processed data and visualizations.

Data Processing Functions:

find_complete_samples(directory): Scans the input directory and identifies complete data samples based on predefined naming patterns and the presence of required files.
process_csv(file_path, time_division, label_target, phase_values): Processes individual CSV files by applying signal processing techniques and prepares them for analysis.
Main Execution Flow: Coordinates the processing of data samples by reading raw data, applying transformations using functions from signal_toolkit.py, and saving the processed data and visualizations to the respective directories.

## Signal Processing
The processing pipeline applies several signal processing techniques, including:

Bandpass Filtering: Removes unwanted frequencies from the signal data using a Butterworth bandpass filter.
Moving Average: Smooths the data by calculating the moving average over a specified window.
Differencing: Makes the data stationary by calculating differences between consecutive data points.
Scaling: Standardizes the data using scaling methods from sklearn.
These functions are implemented in signal_toolkit.py.

## Running the Application
To run the application using Docker and Docker Compose:

This command builds the Docker image defined in the Dockerfile and starts the service as defined in compose.yml.

## Continuous Integration and Deployment
The project uses GitLab CI/CD, configured in .gitlab-ci.yml, to automatically build and register the Docker container when changes are pushed to the main branch.

## Dependencies
All Python dependencies are listed in requirements.txt and are installed during the Docker image build process.

## Notes
Ensure that the input data files are placed in the data directory.
Processed data and visualizations will be saved in the etl_processed and etl_finished directories, respectively.

## Quickstart