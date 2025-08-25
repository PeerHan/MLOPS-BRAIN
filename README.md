# MLOPS-BRAIN

Welcome to Birthdate Retrieval from Analysis of Inferred Neurodata (BRAIN) - A MLOps System for the prediction of Birthdates based on measured neuro data comming from the [Neurosity Crown](https://neurosity.co/).

# Overview
```mermaid
flowchart LR
    %% === LOCAL MACHINE ===
    subgraph A[Mobile Machine]
        EEG(EEG Headset)
        COLLECTOR([brain_collector<br/>Docker container])
        EEG --> COLLECTOR
    end

    %% === SERVER SIDE ===
    subgraph B[Server]
        RECEIVER([brain_receiver<br/>Docker container])
        DATA_MONITOR([brain_data<br/>Docker container<br/>Monitors input folder])

        COLLECTOR --> RECEIVER
        RECEIVER --> DATA_MONITOR

        RAW_DVC((brain_data_raw<br/>DVC Repo<br/>AWS S3 backend))
        DATA_MONITOR --> RAW_DVC

        %% --- Data Drift Monitoring ---
        BRAIN_DRIFT([brain_drift<br/>Docker container<br/>Monitors data drift])
        RECEIVER --> BRAIN_DRIFT
        BRAIN_DRIFT -->|Detects drift| UPTIME_KUMA([uptime kuma<br/>Docker Container])
        UPTIME_KUMA -->|Sends alert| RESPONSIBLE_PERSON([Responsible Person])

        %% --- Inference Flow ---
        DEPLOY([brain_inference<br/>Docker Container<br/>Serves best available model])
        RECEIVER -->|Uses model for incoming data| DEPLOY

        %% --- Transformation Sub-flow ---
        subgraph TRANSFORM[Data Transformation]
            AIRFLOW([apache_airflow<br/>Docker container<br/>Schedules every hour])
            PIPELINE([brain_transformer_pipeline<br/>Docker container])
            TRANSFORM_API([brain_transform_service<br/>Docker container<br/>API])

            %% Pipeline pulls data from the raw DVC repo
            RAW_DVC -->|Pull new raw data| PIPELINE

            AIRFLOW --> PIPELINE
            PIPELINE --> TRANSFORM_API
            PIPELINE -->|Returns transformed data| TRANSFORM_UPLOADER([brain_data<br/>Docker container<br/>Uploads to feature store])

            FEATURE_DVC((brain_feature_store<br/>DVC Repo<br/>AWS S3 backend))
            TRANSFORM_UPLOADER --> FEATURE_DVC
        end

        %% --- Modeling & Training Sub-flow ---
        subgraph MODELING[Model Training & Serving]
            MODEL([brain_model<br/>Docker container])
            MLFLOW([mlflow<br/>Docker container])

            MODEL -->|Pulls feature data| FEATURE_DVC
            MODEL -->|Logs metrics| MLFLOW
            MLFLOW --> DEPLOY

            %% Show that deployed model uses transform service for inference
            DEPLOY -->|Sends incoming data| TRANSFORM_API
        end
    end

    %% === CI/CD INFRA ===
    subgraph C[CI/CD - GitLab]
        GITLAB([GitLab<br/>Source Code & Actions])
        BRAIN_DAGS([brain_dags<br/>DAG Code])
        BUILDSRV([GitLab Actions<br/>Build Containers])
        REGISTRY([Container Registry])

        GITLAB --> BUILDSRV
        BUILDSRV --> REGISTRY

        %% Airflow DAGs are pushed from GitLab to Airflow
        BRAIN_DAGS -->|Pushes changes to airflow| AIRFLOW
    end
```

## User Guide
To use BRAIN you can use the services hosted on openstack under the domain *mlopsbrain.world*.
To run an experiment, the brain_collector repository must be cloned to a local machine and run as described in its repository.
Additionally a `.env` file must be created containing the following content:
<details>
<summary>Click to show the .env contents</summary>

```
API_URL=https://brnrcvr.mlopsbrain.world/upload/
API_USERNAME="brain_collector"
API_PASSWORD="Bruchteil5-Baumholder"
TIME_INTERVAL=0.5
START_YEAR=1990
END_YEAR=2007
```
</details>
While `START_YEAR` and `END_YEAR` can be adapted as needed, `TIME_INTERVAL` should stay at `0.5` as the experiment does not handle different values correctly and a new s3 bucket would need to be created so the different data would not get mixed up. TIME_INTERVAL is the time, each state (day, month or year) is shown to the subject. 
The next step is to create a new conda environment based on the provided ènvironment.yml` running the following commands inside the repo root directory:
```bash
conda env create -f environment.yml
conda activate brain
``` 

Before actually starting the experiment the Neurosity Crown must be on the same wireless network as the machine running brain_collector. Also, the WLAN must actually be connected to the internet, as the headset will only send data when it is connected to the Neurosity server (which is technically not necessary...).

Once the brain_collector has been started with `streamlit run app.py`, you can start the web application at [localhost:8501](http://localhost:8501/).

Before the experiments starts, make sure you are in an quiet environment and are actually thinking about your date of birth. (*At this time, as most effort went into the infrastructure around this model, the model is not very accurate, due to too few samples and low quality of the samples*.)

After clicking on **Start Animation and Data Capture** the days, then months and finally years will be displayed. 
Don't worry, after the animation for days, months or year is finished the processing in the background takes 30ish to 60ish seconds to complete and the app my seem unresponsive. This is normal, so please be patient.
You can also see the predicted values and the distribution, how much each state (displayed day, month or year) seems to be an "outlier".
Since little effort has been put into the model and the acquisition of qualitative data (quiet environment, concentrated subjects, etc.), the validity of the data is rather low.

In Order to train a new model a new, complete datasample must be recorded and stored to the data_raw dvc bucket. This can be achieved by completing one experiment using the brain_collector webApp and providing your real date of birth at the end. 
If the last step was not completed, no new model can be trained, as the lables for the data are missing.
If a new experiment was recorded, at every full hour airflow schedules our etl pipeline, which, if new data is available, transforms the new data and then, starts the model training DAG, which again trains a new model. Currently, to save time, only 2 trials are done for hyperparameter tuning, so the serverload is low and the it is easier to test the whole MLOps cycle.

For local deployment of the infrastructure with Docker, the brain_infrastructure repository can be consulted, or the repositories of the respective containers can be searched. 

# Data Receiving and Storage
For storage we decided to use AWS S3 buckets for our data.
In total we created 3 buckets, one for the raw data, one for processed data before being converted into an image, and one bucket for our feature store containing the image data, which is used for the model training.

Storing the processed data in this case is not necessary, but we figured, it wouldn't hurt at the moment and might be useful later on.

## Raw Data
Our Raw data bucket has two folders, **test** and *trainval**.

- **test** contains hand selected samples used for model selection following hyperparameter-tuning
- **trainval** contains all other data samples and new samples are stored in this folder automatically

## Processed and Feature Store
Both buckets are structured the same.
The firs level of the folder contains folders with the respective commit version of the transform service git repository in order to be able to backtrack, which code version transformed the data inside this folder.
The second level is analogous to the Raw Data bucket.  
```bash
.
├── 0eab308f # This is the commit hash of the transform pipeline commit the transform container was build on
│   ├── test
│   │   ├── false
│   │   │   ├── 20241212_153340_days_part_10_0eab308f.png
│   │   │   └── 20250114_161342_years_part_9_0eab308f.png
│   │   └── true
│   │       ├── 20241212_153340_days_part_1_0eab308f.png
│   │       ├── 20241212_153340_months_part_1_0eab308f.png
│   │       ├── 20241212_153340_years_part_11_0eab308f.png
│   │       ├── 20250114_161342_days_part_1_0eab308f.png
│   │       ├── 20250114_161342_months_part_1_0eab308f.png
│   │       └── 20250114_161342_years_part_11_0eab308f.png
│   └── trainval
│       ├── false
│       │   └── 20250126_175459_years_part_9_0eab308f.png
│       └── true
│           ├── 20241211_145644_days_part_10_0eab308f.png
│           └── 20250126_175459_years_part_11_0eab308f.png
.
.
.
```
# Feature Engineering and Feature Pipeline

## Data Receiving
We receive a dataframe consisting of 8 Feature-Columns which represents brain streaming data and 2 Metadata-Columns (Timestamp + ID).
This is done 3 times because we want to predict different states: a day, a month and a year.
The resulting dataframes have different shapes since the amount of possibilities differ per state. Days have 31 states, Months have 12 states and Year is variable.
The raw data is received through the neurosity crown and contains the following Shape (*note: the first dimension depends on the `TIME_INTERVAL` variable set in brain_collector at the time the data was recorded. this always needs to be the same for one dataset*):
- Day: 3675 x 10 
- Month: 1536 x 10
- Year: m x 10

Each state dataframe is saved in our DVC Repository (brain_data_raw) along with a metadata.json file. The raw data is saved in a S3 Bucket.

## Data Processing
We generate features through various processings tasks triggered by our Data processing pipeline (brain__transform). Therefore, The feature columns of each state-dataframe is tranformed through different signal-transformation functions to leverage trends and reduce noise. The following functions from our signal\_toolkit.py are used:
1. apply\_butter\_bandpass_filter(df, col\_names, low\_cut=0.5, high\_cut=30, fs=250, order=5):
    - Purpose: Applies a bandpass filter to specific columns in a DataFrame to isolate a desired frequency range.
    - Inputs:
        - df: DataFrame containing time series data.
        - col\_names: List of column names to be filtered.
        - low\_cut, high\_cut: Frequency range for the filter.
        - fs: Sampling frequency of the data.
        - order: Order of the Butterworth filter.
    - Output: A DataFrame with filtered signals for the specified columns.
    - Use Case: Removes unwanted noise or frequencies irrelevant to the analysis.

2. apply\_moving\_average(df, col\_names, window=10, min\_periods=1):
    - Purpose: Computes the moving average for specific columns to smooth out noise.
    - Inputs:
        - df: DataFrame containing time series data.
        - col\_names: List of columns to smooth.
        - window: Number of observations for the moving average window.
        - min\_periods: Minimum number of observations required for a valid result.
    - Output: A DataFrame with smoothed values for the specified columns.
    - Use Case: Reduces fluctuations in noisy signals.

3. apply\_stationary(df, col\_names, periods=1):
    - Purpose: Makes time series data stationary by differencing the values.
    - Inputs:
        - df: DataFrame containing time series data.
        - col_names: List of columns to make stationary.
        - periods: Number of periods to use for differencing.
    - Output: A DataFrame with differenced signals for the specified columns.
    - Use Case: Removes trends and stabilizes the mean.

4. generate\_img(data, label, filename_prefix, subdir, tag_viz_dir):
    - Purpose: Visualizes time series data and saves it as an image.
    - Inputs:
        - data: DataFrame containing time series.
        - label: Binary label (e.g., 1 for true, 0 for false).
        - filename_prefix: Prefix for the saved image filename.
        - subdir: Subdirectory name for saving.
        - tag\_viz\_dir: Base directory for visualization outputs.
    - Output: Saves a plot of the time series for selected channels.

Each state-dataframe is divided in x parts where x depends on the amount of different states per dataframe. For Example, the day-dataframe is divided in 31 subdataframes since each time series should correspond to a displayed day. Each subdataframe is then tranformed consecutive with the previously named functions (function 1 - 3).

Thereby, a state-dataframe is received through the use of the neurosity crown, divided in x subdataframes and transformed with various signal-transformations before these transformed subdataframes are saved as images. These transformations are applied from our Data processing pipeline (brain_data_processed). The resulting artefacts are saved in a S3 Bucket (Processed Dataframes and Images).

## Image Processing
The images contain the multivariate timeseries and must be further processed to enable and leverage the training with a CNN. [Rodrigues et al.](https://arxiv.org/pdf/2102.04179) propose a scaling of the pixel values to a range of [0, 1] followed by a color invertion operation to reduce the activation of feature maps in empty areas. We adopted and extended those transformations in our image processing pipeline which is part of our Model repository (brain_model) since the transformations are applied while model-training. Therefore the image processing contains the following transformations:

1. transforms.ToTensor():
    - Purpose: Converts a PIL image or NumPy array into a PyTorch tensor.
    - Scales pixel values from [0, 255] (integers) to [0.0, 1.0] (floating point).
    - Use Case: Necessary for inputting image data into a PyTorch model.
2. transforms.Resize(512)
    - Purpose: Resizes the input image so that the smaller edge is 512 pixels, maintaining the aspect ratio.
3. transforms.Grayscale()
    - Purpose: Converts an image to grayscale (single channel) because our model does not require rgb information.
4. InvertColors()
    - Purpose: Inverts pixel values (e.g., 1.0 becomes 0.0 and vice versa) to reduce the activation of feature maps in empty ares
5. SamplewiseNormalize()
    - Purpose: Normalizes image data on a per-sample basis.
    - Scales the pixel values to have a mean of 0 and a standard deviation of 1, per image.


## Summary
We receive the brain data as a multivariate Timeseries (8 Channels) which is divided into x subdataframes dependend on the amount of different realisations of the state. Those subdataframes are processed with various signal-transformations (butter-bandpass-filter, moving-average, stationary) and then saved as images, capturing the information of the timeseries. Those images are then further processed (scaled, resized, converted to grayscale, inverted, normalized) on runtime of the model-training.

# Experiment Tracking and Model Registry
We use brain_model for the birthdate prediction.
Since we featurize the batched multivariate timeseries as images we use a 2D-CNN Architecture for a binary classification problem: Is the given Image (capturing the multivariate timeseries for a given timestamp) an outlier or not? An outlier is defined by the biggest logit output within a state.

## Hyperparameter Tuning
Therefore, Grid and Random search approaches could lead to unefficient hyperparameter tuning results because neural networks are notorious for large hyper parameter spaces. For that reason, we use a [bayesian hyperparameter optimizer](https://de.wikipedia.org/wiki/Parzen-Tree_Estimator) which uses a-prior information to find the best set of hyperparameters for a given validation set.

Tuned hyparameter involves:
- Parameters which are no part of the Model:
    - batch_size: Batchsize of the trainingset
    - learning_rate: Initial learning rate of the model
    - weight_decay: Regularization term to avoid overfitting
- Parameters which define the Model:
    - dropout_rate: Amount of dropouts per layer
    - hidden_dim: Amount of hidden dims in a linear layer for the Feedforward Network
    - hidden_dims_flattened: Amount of hidden dims for the first Feedforward Layer (after the flattening of the feature maps)
    - kernel_size: Dimension of the kernel for CNN Layers
    - mlp_layers: Amount of Feedforward layers
    - start_feature_maps: Amount of feature maps for the first CNN Layer (each consecutive layer has the doubled amount of feature maps)

## Experiment Tracking and Model registry
We use mlflow to track hyperparameter, experimentparameter and metrics for various experiments. For this, each experiment is tracked as nested runs. Each nested run is one trained model involving hyperparameters and metrics of a run. The outer run summs up the whole experiment capturing images and other artifacts as experiment-summary and a model registry including the best model and its metadata based on the validation metrics.

Besides the hyperparameter from above we track the following parameters:
- Optimizer: The optimizer is not changing and is per default the AdamW optimizer
- Random State: Random state for reproducibility 

Each hyperparameter run involves the following metrics based on the validation set:
- final_val_acc: The accuracy score for the validation set
- final_val_f1: The F1 score for the validation set
- final_val_mcc: The matthews correlation coefficient for the validation set
- train_loss: The last training loss (Binary Cross Entropy)
- val_accuracy: The last validation accuracy (last epoch)
- val_loss: The last validation loss (Binary Cross Entropy)

Parameter per Parent Run:
- brain_model_hash
- brain_transform_hash
- dvc_hash
- git_dataset_hash

Other Artifacts per Parent Run:
- Summary of the Experiments: Various Images which summ up the experiment
    - E.G. The MCC Validation Value over Time:
      <img width="1281" height="701" alt="image" src="https://github.com/user-attachments/assets/17c79683-27c8-4371-8776-80d5fec00d66" />

- artifacts.csv: Data of all runs of the experiment involving the hyperparameters, metadata and validation results per run
- test_res.csv: Dataframe of the validation predictions of the best model. Involves Original Label, Predicted Label, Probability and Logits


Model Registry:
- model weights
- requirements
- conda.yml
- serving input schema

The brain_inference repository contains the code which serves the model the latest model from our model registry.

# Model selection`
Models are registered with the name `model_{brain_transform_hash}`and are chronologically versioned with `Version 1`, `Version 2`etc.. Tracing back the Model to the Experiment it was created during, one can finde the Prameter `brain_transform_hash` which contains the version of brain_transform that generated features used to train that specific model. Additionally the `git_dataset_hash` parameter tells us the exact dataset commit inside the brain_feature_store Repository
When the model starts, it looks for the currently deployed transform service version by checking the `brain_transform_hash` attribute from [transform.mlopsbrain.world/version](https://transform.mlopsbrain.world/version), and looks for the best version of the model compatible with the `brain_transform_hash`, and then serves the best version through `/predict` API.
The container only does this once it is started, so the container needs to be restarted every once in a while, e.g. in the night, in order to server the best available modek, which can be realised via cron schedules

# `/predict` API
Receives a csv file and a json file generated from our EEG data collecting point (brain_receiver). Send POST request to [transform API](https://transform.mlopsbrain.world/transform) for features (pictures). Feeds features as input of the model and then returns result to client.

Curl command example:
```
export CSV_FILE=YOUR_CSV_FILE
export METADATA_FILE=YOUR_METADATA_FILE
curl -X POST "https://infer.mlopsbrain.world/predict/" \
-F "csv_file=@$CSV_FILE;type=text/plain" \
-F "metadata_file=@$METADATA_FILE;type=text/plain"
```
Response:
```
{"predictions": LIST_OF_MODEL_PREDICTIONS_FOR_PICTURES}
```

# Orchestration
For orchestration of our services we decided to use [apache airflow](https://airflow.apache.org) for job scheduling and [watchtower](https://containrrr.dev/watchtower/) to keep the deployed services up to date with the latest versions of out docker containers.

## [Docker](https://www.docker.com)
In generell we deploy all our code, besides the brain_collector, inside a docker container which makes it easy to deploy and makes dependency problems less an issue, as only docker needs to be installed on the host system to get everything up and running.

## [Airflow](https://airflow.apache.org)
Airflow it self is hosted as documented inside the brain_infrastructure repository inside the airflow directory.
The scheduled jobs, called Directed Acyclic Graphs (DAGs), are defined as python scripts which are stored inside the brain_dags repo and actively pushed to the server via gitlab actions.

Our instance is available under [airflow.mlopsbrain.world](https://airflow.mlopsbrain.world)

## [Watchtower](https://containrrr.dev/watchtower/)
Watchtower simply checks if any docker container tagged with `com.centurylinklabs.watchtower.enable=true` is outdated and respectively pulls new images and restarts the service.
To avoid service disruptions this is only done once every night.
It is configured inside the brain_infrastructure repository

## Cron Jobs
Additionally we have one cron job which is deployed via the brain_cron repository.
It restarts the brain_inference container every night, so every night the best available model is pulled and deployed. this is also timed in a manner, that it is executed some while, after potentially the brain_transform service container was updated from watchtower, so that it is very likely that a new model is available to host after an transform pipeline upgrade.
This means though, that between 12:00am and approximately 6:00am the service might not be available or working as expected, which is outside the times of the day the service is assumed to be used anyways.

# Datadrift detection with Brain Monitor
Our brain_monitor detects data drifts between the distribution of new datapoints and our data. This is done univariat for each of our 8 neurosity features (Channel 1 - Channel 8). 

Suppose BRAIN gets data of a new Person, the following schedule is started:
1. Save the incoming neurosity signals resulting in 3 Dataframes (n x 8 Features) for each state (day, month, year).
2. For each Dataframe do:

    <img width="1958" height="1456" alt="image" src="https://github.com/user-attachments/assets/55f73bb3-1931-46e0-8115-4f7776f91f47" />


So for each channel for each state dataframe we get the amount of significant differences between the channel distributions based on the kolmogorov-smirnoff test. We use a sliding window to capture the amount of significant differences over a period - if all observations within the period are significantly different distributed we increment the alpha-counter. The pvalue is corrected with the bonferroni correction since we have a multiple testing problem.

This counter is returned and can give insight in which time period a datadrift occured.  

# Drift Monitoring Deployment
Our Data Drift Monitor can be accessed via [up.mlopsbrain.world/status/drift](https://https://up.mlopsbrain.world/status/drift).
Uptime Kuma classically is a service Status monitor which mostly differentiates between if a service is up and down.
We define, that down means, there is a data drift, and up means, no data drift is apparent at the moment.
If the state changes a notification is send into our groups element group using a bot-matrix account.
The website looks like this:
<img width="2820" height="834" alt="image" src="https://github.com/user-attachments/assets/50c2c6a2-1256-4646-aad4-841789a7dbb7" />
