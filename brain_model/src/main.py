import os
import logging
import datetime
import time

import optuna
from torchvision import transforms
import torch
import mlflow
from mlflow.client import MlflowClient
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch_toolkit as ttk
import optuna_toolkit as otk
import experiment_visualizer as evz
from helpers import split_data

# Configure logging for outputs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Retrieve configuration from environment
test_mode = os.getenv("TEST_MODE", "").lower() == "true"
seed = int(os.getenv("SEED", "42"))
trials = int(os.getenv("TRIALS", "1"))
mlflow_server = os.getenv("MLFLOW_SERVER")
transform_api_url = os.getenv("TRANSFORM_API_URL")

def get_transform_hash(max_retries=5, initial_wait=1):
    """Fetch transform hash with exponential backoff retry logic."""
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=2,  # exponential backoff
        status_forcelist=[500, 502, 503, 504],
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    attempt = 0
    while attempt < max_retries:
        try:
            response = session.get(
                transform_api_url+'/version',
                timeout=10
            )
            response.raise_for_status()
            return response.json()["commit_tag"]
        except requests.Timeout:
            wait_time = initial_wait * (2 ** attempt)
            logger.warning(f"Request timed out, retrying in {wait_time} seconds...")
        except requests.RequestException as e:
            wait_time = initial_wait * (2 ** attempt)
            logger.warning(f"Request failed: {e}, retrying in {wait_time} seconds...")
        
        time.sleep(wait_time)
        attempt += 1
    
    raise RuntimeError("Failed to fetch transform hash after maximum retries")

try:
    brain_transform_hash = get_transform_hash()
    logger.info(f"Successfully retrieved transform hash: {brain_transform_hash}")
except RuntimeError as e:
    logger.critical(f"Fatal error: {e}")
    raise

brain_model_hash=os.getenv("GIT_COMMIT_TAG")
brain_dataset_hash=os.getenv("GIT_COMMIT_HASH_DATASET")
logger.info(f"Starting data split with transform hash: {brain_transform_hash}")
data_split_folder = "data_split"
split_data(brain_transform_hash=brain_transform_hash,
           original_data_dir="feature_store",
           output_data_dir=data_split_folder,
           random_state=seed)
logger.info(f"Setting random seed to {seed}")
torch.manual_seed(seed)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(512),
     transforms.Grayscale(),
     ttk.InvertColors(),
     ttk.SamplewiseNormalize()
    ])

# Read DVC hash from feature_store.dvc
dvc_file_path = "./repo/feature_store.dvc"
dvc_hash = None
with open(dvc_file_path, "r", encoding="utf-8") as f:
    for line in f:
        if (line.strip().startswith("md5:")):
            dvc_hash = line.strip().split(": ")[1]
            break

if dvc_hash:
    logger.info(f"DVC hash: {dvc_hash}")
else:
    logger.warning("DVC hash not found in feature_store.dvc")

# Optuna-Studie erstellen
logger.info(f"Connecting to MLflow server at {mlflow_server}")
mlflow.set_tracking_uri(uri=mlflow_server)

start_time = datetime.datetime.now(datetime.UTC).isoformat(timespec="minutes")
logger.info(f"Setting experiment: brain_model_m{brain_model_hash}_t{brain_transform_hash}")
mlflow.set_experiment(f"brain_model_m{brain_model_hash}_t{brain_transform_hash}")
# Besten Trial in MLflow registrieren
logger.info("Starting MLflow experiment")
with mlflow.start_run(run_name=f"experiment_{start_time}") as run:
    logger.info("Initiating Optuna optimization")
    mlflow.log_param("brain_transform_hash", brain_transform_hash)
    mlflow.log_param("brain_model_hash", brain_model_hash)
    mlflow.log_param("dvc_hash", dvc_hash)
    mlflow.log_param("git_dataset_hash", brain_dataset_hash)

    study = optuna.create_study(direction="maximize", study_name="BrainDemo", sampler=optuna.samplers.TPESampler())
    study.optimize(
        lambda trial: otk.cnn_objective(trial, data_split_folder, transform=transform, seed=seed),
        n_trials=trials)
    logger.info("Hyperparameter tuning completed")
    # mlflow.log_params(study.best_params)
    # mlflow.log_param("brain_transform_hash", brain_transform_hash)
    # mlflow.log_param("brain_model_hash", brain_model_hash)

    mlflow.log_metric("best_mcc", study.best_value)
    logger.info("Logging optimization artifacts")
    study.trials_dataframe().to_csv("artifacts.csv", index=False)
    mlflow.log_artifact("artifacts.csv")

    trial_df = study.trials_dataframe()
    trial_df = trial_df.rename({"value" : "MCC"}, axis=1)
    trial_df["Rank"] = trial_df.MCC.transform(evz.match_label)
    artifact_folder = "experiment_summary"
    logging.info("Creating artifact Folder")
    os.makedirs(artifact_folder)
    logging.info("Line Plot")
    evz.line_plot(trial_df, "number", artifact_folder)
    logging.info("Scatter Plots")
    evz.scatter_plot(trial_df, "params_lr", "params_wd", artifact_folder)
    evz.scatter_plot(trial_df, "params_dropout", "params_batch_size", artifact_folder)
    logging.info(trial_df.duration)
    evz.duration_plot(trial_df, artifact_folder)
    logging.info("Boxplots")
    evz.category_plot(trial_df, "params_hidden_dims", artifact_folder)
    evz.category_plot(trial_df, "params_hidden_dim", artifact_folder)
    evz.category_plot(trial_df, "params_kernel_size", artifact_folder)
    evz.category_plot(trial_df, "params_mlp_layers", artifact_folder)
    evz.category_plot(trial_df, "params_cnn_layers", artifact_folder)
    logging.info("Countplot")
    evz.count_plot(trial_df, "state", artifact_folder)
    logging.info("Logging Artifact Folder")
    mlflow.log_artifacts(artifact_folder, artifact_path="experiment_summary")
    series = trial_df.sort_values("MCC", ascending=False).iloc[0, :]
    logger.info("Starting model retraining with best parameters")
    best_weight_path = "Weights/best_trial.pt"
    test_acc, test_mcc, test_res, model = otk.retrain_and_eval(series, best_weight_path, data_split_folder, transform)

    test_res.to_csv("test_res.csv", index=False)
    mlflow.log_artifact("test_res.csv")

    # Create sample input for model signature and convert to numpy array
    sample_input = torch.randn(1, 1, 512, 512).numpy()  # Convert tensor to numpy array

    model_name = f"model_{brain_transform_hash}"
    mlflow.pytorch.log_model(
        model,
        model_name,
        registered_model_name=model_name,
        input_example=sample_input
    )

    # set extra tags on the model
    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = client.get_latest_versions(model_name)[0]
    client.set_model_version_tag(
        name=model_name,
        version=model_info.version,
        key='test_mcc',
        value=test_mcc
    )
    logger.info("Registering model in MLflow")
    # model_uri = f"runs:/{run.info.run_id}/{model_name}"
    # mlflow.register_model(
    #     model_uri, model_name,
    #     tags={
    #         "brain_model_hash": brain_model_hash,
    #         "brain_transform_hash": brain_transform_hash,
    #         test_acc: test_acc,
    #         # test_res: test_res,
    #         test_mcc: test_mcc # model selection criteria
    #     }
    # )
    logger.info("Experiment completed successfully")

    stage, Production, Archived = "stage", "Production", "Archived"
    production_versions_benchmarks = [
        (v.version, float(v.tags["test_mcc"]))
        for v in client.search_model_versions(f"name='{model_name}' and tag.stage='{Production}'")]
    logger.info("Updating model version stage tag...")
    if test_mcc > max([0] + [float(b) for v, b in production_versions_benchmarks]):
        client.set_model_version_tag(
            name=model_name,
            version=model_info.version,
            key=stage,
            value=Production
        )
        for v, b in production_versions_benchmarks:
            client.set_model_version_tag(
                name=model_name,
                version=v,
                key=stage,
                value=Archived
            )
    else:
        client.set_model_version_tag(
            name=model_name,
            version=model_info.version,
            key=stage,
            value=Archived
        )
    logger.info("Update model version stage tag successfully")
