import os
import sys
import logging
import requests
import mlflow
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from helpers import transform
from mlflow.tracking import MlflowClient
import shutil

api_base_url = os.getenv('API_URL')
mlflow_server = os.getenv("MLFLOW_SERVER")
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

port = int(os.getenv('PORT', '8000'))
viz_dir = "./viz"
logging.basicConfig(level=logging.INFO)

# Get commit_tag from environment or API
commit_tag = os.getenv('COMMIT_TAG')
if commit_tag is None:
    try:
        response = requests.get(f"{api_base_url}/version")
        response.raise_for_status()
        commit_tag = response.json()["commit_tag"]
        logging.info(f"Using API commit tag: {commit_tag}")
    except Exception as e:
        logging.error(f"Failed to get API version: {str(e)}")
        raise

app = FastAPI()

mlflow.set_tracking_uri(uri=mlflow_server)

logging.basicConfig(level=logging.INFO)

model_name = f"model_{commit_tag}"
client = MlflowClient()
model_select_criterion = "test_mcc"
versions_benchmarks = [
    (v.version, float(v.tags[model_select_criterion]))
    for v in client.search_model_versions(f"name='{model_name}' and tag.stage = 'Production'")]

if len(versions_benchmarks) > 0:
    best_version, benchmark = sorted(versions_benchmarks, key=lambda x: x[1], reverse=True)[0]
    best_model = f"models:/{model_name}/{best_version}"
    logging.info(f"Loading model: {best_model} with {model_select_criterion}: {benchmark}")
    loaded_model = mlflow.pytorch.load_model(best_model)
else:
    sys.exit("No model tagged with 'Production' stage, exiting...")

def preprocess(
    csv_file,
    metadata_file,
):
    files = {
        'csv_file': csv_file,
        'metadata_file': metadata_file
    }

    response = requests.post(f"{api_base_url}/transform/", files=files)
    response.raise_for_status()

    return response

@app.post("/predict/")
async def predict(
    csv_file: UploadFile = File(...),
    metadata_file: UploadFile = File(...)
):
    try:
        logging.info(f"received prediction request, preprocessing...")
        logging.info(f"csv file: {csv_file.filename}")
        logging.info(f"metadata file: {metadata_file.filename}")
        preprocess_response = preprocess(await csv_file.read(), await metadata_file.read())

        logging.info(f"preprocessing complete, predicting...")
        for part in preprocess_response.json()['parts']:
            part_suffix = f"_part_{part['part_index']}_{commit_tag}"
            viz_url = f"{api_base_url}{part['visualization']}"
            viz_response = requests.get(viz_url)
            viz_response.raise_for_status()

            viz_output_path = os.path.join(viz_dir, "inference", f"{part_suffix}.jpg")
            os.makedirs(os.path.dirname(viz_output_path), exist_ok=True)

            with open(viz_output_path, 'wb') as f:
                f.write(viz_response.content)

            inf_set = ImageFolder(viz_dir, transform=transform)

            inf_loader = DataLoader(inf_set, batch_size=1, shuffle=False)

            results = []

            for data in inf_loader:
                inputs, labels = data
                results.append(loaded_model(inputs).data.item())

        logging.info(f"predicting complete, returning results...")

        return {"predictions": results}

    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.isdir(viz_dir):
            shutil.rmtree(viz_dir)
        await csv_file.close()
        await metadata_file.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
