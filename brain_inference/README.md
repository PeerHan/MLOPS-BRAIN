# Brain Inference
This repo contains inference service for `BRAIN` project. It set up an endpoint receiving raw data and returning predictions of a trained model. 

## src
- helpers
- main

# endpoint
`/predict`

* receives raw data from [brain_receiver](https://code.fbi.h-da.de/mlops-brain/brain_receiver)
* transforms raw data via [brain_transform](https://code.fbi.h-da.de/mlops-brain/brain_transform) api 
* predicts with best registered models from [mlflow server](https://mlflow.mlopsbrain.world) (tagged with `{"stage": "Production")`

example curl command:
```
export CSV_FILE=YOUR_CSV_FILE
export METADATA_FILE=YOUR_METADATA_FILE
curl -X POST "https://infer.mlopsbrain.world/predict/" \
-F "csv_file=@$CSV_FILE;type=text/plain" \
-F "metadata_file=@$METADATA_FILE;type=text/plain"
```

# Quickstart
* Define required environment variables in `model.env` like it in `model.env.template`
* Run
```
docker-compose up --build
```