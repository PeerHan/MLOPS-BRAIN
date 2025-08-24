# MLflow Installation
Mlflow is our machine learning tracking server and model registry. 
In order to run it you need to create a _db.env_ file and fill in the needed variables. 
additionally the database credentials need to be inserted into the compose.yml file for the mlflow service.
To run the server simply run `docker-compose up --build -d`.
Below are some examles how to track to the server.
Credentials are setup inside nginx proxy manager, or if not, are not needed.


## How to Track to the Server

MLflow is secured using HTTP basic authentication. To access the server from within Python code, see the following minimal example:

```python
import mlflow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set MLflow tracking URI
mlflow.set_tracking_uri(uri="https://mlflow.mlopsbrain.world/")

# Set environment variables for MLflow tracking
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Set MLflow experiment
experiment_name = "example_experiment"
mlflow.set_experiment(experiment_name=experiment_name)

# Start the experiment run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param('param1', 5)
    mlflow.log_param('param2', 10)

    # Log metrics
    mlflow.log_metric('metric1', 0.85)
    mlflow.log_metric('metric2', 0.95)

    # Set tags
    mlflow.set_tag("example_tag", "example_value")

# End the MLflow experiment run
# (Note: The 'with' statement automatically ends the run)
```

Ensure your `.env` file contains the following:

```bash
MLFLOW_TRACKING_USERNAME=username
MLFLOW_TRACKING_PASSWORD=password
```

## Troubleshooting
### mlflow.exceptions.MlflowException: Cannot set a deleted experiment
If you encounter this error message simply delete the `.trash` folder inside the mlruns folder with:
```bash
sudo rm -rf mlruns/.trash
```