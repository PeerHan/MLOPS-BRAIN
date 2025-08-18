# Brain Model
This repository contains code for model training and hyper parameter optimizing with a bayesian tuning schedule. The optimizing process is tracked with mlflow as experiment tracking framework. The main objective is to find the right CNN architecture and hyperparams which will be used to register the model.

# `stage` tag of model versions
After training, the model will be registered with `stage` tag:
* `stage: Production` if it is the best version (i.e. with highest `test_mcc`).
* `stage: Archived` otherwise.

The model version tagged with `Production` will be served by [brain_inference](https://code.fbi.h-da.de/mlops-brain/brain_inference)

The model versions tagged with `Archived` will be manually removed if necessary. 

## src
- entrypoint.sh 
- optun_toolkit - Optuna Functions to make bayesian optimization with mlflow logging:
    - optuna_cnn: Creates a CNN based on the bayesian hyper parameter tuning
    - cnn_objective: Creates and Trains a CNN based on training and validation sets to find the best hyper parmameter
    - create_cnn_from_trial: Recreates a CNN Based on the best trial
    - retrain_and_eval: Recreates the best CNN based on the experiment and evaluates on the test_set
- torch_toolkit - PyTorch Functions to make model training easier:
    - SamplewiseNormalize: Custom PyTorch Image Transformation to normalize an image
    - InvertColors: Custom PyTorch Image Transformation to invert a grey pixel image
    - EarlyStopper: PyTorch Early Stopper to stop training if no positive changes occur for training loss 
    - train_one_epoch: Training a PyTorch Model for one epoch based on a train loader
    - validate_one_epoch: Validating a PyTorch Model for one epoch based on a validation loader
    - train_and_validate_model: Combine train_one_epoch, validate_one_epoch, EarlyStopper and Schedulers to train a PyTorch model
    - test_model: Test a PyTorch Model for a test_loader
- experiment_visualizer: Various plotting functions to sum up a hyper parameter tuning experiment through visualizing the tuned params in dependence of the validation metric. Each Plot will be saved as an image in an artifact folder which will be logged.
    - match_label: Discretize the Validation MCC into Quantiles
    - line_plot: Create a lineplot to visualize the validation MCC over time
    - scatter_plot: Create a scatterplot to visualize 2 continuous hyper params and color for the MCC quantiles
    - duration_plot: Creates KDE plots for the duration per trials and color for the MCC quantiles
    - category_plot: Creates Boxplots for discrete hyper params and color for the MCC quantiles
    - count_plot: Creates Countplots for discrete hyper params and color for the MCC quantiles
- helpers: Helper Functions like validation splitting or loading of test set
    - split_data: Splits data and fills Train/Val/Test folder in a desired folder structure
- main: Start Hyper parameter Tuning and Model Registry. This is the toplevel script to create a CNN based on the best hyperparams and do n_trial runs for a experiment. The best model (based on the validation MCC) will be used for the model registry. Each Experiment is summed up with various artifacts.

## additional_scripts
- model_trigger: Trigger for training automation. A model is automatically trained when new data has appeared.

## tests
- test_optunatoolkit: Script for Testing self written optuna functions to create, recreate and tune models
    - test_optuna_model_creation: Checks if the output of an optuna trial gives the expected results
    - test_optuna_recreation: Checks if the optuna_recreation function works - Can a model be build based on a trial
- test_torchtoolkit: Script for Testing various functions from torch_toolkit.py like training/validation/test methods or custom pytorch transformations for imgs:
    - test_transformation: Checks if the custom transformation functions create the expected aggregation values
    - test_model_train_and_test: Dummy Test for a Dummy Model for the train/test Functions. Checks if the model weights are changing and the metrics are meaningfull.

## Quickstart
* Define required environment variables in `model.env` like it in `model.env.template`
* Run
```
docker-compose up --build
```