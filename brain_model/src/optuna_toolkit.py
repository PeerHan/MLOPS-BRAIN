"""
OptunaToolkit
Functions for Bayesian HP Tuning
"""


import os

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
import mlflow

import torch_toolkit as ttk

def optuna_cnn(trial):

    cnn_layers = trial.suggest_int("cnn_layers", 1, 3)
    feature_maps = trial.suggest_int("start_fm", 2, 16, step=1)
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    layers = []
    init_feature_maps = feature_maps
    in_channels = 1
    for _ in range(cnn_layers):
        layers.append(nn.Conv2d(in_channels, feature_maps, kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = feature_maps
        feature_maps *= 2

    layers.append(nn.Flatten())

    mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
    hidden_dims_flattened = trial.suggest_int("hidden_dims", 100, 500, step=100)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 250, step=50)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)

    layers.append(nn.LazyLinear(hidden_dims_flattened))
    input_dim = hidden_dims_flattened
    for _ in range(mlp_layers):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Mish())
        layers.append(nn.Dropout(p=dropout_rate))
        input_dim = hidden_dim

    layers.append(nn.Linear(hidden_dim, 1))
    mlflow.log_param("cnn_layers", cnn_layers)
    mlflow.log_param("start_feature_maps", init_feature_maps)
    mlflow.log_param("kernel_size", kernel_size)
    mlflow.log_param("mlp_layers", mlp_layers)
    mlflow.log_param("hidden_dims_flattened", hidden_dims_flattened)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("dropout_rate", dropout_rate)

    return nn.Sequential(*layers)

def cnn_objective(trial, folder_path, transform, seed):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):  # Verschachtelter Run für jeden Trial
        try:
            train_set = ImageFolder(f"{folder_path}/Train", transform=transform)
            val_set = ImageFolder(f"{folder_path}/Val", transform=transform)

            batch_size = trial.suggest_int("batch_size", 2, 64, step=2)

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            
            epochs = 500

            model = optuna_cnn(trial)
            criterion = nn.BCEWithLogitsLoss()

            lr = trial.suggest_float("lr", 0.0005, 0.01)
            wd = trial.suggest_float("wd", 1e-10, 1e-7, log=True)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            _, _, _, state_dict = ttk.train_and_validate_model(
                model, train_loader, val_loader, criterion, optimizer, epochs, show_every=1
            )
            
            mean_acc, mean_mcc, mean_f1, _ = ttk.test_model(model, val_loader)
            if not os.path.exists("Weights"):
                os.makedirs("Weights")

            # Save only the best model
            if trial.number == 0 or mean_mcc > trial.study.best_value:
                torch.save(state_dict, "Weights/best_trial.pt")
            
            # Already logged in train_and_validate_model, but we'll log final metrics here
            mlflow.log_metric("final_val_acc", mean_acc)
            mlflow.log_metric("final_val_mcc", mean_mcc)
            mlflow.log_metric("final_val_f1", mean_f1)
            mlflow.log_param("random_state", seed)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("weight_decay", wd)
            mlflow.log_param("optimizer", "AdamW")

            return mean_mcc
        except Exception as e:
            # log error
            mlflow.log_param("error", str(e))
            return None

def create_cnn_from_trial(series):

    res = {
            "batch_size" : int(series.params_batch_size),
            "lr" : series.params_lr,
            "wd" : series.params_wd
            }

    cnn_layers = series.params_cnn_layers
    dropout_rate = series.params_dropout
    hidden_dim = series.params_hidden_dim
    hidden_dims_flattened = series.params_hidden_dims
    kernel_size = series.params_kernel_size
    mlp_layers = series.params_mlp_layers
    feature_maps = series.params_start_fm

    layers = []
    in_channels = 1
    for _ in range(cnn_layers):
        layers.append(nn.Conv2d(in_channels, feature_maps, kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = feature_maps
        feature_maps *= 2

    layers.append(nn.Flatten())

    layers.append(nn.LazyLinear(hidden_dims_flattened))
    input_dim = hidden_dims_flattened
    
    for _ in range(mlp_layers):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Mish())
        layers.append(nn.Dropout(p=dropout_rate))
        input_dim = hidden_dim

    layers.append(nn.Linear(hidden_dim, 1))

    res["model"] =  nn.Sequential(*layers)
    return res
def retrain_and_eval(series, weight_path, test_path, transform):
    res = create_cnn_from_trial(series)
    test_set = ImageFolder(f"{test_path}/Test", transform=transform)
    test_loader = DataLoader(test_set, batch_size=res["batch_size"])

    model = res["model"]
    model.load_state_dict(torch.load(weight_path, weights_only=True))

    test_acc, test_mcc, _, test_res = ttk.test_model(model, test_loader, return_as_df=True)
    return test_acc, test_mcc, test_res, model

