"""
TorchToolkit
PyTorch Functions
"""

import torch
from torch.nn import functional as F
from torchmetrics.classification import accuracy, matthews_corrcoef, BinaryF1Score
import mlflow
import numpy as np
import pandas as pd

class SamplewiseNormalize:
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std

class InvertColors:
    def __call__(self, tensor):
        return 1 - tensor 

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    
    
    running_loss = 0.0

    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()

    running_loss /= len(train_loader)

    return running_loss

def validate_one_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    acc = accuracy.Accuracy("binary")
    mean_acc = []
    for data in val_loader:
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), labels.float())
        running_loss += loss.item()
        probs = F.sigmoid(outputs.squeeze(1))
        preds = (probs >= 0.5).int()
        val_acc = acc(preds, labels)
        mean_acc.append(val_acc.item())
    running_loss /= len(val_loader)
    mean_acc_res = np.mean(mean_acc)
    return running_loss, mean_acc_res

def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, epochs, show_every=None):
    early_stopper = EarlyStopper()
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val = float("inf")
    best_state_dict = None
    for epoch in range(epochs):
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = validate_one_epoch(model, val_loader, criterion)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_accuracies.append(v_acc)

        if show_every and epoch % show_every == 0:
            print(f"Train Loss: {t_loss:.3} | Val Loss: {v_loss:.3} | Val Accuracy: {v_acc:.3}")
        if v_loss < best_val:
            best_val = v_loss
            best_state_dict = model.state_dict()
        if early_stopper.early_stop(v_loss):
            print("Early Stopped")
            break
        
        # Log metrics if mlflow is active
        try:
            if mlflow.active_run():
                mlflow.log_metric("train_loss", t_loss, step=epoch)
                mlflow.log_metric("val_loss", v_loss, step=epoch)
                mlflow.log_metric("val_accuracy", v_acc, step=epoch)
        except ImportError:
            pass

    return train_losses, val_losses, val_accuracies, best_state_dict

def test_model(model, test_loader, return_as_df=True):
    all_preds = []
    all_targets = []
    all_probs = []
    all_logits = []
    mean_acc = []
    mean_mcc = []
    mean_f1 = []
    with torch.no_grad():
        model.eval()
        acc = accuracy.Accuracy("binary")
        mcc = matthews_corrcoef.BinaryMatthewsCorrCoef()
        f1_score = BinaryF1Score()
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            probs = F.sigmoid(outputs.squeeze(1))
            preds = (probs >= 0.5).int()
            val_acc = acc(labels, preds)
            val_mcc = mcc(labels, preds)
            val_f1 = f1_score(labels, preds)
            mean_acc.append(val_acc.item())
            mean_mcc.append(val_mcc.item())
            mean_f1.append(val_f1.item())
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(labels)
            all_logits.append(outputs.squeeze(1))
    mean_acc_res = np.mean(mean_acc)
    mean_mcc_res = np.mean(mean_mcc)
    mean_f1_res = np.mean(mean_f1)
    logits = torch.cat(all_logits).numpy()
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = torch.cat(all_probs).numpy().round(4)
    if return_as_df:
        res_df = pd.DataFrame({"Targets" : targets,
                  "Preds" : preds,
                  "Probs" : probs,
                  "Logits" : logits})
        return mean_acc_res, mean_mcc_res, mean_f1_res, res_df
    return mean_acc_res, mean_mcc_res, mean_f1_res, (targets, preds, probs, logits)
