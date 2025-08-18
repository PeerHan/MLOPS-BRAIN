import sys
import os
import numpy as np  # Ensure NumPy is imported

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import torch_toolkit as ttk 
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def test_transformation():
    sample_img = torch.tensor(np.ones((256, 256)))
    invert = ttk.InvertColors()
    assert sample_img.sum().item() == 256 * 256, "Sum should be 256^2"
    inverted_img = invert(sample_img)
    assert inverted_img.sum().item() == 0, "Sum should be 0 after inverting"
    random_img = torch.randn((32, 32))
    normalizer = ttk.SamplewiseNormalize()
    normalized_img = normalizer(random_img)
    assert round(normalized_img.mean().item()) == 0, "Normalized mean should be 0"
    assert round(normalized_img.std().item()) == 1, "Normalized std should be 1"
    return

def test_model_train_and_test():

    class TinyModel(torch.nn.Module):

        def __init__(self):
            super(TinyModel, self).__init__()
    
            self.linear1 = torch.nn.Linear(50, 50)
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(50, 1)
    
        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x

    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    tensor = torch.rand((100, 50, 50))
    targets = torch.randint(low=0, high=1, size=(100, 50, 1))
    loader = DataLoader(TensorDataset(tensor, targets), batch_size=100)
    loss = ttk.train_one_epoch(model, loader, optimizer, nn.BCEWithLogitsLoss())
    assert loss >= 0, "Loss Should be a positive Number"

    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    ttk.train_one_epoch(model, loader, optimizer, torch.nn.BCEWithLogitsLoss())
    weights_changed = False
    for name, param in model.named_parameters():
        if not torch.equal(initial_weights[name], param):
            weights_changed = True
            break
    
    assert weights_changed, "No weights changed after the training epoch"

    val_loss, val_acc = ttk.validate_one_epoch(model, loader, nn.BCEWithLogitsLoss())

    assert val_loss >= 0, "Loss should be a positivde Number"
    assert 1 >= val_acc >= 0, "Accuracy should be between 0 and 1"
    loader = DataLoader(TensorDataset(tensor, targets), batch_size=1)
    mean_acc, mean_mcc, mean_f1, res_tuple = ttk.test_model(model, loader, return_as_df=False)
    targets, preds, probs, logits = res_tuple
    assert 1 >= mean_acc >= 0, "Accuracy should be between 0 and 1"
    assert 1 >= mean_mcc >= -1, "MCC should be between 0 and 1"
    assert 1 >= mean_f1 >= 0, "F1 Score should be between 0 and 1"
    assert np.isin(targets, [0, 1]).all(), "At least one value which is not 0 or 1"
    assert np.isin(preds, [0, 1]).all(), "At least one value which is not 0 or 1"
    assert np.logical_and(probs >= 0, probs <= 1).all(), "Probabilities should be between 0 and 1"
