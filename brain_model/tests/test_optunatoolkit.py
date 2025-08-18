import sys
import os
import pandas as pd
import numpy as np
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from optuna_toolkit import optuna_cnn, create_cnn_from_trial
import optuna

def test_optuna_model_creation():
    study = optuna.create_study(direction="maximize", study_name="TestStudy", sampler=optuna.samplers.TPESampler())
    trial = study.ask()
    optuna_model = optuna_cnn(trial)
    for i, module in enumerate(optuna_model):
        if isinstance(module, nn.Conv2d):
            assert module.out_channels >= 1, "Channels must be a positiv number"
            assert module.in_channels >= 1, "Channels must be a positiv number"
            assert module.kernel_size in [(i, i) for i in range(3, 12, 2)], "Kernel must be a positive odd number"
        elif isinstance(module, nn.MaxPool2d):
            assert module.kernel_size == 2, "Pooling Layer "
            assert module.stride == 2, "Pooling Layer"
        elif isinstance(module, nn.LazyLinear):
            assert module.in_features == 0, "LazyLinear should have 0 in features"
            assert 500 >= module.out_features >= 100, "Out should be between 500 and 500"
        elif isinstance(module, nn.Linear):
            assert module.in_features >= 100, "Hidden dim should be >= 100"
            assert module.out_features >= 1, "Output dim should be >= 1"
        elif isinstance(module, nn.Dropout):
            assert 0.5 >= module.p >= 0.1, "Dropout should be between 0.1 - 0.5"
        else:
            isinstance(module, nn.ReLU) or isinstance(module, nn.Mish) or isinstance(module, nn.Flatten), "Other Module should be Activation or Flatten"
    return

def test_optuna_recreation():

    bs = 1
    lr = 0.01
    wd = 0
    cnn_layers = 1
    dropout = 0
    hidden_dim = 100
    hidden_dims = 500
    kernel_size = 3
    mlp_layers = 1
    fm = 16
    
    test_series = pd.Series({"params_batch_size" : bs,
                             "params_lr" : lr,
                             "params_wd" : wd,
                             "params_cnn_layers" : cnn_layers,
                             "params_dropout" : dropout,
                             "params_hidden_dim" : hidden_dim,
                             "params_hidden_dims" : hidden_dims,
                             "params_kernel_size" : kernel_size,
                             "params_mlp_layers" : mlp_layers,
                             "params_start_fm" : fm},
                            dtype="object")
    
    res_dict = create_cnn_from_trial(test_series)
    
    assert res_dict["batch_size"] == bs, "Should be batch size from dummy data"
    assert res_dict["lr"] == lr, "Should be learning rate from dummy data"
    assert res_dict["wd"] == wd, "Should be weight decay from dummy data"
    assert isinstance(res_dict["model"], nn.Sequential), "Model Should be instance of Sequential"

    return
