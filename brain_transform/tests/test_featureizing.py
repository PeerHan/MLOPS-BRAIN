import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from signal_toolkit import *
import numpy as np
import pandas as pd
from matplotlib.image import imread
import shutil
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

def test_butter_bandpass_filter():
    multi_timeseries = np.random.randn(100, 10)
    filtered = butter_bandpass_filter(multi_timeseries, 0.5, 1, 5)
    assert np.isclose(multi_timeseries, filtered).sum() == 0, "Every Value should change after the filter"
    assert filtered.shape == multi_timeseries.shape, "The Shape should not change through transformation"
    df = pd.DataFrame(multi_timeseries, columns=[f"col_{i}" for i in range(multi_timeseries.shape[1])])
    transformed_df = apply_butter_bandpass_filter(df, col_names=df.columns)
    assert type(df) == pd.DataFrame and type(transformed_df) == pd.DataFrame, "in and output should be Dataframes"
    assert df.eq(transformed_df).sum().sum() == 0, "Input df should not change"
    x_cols = df.columns
    df["Dummy"] = 1
    transformed_df = apply_butter_bandpass_filter(df, col_names=x_cols)
    assert (df.Dummy == transformed_df.Dummy).sum() == df.shape[0], "Dummy col should remain the same"    

def test_moving_average():
    df = pd.DataFrame({"Col1" : list(range(10)), "Col2" : list(range(10, 20))})
    transformed_df = apply_moving_average(df, df.columns, window=1)
    assert type(df) == pd.DataFrame and type(transformed_df) == pd.DataFrame, "in and output should be Dataframes"
    assert df.equals(transformed_df), "Dataframe should not change with window=1"
    transformed_df = apply_moving_average(df, df.columns, window=2)
    assert not df.equals(transformed_df), "Input df should not change"
    assert df.shape == transformed_df.shape, "Shape should not change"
    transformed_df = apply_moving_average(df, df.columns, window=10)
    assert (transformed_df.iloc[-1, :] != df.mean()).sum() == 0, "Last Row should be the same as the average with full window"
    x_cols = df.columns
    df["Dummy"] = 1
    transformed_df = apply_moving_average(df, ["Col1", "Col2"], window=5)
    assert (df.Dummy == transformed_df.Dummy).sum() == df.shape[0], "Dummy col should remain the same"  

def test_stationary():
    df = pd.DataFrame({"Col1" : list(range(10)), "Col2" : list(range(10, 20))})
    transformed_df = apply_stationary(df, df.columns, periods=1)
    assert type(df) == pd.DataFrame and type(transformed_df) == pd.DataFrame, "in and output should be Dataframes"
    assert transformed_df.isna().sum().sum() == transformed_df.shape[1], "First Row should be NaN"
    assert transformed_df.sum().sum() == df.shape[0] * 2 - df.shape[1], "Every Entry should be 1 Except first row"
    assert not df.equals(transformed_df), "Input df should not change"
    x_cols = df.columns
    df["Dummy"] = 1
    transformed_df = apply_stationary(df, ["Col1", "Col2"], periods=5)
    assert (df.Dummy == transformed_df.Dummy).sum() == df.shape[0], "Dummy col should remain the same"   

def test_csv_processing():

    time_division = 5
    multi_timeseries = np.random.randn(100, 8)
    df = pd.DataFrame(multi_timeseries, columns=[f"Channel_{i}" for i in range(1, multi_timeseries.shape[1]+1)])
    df["Timestemp"] = list(range(len(df)))
    df["Target"] = (df.Timestemp == 30).astype(int)
    parts = [df.iloc[i:i+time_division].copy() for i in range(len(df) - time_division + 1)]

    os.makedirs("Test_Tag", exist_ok=True)
    os.makedirs("Test_Tag/Test_Dir", exist_ok=True)
    os.makedirs("Test_Tag/Test_Dir/true", exist_ok=True)
    os.makedirs("Test_Tag/Test_Dir/false", exist_ok=True)
    
    for i, part in enumerate(parts):
        assert type(part) == pd.DataFrame, "input should be a Dataframe"
        label = int(part.Target.sum() > 0)
        sub_dir = "true" if label else "false"
        prev_files = len(os.listdir(f"Test_Tag/Test_Dir/{sub_dir}/"))
        generate_img(part, label, f"TestPrefix_{i}", "Test_Dir/", "Test_Tag")
        files = len(os.listdir(f"Test_Tag/Test_Dir/{sub_dir}/"))
        assert files == prev_files + 1, "The Folder should have one more file then before"

    sample_file = os.listdir("Test_Tag/Test_Dir/true/")[0]
    img = imread(f"Test_Tag/Test_Dir/true/{sample_file}")
    assert type(img) == np.ndarray, "Image must be a np array"
    assert len(img.shape) == 3, "Image must be a 3 Dim Array"
    assert len(os.listdir("Test_Tag/Test_Dir/false/")) + len(os.listdir("Test_Tag/Test_Dir/true")) == len(parts)
    shape = img.shape
    assert all([imread(f"Test_Tag/Test_Dir/false/{img}").shape == shape for img in os.listdir("Test_Tag/Test_Dir/false/")]), "Every Img must have the same shape"
    assert all([imread(f"Test_Tag/Test_Dir/true/{img}").shape == shape for img in os.listdir("Test_Tag/Test_Dir/true/")]), "Every Img must have the same shape"
    assert all([img.startswith("TestPrefix") for img in os.listdir("Test_Tag/Test_Dir/false/")]), "Every Img must start with same Prefix"
    assert all([img.startswith("TestPrefix") for img in os.listdir("Test_Tag/Test_Dir/true/")]), "Every Img must start with same Prefix"
    shutil.rmtree('Test_Tag')
