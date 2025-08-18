# copied from peer schliephake's code
"""
SignalToolkit
"""
import os
import logging
from scipy.signal import butter, sosfilt
from matplotlib import pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Butter Bandpass Function
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Butter Bandpass Filter
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def apply_butter_bandpass_filter(df,
                                 col_names,
                                 low_cut=0.5,
                                 high_cut=30,
                                 fs=250,
                                 order=5):
    """
    Apply Butter Bandpass Filter 
    """
    transformed_df = df.copy()
    for col in col_names:
        data = df.loc[:, col]
        transformed_df.loc[:, col] = butter_bandpass_filter(data, low_cut, high_cut, fs, order)
    return transformed_df

def apply_moving_average(df, col_names, window=10, min_periods=1):
    """
    Apply Moving Average
    """
    ma_df = df.copy()
    for col in col_names:
        data = df.loc[:, col]
        ma_df.loc[:, col] = data.rolling(window=window, min_periods=min_periods).mean()
    return ma_df

def apply_stationary(df, col_names, periods=1):
    """
    Apply Stationary Function
    """
    diff_df = df.copy()
    for col in col_names:
        data = df.loc[:, col]
        diff_df.loc[:, col] = data.diff(periods=periods)
    return diff_df

def generate_img(data,
                 label,
                 filename_prefix,
                 subdir,
                 tag_viz_dir): # pylint disable=too-many-locals
    """
    Generate Img from Data
    """
    y_lim_min = -10
    y_lim_max = 10
    figsize = (10, 10)
    fig, axs = plt.subplots(1, 1, figsize=figsize)

    # Use the correct directory structure
    viz_output_dir = os.path.join(tag_viz_dir,
                                  subdir,
                                  'true' if label == 1 else 'false')
    output_file = os.path.join(viz_output_dir,
                               f"{filename_prefix}.png")
    # EEG Channels
    col_names = ["Channel_1", "Channel_2",
                 "Channel_3", "Channel_4",
                 "Channel_5", "Channel_6",
                 "Channel_7", "Channel_8"]
    for i in range(8):
        col_name = col_names[i]
        current_col = data[col_name]
        axs.plot(current_col.index, current_col.values, color="black")
    axs.set_ylim(y_lim_min, y_lim_max)
    axs.grid(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Generated visualization: {output_file}")
