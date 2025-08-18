"""
Experiment Visualizer
Prewritten Functions to visualize
Experiment Data
"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

def match_label(value):
    mcc_map = {
        0 : "<= 0.0",
        0.1 : "<= 0.1",
        0.25 : "<= 0.25",
        0.5 : "<= 0.5",
        0.75 : "<= 0.75",
        0.9 : "<= 0.9",
        1 : "<= 1"
    }
    label = None
    for threshold, label in mcc_map.items():
        if value <= threshold:
            break
    return label

def line_plot(df, feat, artifact_folder):
    plt.figure(figsize=(10, 5), dpi=150)
    sns.lineplot(df,
                 x=feat,
                 y="MCC")
    plt.title("Improvemend over Trials")
    artifact_name = "MCC_Over_Trials.png"
    plt.savefig(f"{artifact_folder}/{artifact_name}", bbox_inches="tight", dpi=150)

def scatter_plot(df, feat1, feat2, artifact_folder):
    plt.figure(figsize=(10, 5), dpi=150)
    sns.scatterplot(df, x=feat1, y=feat2, hue="Rank")
    plt.title(f"Dependency of HP {feat1} and {feat2}")
    artifact_name = f"scatter_plot_{feat1}_{feat2}.png"
    plt.savefig(f"{artifact_folder}/{artifact_name}", bbox_inches="tight", dpi=150)

def duration_plot(df, artifact_folder):
    plt.figure(figsize=(10, 5), dpi=150)
    df["Duration [Min]"] = df.duration.astype(str).str.split(" ").str.slice(2,3).transform(lambda lst : lst[0].split(":")[1]).astype(int)
    sns.kdeplot(df,
               x="Duration [Min]",
               hue="Rank")
    plt.title("Trial Duration Density")
    artifact_name = "Duration_of_Trial_density.png"
    plt.savefig(f"{artifact_folder}/{artifact_name}", bbox_inches="tight", dpi=150)

def category_plot(df, feat, artifact_folder):
    plt.figure(figsize=(10, 5), dpi=150)
    sns.boxplot(df, x=feat, y="MCC", hue="Rank")
    plt.title(f"Distribution of {feat} per Trial")
    artifact_name = f"distribution_of_{feat}.png"
    plt.savefig(f"{artifact_folder}/{artifact_name}", bbox_inches="tight", dpi=150)

def count_plot(df, feat, artifact_folder):
    plt.figure(figsize=(10, 5), dpi=150)
    sns.countplot(df, x=feat,  hue="Rank")
    plt.title(f"Counts of {feat}")
    artifact_name = f"counts_of_{feat}.png"
    plt.savefig(f"{artifact_folder}/{artifact_name}", bbox_inches="tight", dpi=150)
