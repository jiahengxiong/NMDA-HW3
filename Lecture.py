import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure


def read_csv(base_dir):
    df_list = list()

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, encoding='latin1')
                df_list.append(df)
    combine = pd.concat(df_list, ignore_index=True)
    combine['Timestamp'] = pd.to_datetime(combine['Timestamp'], unit='s')
    return combine


def remove_NaN(df):
    nan_sum = df.isna().sum()

    nan_percentage = (nan_sum / len(df)) * 100

    print(nan_percentage)

    col_drop = []

    for i in range(0, len(nan_percentage)):
        if nan_percentage[i] > 60:
            col_drop.append(nan_percentage.index[i])

    print(col_drop)

    removed_df = df.drop(col_drop, axis=1)
    return removed_df


def find_features_less_20(df):
    unique_elem = df.nunique()

    print(unique_elem)

    features = []

    for i in range(0, len(unique_elem) - 1):
        if unique_elem[i] < 20:
            features.append(unique_elem.index[i])

    print(features)
    return features


def encoder(df, features):
    df_tmp = df.copy()
    for elem in features:
        if elem != 'Lenght' and elem != 'Channel' and elem != 'DS Channel':
            label_encoder = LabelEncoder()
            df_tmp[elem] = label_encoder.fit_transform(df_tmp[elem].astype(str))

    df_tmp[features] = df_tmp[features].astype(float)
    return df_tmp


def plot_heatmap(df, column1, column2, colormap="Blues"):
    crosstab = pd.crosstab(df[column1], df[column2])

    sns.set_theme(style="whitegrid", font_scale=1)
    plt.figure(figsize=(15, 10))

    sns.heatmap(crosstab, annot=True, fmt="d", cmap=colormap)

    plt.title(f"Heatmap of {column1} vs {column2}")
    plt.xlabel(column2)
    plt.ylabel(column1)

    plt.show()


if __name__ == "__main__":
    base_lecture_dir = "dataset\\MAC_derand_lecture-dataset"
    base_challenge_dir = "dataset\\MAC_derand_challenge"
    combine_df = read_csv(base_dir=base_lecture_dir)
    display(combine_df)
    clean_df = remove_NaN(df=combine_df)
    display(clean_df)
    features = find_features_less_20(df=clean_df)
    encoded_df = encoder(df=clean_df, features=features)
    burst_df = encoded_df.drop(['Timestamp'], axis=1)
    burst_df = burst_df.groupby(['MAC Address'])
    burst_df = burst_df.first().reset_index()
    display(burst_df)
    label_count = burst_df["Label"].value_counts()
    print(label_count)

    plot_heatmap(encoded_df, 'Label', "Supported Rates")
    plot_heatmap(encoded_df, 'Label', "Extended Supported Rates")
    plot_heatmap(encoded_df, 'Label', "Vendor Specific Tags")
    plot_heatmap(encoded_df, 'Label', "HT Capabilities")
    plot_heatmap(encoded_df, 'Label', "Extended Capabilities")

    columns_to_keep = features
    features.insert(0, "MAC Address")
    features.append("Label")

    cluster_df = burst_df[features].copy().reset_index()
    print(f"features:{features}")

    selected_features = ["HT Capabilities", "Extended Capabilities"]
    cluster_df["Cluster ID"] = cluster_df.groupby(selected_features).ngroup()

    display(cluster_df)

    plot_heatmap(cluster_df, 'Label', "Cluster ID")

    n_unique_clusterid = len(np.unique(cluster_df["Cluster ID"]))
    n_unique_label = len(np.unique(cluster_df["Label"]))
    print("Error", n_unique_clusterid - n_unique_label)

    h, c, v = homogeneity_completeness_v_measure(cluster_df["Label"], cluster_df["Cluster ID"])

    print("Homog: ", h)
    print("Completeness: ", c)
    print("V-Meas: ", v)
