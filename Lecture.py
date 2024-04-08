import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure


def read_csv(base_dir):
    """Reads CSV files from a directory and concatenates them into a single DataFrame."""
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
    """Removes columns from DataFrame that have more than 60% NaN values."""
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
    """Identifies features with less than 20 unique values."""
    unique_elem = df.nunique()

    print(unique_elem)

    features = []

    for i in range(0, len(unique_elem) - 1):
        if unique_elem[i] < 20:
            features.append(unique_elem.index[i])

    print(features)
    return features


def encoder(df, features):
    """Encodes categorical features using LabelEncoder."""
    df_tmp = df.copy()
    for elem in features:
        if elem != 'Lenght' and elem != 'Channel' and elem != 'DS Channel':
            label_encoder = LabelEncoder()
            df_tmp[elem] = label_encoder.fit_transform(df_tmp[elem].astype(str))

    df_tmp[features] = df_tmp[features].astype(float)
    return df_tmp


def plot_heatmap(df, column1, column2, colormap="Blues"):
    """Plots a heatmap for two columns of a DataFrame."""
    crosstab = pd.crosstab(df[column1], df[column2])

    sns.set_theme(style="whitegrid", font_scale=1)
    plt.figure(figsize=(15, 10))

    sns.heatmap(crosstab, annot=True, fmt="d", cmap=colormap)

    plt.title(f"Heatmap of {column1} vs {column2}")
    plt.xlabel(column2)
    plt.ylabel(column1)

    plt.show()


def num_same_feature(row_1, row_2):
    """Calculates the number of common elements between two rows."""
    num = 0
    for col_name, val_1 in row_1.items():
        val_2 = row_2[col_name]
        if val_1 == val_2:
            num += 1
    return num


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

    columns_to_keep = features
    features.insert(0, "MAC Address")
    features.append("Label")

    cluster_df = burst_df[features].copy().reset_index()
    print(f"features:{features}")
    result = {"H": [], "C": [], "V": [], "ERROR": []}
    for N in range(1, 11):
        Cluster_ID = 1
        for index, row in cluster_df.iterrows():
            flag = False
            temp = 0
            for i, row_i in cluster_df.iterrows():
                same_features = 0
                if i < index:
                    same_features = num_same_feature(row_1=row, row_2=row_i)
                    if same_features >= N:
                        flag = True
                        if same_features > temp:
                            cluster_df.loc[index, "Cluster ID"] = cluster_df.loc[i, "Cluster ID"]
                            temp = same_features
            if flag is False:
                cluster_df.loc[index, "Cluster ID"] = Cluster_ID
                Cluster_ID += 1

        display(cluster_df)

        plot_heatmap(cluster_df, 'Label', "Cluster ID")

        n_unique_clusterid = len(np.unique(cluster_df["Cluster ID"]))
        n_unique_label = len(np.unique(cluster_df["Label"]))
        print(f"*****************{N}******************")
        print("Error", n_unique_clusterid - n_unique_label)

        # cluster_df.fillna(0, inplace=True)

        h, c, v = homogeneity_completeness_v_measure(cluster_df["Label"], cluster_df["Cluster ID"])

        print("Homog: ", h)
        print("Completeness: ", c)
        print("V-Meas: ", v)
        result['H'].append(h)
        result['C'].append(c)
        result['V'].append(v)
        result['ERROR'].append(n_unique_clusterid - n_unique_label)

    print(result)
    H = result['H']
    C = result['C']
    V = result['V']
    ERROR = result['ERROR']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(range(1, 11), H, label='Homogeneity', marker='o')
    axes[0].plot(range(1, 11), C, label='Completeness', marker='o')
    axes[0].plot(range(1, 11), V, label='V-Measure', marker='o')
    axes[0].set_title('Performance Metrics vs N')
    axes[0].set_xlabel('N')
    axes[0].set_ylabel('Score')
    axes[0].legend()

    axes[1].plot(range(1, 11), ERROR, label='Error', marker='o', color='red')
    axes[1].set_title('Error vs N')
    axes[1].set_xlabel('N')
    axes[1].set_ylabel('Error')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("result_figure\\Lecture_dataset_Performance.png")
    plt.show()
