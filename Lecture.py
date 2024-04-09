import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import LabelEncoder


def read_csv(base_dir):
    """
    Reads CSV files from a directory and concatenates them into a single DataFrame.

    Parameters:
    - base_dir (str): The base directory containing CSV files.

    Returns:
    - pd.DataFrame: Combined DataFrame containing all CSV data.
    """
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
    """
    Removes columns from DataFrame that have more than 60% NaN values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with NaN columns removed.
    """
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
    """
    Identifies features with less than 20 unique values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - list: List of features with less than 20 unique values.
    """
    unique_elem = df.nunique()
    print(unique_elem)
    features_list = []

    for i in range(0, len(unique_elem) - 1):
        if unique_elem[i] < 20:
            features_list.append(unique_elem.index[i])

    print(features_list)
    return features_list


def encoder(df, features):
    """
    Encodes categorical features using LabelEncoder.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - features (list): List of features to be encoded.

    Returns:
    - pd.DataFrame: Encoded DataFrame.
    """
    df_tmp = df.copy()
    for elem in features:
        if elem != 'Lenght' and elem != 'Channel' and elem != 'DS Channel':
            label_encoder = LabelEncoder()
            df_tmp[elem] = label_encoder.fit_transform(df_tmp[elem].astype(str))

    df_tmp[features] = df_tmp[features].astype(float)
    return df_tmp


def plot_heatmap(df, column1, column2, colormap="Blues"):
    """
    Plots a heatmap for two columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column1 (str): Name of the first column.
    - column2 (str): Name of the second column.
    - colormap (str, optional): Colormap for the heatmap. Defaults to "Blues".
    """
    crosstab = pd.crosstab(df[column1], df[column2])

    sns.set_theme(style="whitegrid", font_scale=1)
    plt.figure(figsize=(15, 10))

    sns.heatmap(crosstab, annot=True, fmt="d", cmap=colormap)

    plt.title(f"Heatmap of {column1} vs {column2}")
    plt.xlabel(column2)
    plt.ylabel(column1)

    plt.show()


def num_same_feature(row_1, row_2):
    """
    Calculates the number of common elements between two rows.

    Parameters:
    - row_1 (pd.Series): First row.
    - row_2 (pd.Series): Second row.

    Returns:
    - int: Number of common elements.
    """
    num = 0
    for col_name, val_1 in row_1.items():
        val_2 = row_2[col_name]
        if val_1 == val_2:
            num += 1
    return num


if __name__ == "__main__":
    base_lecture_dir = "dataset\\MAC_derand_lecture-dataset"
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

    for n in range(1, 11):
        print(
            f"N={n}, H={result['H'][n - 1]}, C={result['C'][n - 1]}, V={result['V'][n - 1]}, ERROR={result['ERROR'][n - 1]}")
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

"""We start by defining several functions to handle different aspects of the data processing, such as reading CSV 
files, removing NaN values, identifying features with less than 20 unique values, encoding categorical features, 
plotting heatmaps, and calculating the number of common elements between rows. In the main part of the code (if 
__name__ == "__main__":), we read the CSV files, preprocess the data, and then perform clustering based on the online 
clustering technique described in the task requirements. For each value of N from 1 to 10, we iterate through the 
DataFrame and assign probes to existing clusters or create new clusters based on the similarity metric described in 
the task. We calculate the homogeneity, completeness, V-measure, and error for each value of N. Finally, we plot the 
performance metrics (homogeneity, completeness, V-measure) and error against the values of N, and save the resulting 
plots as images.
In the result figure of this script, we can see when N is 7, the performance is best.
"""

"""The results of Task 1 indicate the performance of the clustering approach with different threshold values (N) for 
feature similarity. Here's the analysis:

1. Homogeneity (H): The homogeneity score measures the extent to which each cluster contains only data points 
from a single class. For N=1 and N=2, the homogeneity scores are 0.0, indicating poor clustering, as no clusters are 
homogeneous. However, as N increases, the homogeneity improves significantly, reaching a perfect score of 1.0 for N=9 
and N=10. This indicates that clusters become more homogeneous as more features are required to match for assignment.

2. Completeness (C): The completeness score measures the extent to which all data points of a given class are 
assigned to the same cluster. Similar to homogeneity, completeness starts low for N=1 and N=2 and increases as N 
increases. However, it decreases slightly for N=9 and N=10, indicating that all data points of a class are not fully 
captured within a single cluster.

3. V-Measure (V): The V-measure is the harmonic mean of homogeneity and completeness, providing a balanced 
assessment of clustering quality. As with homogeneity and completeness, the V-measure improves as N increases, 
peaking at N=9 and N=10. However, it is essential to note that a decrease in completeness at higher N values affects 
the V-measure.

4. Error: The error metric quantifies the discrepancy between the number of clusters and the number of true 
classes. Negative error values indicate an overestimation of the number of clusters, while positive values indicate 
an underestimation. For N=1 and N=2, the error is significantly negative, suggesting an excessive number of clusters. 
As N increases, the error decreases and eventually becomes positive, indicating a better alignment between the number 
of clusters and true classes. However, at N=10, the error spikes drastically, indicating severe underestimation of 
the number of clusters, likely due to overly strict feature similarity criteria.

In summary, the clustering performance improves as the threshold value N increases up to a certain point, 
where clusters become more homogeneous and complete. However, setting N too high can lead to overfitting and an 
insufficient number of clusters, impacting the clustering quality negatively. Therefore, the optimal value of N 
should strike a balance between capturing meaningful patterns in the data and avoiding overfitting."""
