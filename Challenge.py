import os
import random

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.metrics import homogeneity_completeness_v_measure

import Lecture


def select_files(K, file_list):
    """Randomly selects K unique files from the given list of files."""
    dataset_list = []
    num_dataset = 0
    while num_dataset < 5:
        selected_files = random.sample(file_list, K)
        if selected_files not in dataset_list:
            dataset_list.append(selected_files)
            num_dataset += 1
        if K == 6:
            break
    return dataset_list


def read_csv(base_dir, file_list):
    """Reads CSV files specified in the file list from the given directory and concatenates them into a single
    DataFrame."""
    df_list = list()
    print(f"Processing file: {file_list}")
    for file in file_list:
        file_path = os.path.join(base_dir, file)
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding='latin1')
            df_list.append(df)
    combine = pd.concat(df_list, ignore_index=True)
    combine['Timestamp'] = pd.to_datetime(combine['Timestamp'], unit='s')
    return combine


if __name__ == '__main__':
    base_challenge_dir = "dataset\\MAC_derand_challenge-dataset\\challenge-dataset\\"
    files = os.listdir(base_challenge_dir)
    N = 7  # Set the threshold value N determined in previous code
    result = {}  # Dictionary to store the results
    for k in range(2, 7):  # Iterate over different values of k
        dataset_list = select_files(K=k, file_list=files)  # Randomly select files for each dataset
        result[k] = {"H": [], "C": [], "V": [], "ERROR": []}  # Initialize the result dictionary for current k
        for dataset in dataset_list:  # Iterate over each dataset
            # Read and preprocess the dataset
            combine_df = read_csv(base_dir=base_challenge_dir, file_list=dataset)
            display(combine_df)
            clean_df = Lecture.remove_NaN(df=combine_df)
            display(clean_df)
            features = Lecture.find_features_less_20(df=clean_df)
            encoded_df = Lecture.encoder(df=clean_df, features=features)
            burst_df = encoded_df.drop(['Timestamp'], axis=1)
            burst_df = burst_df.groupby(['MAC Address'])
            burst_df = burst_df.first().reset_index()
            display(burst_df)
            label_count = burst_df["Label"].value_counts()
            print(f"Label count:\n {label_count}")

            columns_to_keep = features
            features.insert(0, "MAC Address")
            features.append("Label")

            cluster_df = burst_df[features].copy().reset_index()
            print(f"features:{features}")
            Cluster_ID = 1
            # Apply clustering algorithm to assign Cluster IDs
            for index, row in cluster_df.iterrows():
                flag = False
                temp = 0
                for i, row_i in cluster_df.iterrows():
                    same_features = 0
                    if i < index:
                        same_features = Lecture.num_same_feature(row_1=row, row_2=row_i)
                        if same_features >= N:
                            flag = True
                            if same_features > temp:
                                cluster_df.loc[index, "Cluster ID"] = cluster_df.loc[i, "Cluster ID"]
                                temp = same_features
                if flag is False:
                    cluster_df.loc[index, "Cluster ID"] = Cluster_ID
                    Cluster_ID += 1
            display(cluster_df)
            # Visualize clustering results
            Lecture.plot_heatmap(cluster_df, 'Label', "Cluster ID")
            # Compute and print error
            n_unique_clusterid = len(np.unique(cluster_df["Cluster ID"]))
            n_unique_label = len(np.unique(cluster_df["Label"]))
            Error = n_unique_clusterid - n_unique_label
            print("Error", Error)
            # Compute homogeneity, completeness, and V-measure
            h, c, v = homogeneity_completeness_v_measure(cluster_df["Label"], cluster_df["Cluster ID"])

            # Store the results
            result[k]["H"].append(h)
            result[k]["C"].append(c)
            result[k]["V"].append(v)
            result[k]["ERROR"].append(abs(Error))

    # Print the results
    print(result)
    avg_H = []
    avg_C = []
    avg_V = []
    avg_Error = []
    for k in range(3, 6):
        avg_H.append(sum(result[k]['H']) / len(result[k]['H']))
        avg_C.append(sum(result[k]['C']) / len(result[k]['C']))
        avg_V.append(sum(result[k]['V']) / len(result[k]['V']))
        avg_Error.append(sum(result[k]['ERROR']) / len(result[k]['ERROR']))
        print(
            f"K={k}, Average H={sum(result[k]['H']) / len(result[k]['H'])}, Average C={sum(result[k]['C']) / len(result[k]['C'])}, Average V={sum(result[k]['V']) / len(result[k]['V'])}, Average Error={sum(result[k]['ERROR']) / len(result[k]['ERROR'])}")

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(range(2, 7), avg_H, label='Average Homogeneity', marker='o')
    axes[0].plot(range(2, 7), avg_C, label='Average Completeness', marker='o')
    axes[0].plot(range(2, 7), avg_V, label='Average V-Measure', marker='o')
    axes[0].set_title('Performance Metrics vs K')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Score')
    axes[0].legend()

    axes[1].plot(range(2, 7), avg_Error, label='Average Error', marker='o', color='red')
    axes[1].set_title('Error vs K')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Error')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("result_figure\\Challenge_dataset_Performance.png")
    plt.show()
