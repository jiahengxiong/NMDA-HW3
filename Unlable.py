import numpy as np
from IPython.display import display
import Lecture

if __name__ == '__main__':
    # Set the threshold value N
    N = 7

    # Directory containing the unlabeled dataset
    base_unlabeled_dir = "dataset"

    # Read and display the unlabeled dataset
    combine_df = Lecture.read_csv(base_dir=base_unlabeled_dir)
    display(combine_df)

    # Remove NaN values from the dataset
    clean_df = Lecture.remove_NaN(df=combine_df)
    display(clean_df)

    # Identify features with less than 20 unique values
    features = Lecture.find_features_less_20(df=clean_df)

    # Encode categorical features using LabelEncoder
    encoded_df = Lecture.encoder(df=clean_df, features=features)

    # Drop the 'Timestamp' column from the encoded dataset
    burst_df = encoded_df.drop(['Timestamp'], axis=1)

    # Group the dataset by 'MAC Address' and select the first row of each group
    burst_df = burst_df.groupby(['MAC Address']).first().reset_index()
    display(burst_df)

    # Set the columns to keep for clustering
    columns_to_keep = features
    features.insert(0, "MAC Address")

    # Create a copy of the burst_df DataFrame for clustering
    cluster_df = burst_df[features].copy().reset_index()
    print(f"features:{features}")

    # Initialize Cluster_ID
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

    # Display the clustered DataFrame
    display(cluster_df)

    # Calculate the number of unique cluster IDs
    n_unique_clusterid = len(np.unique(cluster_df["Cluster ID"]))
    print(f"Number of unique cluster_id: {n_unique_clusterid}")


"""This script runs the best clustering technique, determined by the threshold value N=7, on the provided unlabeled 
dataset unlabelled-challenge.csv to estimate the number of devices. It reads the dataset, preprocesses it by removing 
NaN values and encoding categorical features, and then performs clustering to assign Cluster IDs. Finally, 
it calculates the number of unique Cluster IDs, which represents the estimated number of devices in the dataset. The 
number of deices is 93"""