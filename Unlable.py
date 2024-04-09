import numpy as np
from IPython.display import display

import Lecture

if __name__ == '__main__':
    N = 7
    base_unlabeled_dir = "dataset"
    combine_df = Lecture.read_csv(base_dir=base_unlabeled_dir)
    display(combine_df)
    clean_df = Lecture.remove_NaN(df=combine_df)
    display(clean_df)
    features = Lecture.find_features_less_20(df=clean_df)
    encoded_df = Lecture.encoder(df=clean_df, features=features)
    burst_df = encoded_df.drop(['Timestamp'], axis=1)
    burst_df = burst_df.groupby(['MAC Address'])
    burst_df = burst_df.first().reset_index()
    display(burst_df)
    columns_to_keep = features
    features.insert(0, "MAC Address")

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

    n_unique_clusterid = len(np.unique(cluster_df["Cluster ID"]))
    print(f"Number of unique cluster_id: {n_unique_clusterid}")
