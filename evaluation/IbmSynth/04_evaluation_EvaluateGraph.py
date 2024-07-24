import pickle

from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, cdist

from sklearn.cluster import KMeans


models = ['DOPPELGANGER', 'FINDIFF', 'TVAE', 'WGAN', 'CTGAN']
keep_col = ['Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering', 'transaction_clusters']

real_data = pd.read_csv("./working/transformed_df_graph.csv")
real_data = real_data.sample(1000000)
## get the unique nodes for real data
real_target_nodes = real_data[["target_id_{}".format(i) for i in range(6)]]
real_target_nodes.columns = ["id_{}".format(i) for i in range(6)]
real_source_nodes = real_data[["source_id_{}".format(i) for i in range(6)]]
real_source_nodes.columns = ["id_{}".format(i) for i in range(6)]
real_nodes = pd.concat((real_target_nodes, real_source_nodes), axis= 0).drop_duplicates()
num_real_nodes = real_nodes.shape[0]
## calculate average distance between the nodes for real data
real_nodes_avg_distance = []
for i, chunk in tqdm(real_nodes.groupby(np.arange(len(real_nodes))//10000), desc= "Calculating Average Distance between Real Nodes"):
    real_nodes_avg_distance.append(np.mean(cdist(real_nodes.drop(chunk.index).values, chunk.values, 'euclid')))
real_nodes_avg_distance = np.mean(real_nodes_avg_distance)

for model in models:
    print("Currently Processing Model: {}".format(model))

    with open("./model/{}.pkl".format(model), 'rb') as file:
        synthesizer = pickle.load(file)

    # sample in batches
    synthetic_data = pd.DataFrame()
    for i in tqdm(range(0, real_data.shape[0], 500000), desc= "Generate Synthetic Data"):
        synthetic_data = pd.concat((synthetic_data, synthesizer.sample(num_rows=500000)))

    # Restore Graph Structure
    ## Use KNN to find transactions refering to the same node
    ## optimize to keep the average distance between the nodes similar between real and synthetic data
    ## get the unique nodes for synthetic data
    synthetic_target_nodes = synthetic_data[["target_id_{}".format(i) for i in range(6)]]
    synthetic_target_nodes.columns = ["id_{}".format(i) for i in range(6)]
    synthetic_source_nodes = synthetic_data[["source_id_{}".format(i) for i in range(6)]]
    synthetic_source_nodes.columns = ["id_{}".format(i) for i in range(6)]
    synthetic_nodes = pd.concat((synthetic_target_nodes, synthetic_source_nodes), axis=0).drop_duplicates()

    synth_nodes_avg_distances = {}
    for n_clusters in tqdm(range(int(num_real_nodes*0.5), int(num_real_nodes*2), 10000), desc="Search for optimal Number of Nodes"):
        kms = KMeans(n_clusters= n_clusters, n_init= "auto")
        kms.fit(synthetic_nodes)
        new_synthetic_nodes = kms.cluster_centers_

        synthetic_nodes_avg_distance = []
        for i, chunk in tqdm(new_synthetic_nodes.groupby(np.arange(len(new_synthetic_nodes)) // 10000)):
            synthetic_nodes_avg_distance.append(np.mean(cdist(new_synthetic_nodes.drop(chunk.index).values, chunk.values, 'euclid')))
        synthetic_nodes_avg_distance = np.mean(synthetic_nodes_avg_distance)

        synth_nodes_avg_distances[n_clusters] = synthetic_nodes_avg_distance

    n_cluster = min(synth_nodes_avg_distances, key=lambda x:abs(x- real_nodes_avg_distance))
    kms = KMeans(n_clusters=n_cluster, random_state=42, n_init="auto")
    synthetic_nodes["node_id"] = kms.fit_predict(synthetic_nodes)

    synthetic_data = synthetic_data.merge(synthetic_nodes, left_on= ["target_id_{}".format(i) for i in range(6)], right_on= ["id_{}".format(i) for i in range(6)], how="left")
    synthetic_data = synthetic_data.merge(synthetic_nodes, left_on=["source_id_{}".format(i) for i in range(6)], right_on=["id_{}".format(i) for i in range(6)], how="left")
    synthetic_data.drop(columns = ["target_id_{}".format(i) for i in range(6)])
    synthetic_data.drop(columns = ["source_id_{}".format(i) for i in range(6)])

    synthetic_data.to_csv("./synth/{}_synthetic_data_graph.csv".format(model), index=False)

