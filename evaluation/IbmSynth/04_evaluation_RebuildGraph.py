import os.path
import pickle

from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, cdist

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


models = ['DOPPELGANGER', 'FINDIFF', 'TVAE', 'WGANGPwDRS', 'CTGAN']
keep_col = ['Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering', 'transaction_clusters']

real_data = pd.read_csv("./working/transformed_df_graph.csv")
#real_data = real_data.sample(10000)
## get the unique nodes for real data
real_target_nodes = real_data[["target_id_{}".format(i) for i in range(6)]]
real_target_nodes.columns = ["id_{}".format(i) for i in range(6)]
real_source_nodes = real_data[["source_id_{}".format(i) for i in range(6)]]
real_source_nodes.columns = ["id_{}".format(i) for i in range(6)]
real_nodes = pd.concat((real_target_nodes, real_source_nodes), axis= 0).drop_duplicates()
num_real_nodes = real_nodes.shape[0]
## calculate average distance between the nodes for real data
scaler = StandardScaler()
real_nodes_scaled = pd.DataFrame(scaler.fit_transform(real_nodes), columns=real_nodes.columns)
real_nodes_avg_distance = []
for i, chunk in tqdm(real_nodes_scaled.groupby(np.arange(len(real_nodes_scaled))//15000), desc= "Calculating Average Distance between Real Nodes"):
    real_nodes_avg_distance.append(np.mean(cdist(real_nodes_scaled.drop(chunk.index).values, chunk.values, 'euclid')))
real_nodes_avg_distance = np.mean(real_nodes_avg_distance)
real_nodes_per_datapoint = num_real_nodes / real_data.shape[0]

if not os.path.exists("./results"):
    os.makedirs("./results")

# transform data back into graph structure
for model in models:
    print("Currently Processing Model: {}".format(model))

    with open("./model/{}.pkl".format(model), 'rb') as file:
        synthesizer = pickle.load(file)

    synthesizer.device = "cuda"

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
    synthetic_nodes_scaled = pd.DataFrame(scaler.fit_transform(synthetic_nodes), columns=synthetic_nodes.columns)

    if not os.path.exists("./synth/{}_synthetic_data_graph_distance.csv".format(model)):
        for n_clusters in tqdm(range(int(num_real_nodes*0.25), int(num_real_nodes*1.5), 30000), desc="Search for optimal Number of Nodes"):
            if not os.path.exists("./working/kmeans_{}/kmeans_{}_{}.pkl".format(model, model, n_clusters)):
                kms = MiniBatchKMeans(n_clusters= n_clusters, init= "k-means++", n_init= "auto", batch_size= 8192)
                kms.fit(synthetic_nodes_scaled)
                if not os.path.exists("./working/kmeans_{}".format(model)):
                    os.makedirs("./working/kmeans_{}".format(model))
                with open("./working/kmeans_{}/kmeans_{}_{}.pkl".format(model, model, n_clusters), 'wb') as f:
                    pickle.dump(kms, f)
            else:
                with open("./working/kmeans_{}/kmeans_{}_{}.pkl".format(model, model, n_clusters), 'rb') as f:
                    kms = pickle.load(f)

            new_synthetic_nodes = pd.DataFrame(kms.cluster_centers_)

            synthetic_nodes_avg_distance = []
            for i, chunk in new_synthetic_nodes.groupby(np.arange(len(new_synthetic_nodes)) // 15000):
                synthetic_nodes_avg_distance.append(np.mean(cdist(new_synthetic_nodes.drop(chunk.index).values, chunk.values, 'euclid')))
            synthetic_nodes_avg_distance = np.mean(synthetic_nodes_avg_distance)

            synth_nodes_avg_distances[n_clusters] = synthetic_nodes_avg_distance

        n_clusters = min(synth_nodes_avg_distances, key=lambda x:abs(x- real_nodes_avg_distance))
        with open("./working/kmeans_{}/kmeans_{}_{}.pkl".format(model, model, n_clusters), 'rb') as f:
            kms = pickle.load(f)
        synthetic_nodes_distance = synthetic_nodes
        synthetic_nodes_distance["node_id"] = kms.predict(synthetic_nodes_scaled)

        synthetic_data_distance = synthetic_data.merge(synthetic_nodes_distance, left_on= ["target_id_{}".format(i) for i in range(6)], right_on= ["id_{}".format(i) for i in range(6)], how="left", suffixes= ("", "_target"))
        synthetic_data_distance = synthetic_data_distance.merge(synthetic_nodes_distance, left_on=["source_id_{}".format(i) for i in range(6)], right_on=["id_{}".format(i) for i in range(6)], how="left", suffixes= ("", "_source"))

        synthetic_data_distance.to_csv("./synth/{}_synthetic_data_graph_distance.csv".format(model), index=False)

    if not os.path.exists("./synth/{}_synthetic_data_graph_number.csv".format(model)):
        n_clusters = int(real_nodes_per_datapoint * synthetic_data.shape[0])
        kms = MiniBatchKMeans(n_clusters=n_clusters, init="k-means++", n_init="auto", batch_size=8192)
        kms.fit(synthetic_nodes_scaled)
        synthetic_nodes_number = synthetic_nodes
        synthetic_nodes_number["node_id"] = kms.predict(synthetic_nodes_scaled)

        if "source_id" in synthetic_data.columns:
            synthetic_data = synthetic_data.drop(columns= ["source_id"])
        synthetic_nodes_number = synthetic_nodes_number.rename(columns= dict(zip(["id_{}".format(i) for i in range(6)] + ["node_id"], ["target_id_{}".format(i) for i in range(6)] + ["target_id"])))
        synthetic_data_graph_number = synthetic_data.merge(synthetic_nodes_number, on=["target_id_{}".format(i) for i in range(6)], how="left", suffixes=("", "_target"))
        synthetic_nodes_number = synthetic_nodes_number.rename(columns=dict(zip(["target_id_{}".format(i) for i in range(6)] + ["target_id"], ["source_id_{}".format(i) for i in range(6)] + ["source_id"])))
        synthetic_data_graph_number = synthetic_data_graph_number.merge(synthetic_nodes_number, on=["source_id_{}".format(i) for i in range(6)], how="left", suffixes=("", "_source"))

        synthetic_data_graph_number.to_csv("./synth/{}_synthetic_data_graph_number.csv".format(model), index=False)