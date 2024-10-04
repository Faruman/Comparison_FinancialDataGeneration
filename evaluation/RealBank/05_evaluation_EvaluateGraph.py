import os.path
import pickle

from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, cdist

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import networkx as nx

from matplotlib import pyplot as plt

from modified_sitepackages.netcomp import netsimile, deltacon0


#models = ['DOPPELGANGER', 'FINDIFF', 'TVAE', 'WGANGPwDRS', 'CTGAN']
models = ['TVAE']
keep_col = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
min_edges_plot = 3
min_edges_similarity = 3

real_data = pd.read_csv("./working/transformed_pca_extd_df_graph.csv")

if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./plots"):
    os.makedirs("./plots")

real_data = real_data.loc[~(real_data["source_id"] == real_data["target_id"])]
real_data = real_data.reset_index()[["source_id", "target_id", "index"]].groupby(["source_id", "target_id"]).count().reset_index().rename(columns= {"index": "weight"})
real_graph = nx.from_pandas_edgelist(real_data, source= "source_id", target= "target_id", edge_attr="weight", create_using= nx.DiGraph)

node_degree_dict=nx.degree(real_graph)
G_draw= nx.subgraph(real_graph,[x for x in real_graph.nodes() if node_degree_dict[x]>= min_edges_plot])
print('Number of edges: {}'.format(G_draw.number_of_edges()))
print('Number of nodes: {}'.format(G_draw.number_of_nodes()))
edge_weight = np.array(list(nx.get_edge_attributes(G_draw,'weight').values()))
edge_weight = 5 * ((edge_weight - edge_weight.min())/ (edge_weight.max() - edge_weight.min())) + 1
plt.figure()
plt.title("Graph Structure (Real Data)")
nx.draw(G_draw, width=edge_weight, node_size=5, with_labels= False)
plt.savefig("./plots/networkGraph_realData_minEdge{}.png".format(min_edges_plot))
plt.show()

node_degree_dict = nx.degree(real_graph)
real_graph = nx.subgraph(real_graph, [x for x in real_graph.nodes() if node_degree_dict[x] >= min_edges_similarity])

if os.path.exists("./results/evaluation_graph.xlsx"):
    results_df = pd.read_excel("./results/evaluation_graph.xlsx")
else:
    results_df = pd.DataFrame()

#Calculate the similarity in graph structure
for model in models:
    #for cluster_method in ["distance", "number"]:
    for cluster_method in ["number"]:
        synthetic_data = pd.read_csv("./synth/{}_synthetic_data_graph_{}.csv".format(model, cluster_method), encoding = "ISO-8859-1", low_memory=False)
        if "source_id" in synthetic_data.columns and ("node_id_source" in synthetic_data.columns or "node_id_y" in synthetic_data):
            synthetic_data = synthetic_data.drop(columns=["source_id"])
        if "node_id_x" in synthetic_data.columns and "node_id_y" in synthetic_data.columns:
            synthetic_data = synthetic_data.rename(columns= {"node_id_x": "source_id", "node_id_y": "target_id"})
        elif "node_id" in synthetic_data.columns and "node_id_source" in synthetic_data.columns:
            synthetic_data = synthetic_data.rename(columns={"node_id_source": "source_id", "node_id": "target_id"})
        elif "node_id_source" in synthetic_data.columns and "node_id_target" in synthetic_data.columns:
            synthetic_data = synthetic_data.rename(columns={"node_id_source": "source_id", "node_id_target": "target_id"})
        else:
            pass

        # create networkx graphs based on the data
        #synthetic_data = synthetic_data.loc[~(synthetic_data["source_id"] == synthetic_data["target_id"])]
        synthetic_data = synthetic_data.reset_index()[["source_id", "target_id", "index"]].groupby(["source_id", "target_id"]).count().reset_index().rename(columns= {"index": "weight"})
        synthetic_graph = nx.from_pandas_edgelist(synthetic_data, source= "source_id", target= "target_id", edge_attr="weight", create_using= nx.DiGraph)

        node_degree_dict = nx.degree(synthetic_graph)
        G_draw = nx.subgraph(synthetic_graph, [x for x in synthetic_graph.nodes() if node_degree_dict[x] >= min_edges_plot])
        print('Number of edges: {}'.format(G_draw.number_of_edges()))
        print('Number of nodes: {}'.format(G_draw.number_of_nodes()))
        edge_weight = np.array(list(nx.get_edge_attributes(G_draw, 'weight').values()))
        edge_weight = 5 * ((edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())) + 1
        plt.figure()
        plt.title("Graph Structure\n(Synth: {} / Method: {})".format(model, cluster_method))
        nx.draw(G_draw, width=edge_weight, node_size=5, with_labels=False)
        plt.savefig("./plots/networkGraph_synthdata_{}_{}_minEdge{}.png".format(model, cluster_method, min_edges_plot))
        plt.show()

        # create smaller graph for testing
        node_degree_dict = nx.degree(synthetic_graph)
        synthetic_graph = nx.subgraph(synthetic_graph, [x for x in synthetic_graph.nodes() if node_degree_dict[x] >= min_edges_similarity])

        # score NetSimile
        netsimile_distance = netsimile(nx.adjacency_matrix(real_graph), nx.adjacency_matrix(synthetic_graph))

        results_df = pd.concat((results_df, pd.DataFrame({"Model": model, "Method": cluster_method, "NetSimile": netsimile_distance, "DeltaCon0": deltacon0_distance}, index=  [model + "_" + cluster_method])), axis= 0)
        results_df.to_excel("./results/evaluation_graph.xlsx", index=False)

    results_df.to_excel("./results/evaluation_graph_minEdge{}.xlsx".format(min_edges_similarity), index=False)