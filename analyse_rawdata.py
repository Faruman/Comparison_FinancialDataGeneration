import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

import networkx as nx


df = pd.read_csv("./data/transformed_pca_extd_df.csv")
df = df.rename(columns={"Unnamed: 0": "datetimeIndicator"})

print("Number of unique Sources: {}".format(df["source_id"].nunique()))
print("Number of unique Receivers: {}".format(df["target_id"].nunique()))
print("Number of unique Accounts: {}".format(pd.concat((df["source_id"], df["target_id"])).nunique()))

# Analyse time series structure
df_transactions = pd.concat((df.rename(columns= {"source_id": "id"}).drop(columns= ["target_id"]), df.rename(columns= {"target_id": "id"}).drop(columns= ["source_id"])))
df_transactions["sender"] = False
df_transactions.iloc[:df.shape[0], -1] = True
sns.boxplot(df_transactions.groupby("id")["datetimeIndicator"].count())
plt.title("Analysis of Time Series Length")
plt.ylabel("Lengths of Time Series")
plt.savefig("./data/plots/lengthOfTimeSeries.png")
plt.show()

# Analyse graph structure
G = nx.DiGraph()
edgelist = df.loc[df["source_id"] != df["target_id"]].groupby(by= ["source_id", "target_id"])["datetimeIndicator"].count().reset_index()
edgelist = edgelist.rename(columns= {"datetimeIndicator": "count"}).values.tolist()
edgelist = [(x, y, {"count": z}) for x,y,z in edgelist]
G.add_edges_from(edgelist)
node_degree_dict=nx.degree(G)
G_draw= nx.subgraph(G,[x for x in G.nodes() if node_degree_dict[x]>5])
print('Number of edges: {}'.format(G_draw.number_of_edges()))
print('Number of nodes: {}'.format(G_draw.number_of_nodes()))
edge_weight = np.array(list(nx.get_edge_attributes(G_draw,'count').values()))
edge_weight = 10 * ((edge_weight - edge_weight.min())/ (edge_weight.max() - edge_weight.min())) + 1
plt.figure()
plt.title("Analysis of Graph Structure")
nx.draw(G_draw, width=edge_weight, node_size=5, with_labels= False)
plt.savefig("./data/plots/networkGraph.png")
plt.show()