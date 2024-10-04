import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

import networkx as nx
import stellargraph as sg
from stellargraph.mapper import GraphWaveGenerator
from stellargraph.mapper import AdjacencyPowerGenerator
from stellargraph.layer import WatchYourStep
from stellargraph.losses import graph_log_likelihood
from stellargraph.utils import plot_history

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Model, regularizers
import tensorflow as tf

from matplotlib import pyplot as plt


min_number_edges_per_node = 2
embedding_generator = "watchyourstep"
embedding_dim = 6

add_transaction_clusters = True

data_path = "../../data/IbmSynth/transformed_df.csv"

if not os.path.exists("./working/"):
    os.makedirs("./working/")

## replace source_id and target_id with graph structure of ids
if not os.path.exists("./working/test.csv"):
    real_data = pd.read_csv(data_path, index_col=0)
    real_data = real_data.reset_index()
    real_data["index"] = pd.to_numeric(real_data["index"]).astype(int)
    real_data = real_data.rename(columns={"index": "timeIndicator"})

    if add_transaction_clusters:
        if real_data.shape[0] > 500000:
            cl_data = StandardScaler().fit_transform(real_data.drop(["source_id", "target_id"], axis=1).sample(100000))
        else:
            cl_data = StandardScaler().fit_transform(real_data.drop(["source_id", "target_id"], axis=1))
        cl = KMeans(n_clusters=12)
        cl.fit(cl_data)
        real_data["transaction_clusters"] = cl.predict(real_data.drop(["source_id", "target_id"], axis=1))
        #print(len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0))

    G = nx.DiGraph()
    edgelist = real_data.loc[real_data["source_id"] != real_data["target_id"]].groupby(by=["source_id", "target_id"])["timeIndicator"].count().reset_index()
    edgelist = edgelist.rename(columns={"timeIndicator": "count"}).values.tolist()
    edgelist = [(x, y, {"count": z}) for x, y, z in edgelist]
    G.add_edges_from(edgelist)
    node_degree_dict = nx.degree(G)
    G = nx.subgraph(G, [x for x in G.nodes() if node_degree_dict[x] >= min_number_edges_per_node])
    S = sg.StellarGraph.from_networkx(G)

    if embedding_generator == "graphwave":
        # use graphwave to embed nodes (https://arxiv.org/pdf/1710.10321)
        sample_points = np.linspace(0, 100, 64).astype(np.float32)
        degree = 20
        scales = [5, 10]
        generator = GraphWaveGenerator(S, scales=scales, degree=degree)
        embeddings_dataset = generator.flow(node_ids=S.nodes(), sample_points=sample_points, batch_size=1, repeat=False)
        embeddings = [x.numpy() for x in embeddings_dataset]
        embeddings = np.squeeze(np.array(embeddings), axis=1)
        pca = PCA(n_components=embedding_dim)
        embeddings = pca.fit_transform(embeddings)

    elif embedding_generator == "watchyourstep":
        if len(tf.config.list_physical_devices('GPU')) > 0:
            with tf.device(tf.config.list_logical_devices('GPU')[0].name):
                # use watchyourstep to embed nodes (https://arxiv.org/pdf/1710.09599)
                generator = AdjacencyPowerGenerator(S, num_powers=10)
                wys = WatchYourStep(
                    generator,
                    num_walks=80,
                    embedding_dimension= embedding_dim,
                    attention_regularizer=regularizers.l2(0.5),
                )
                x_in, x_out = wys.in_out_tensors()
                model = Model(inputs=x_in, outputs=x_out)
                model.compile(loss=graph_log_likelihood, optimizer=tf.keras.optimizers.Adam(1e-3))
                batch_size = 64
                train_gen = generator.flow(batch_size=batch_size, num_parallel_calls=10)
                history = model.fit(train_gen, epochs=50, verbose=1, steps_per_epoch=int(len(S.nodes()) // batch_size))
        else:
            # use watchyourstep to embed nodes (https://arxiv.org/pdf/1710.09599)
            generator = AdjacencyPowerGenerator(S, num_powers=10)
            wys = WatchYourStep(
                generator,
                num_walks=80,
                embedding_dimension=embedding_dim,
                attention_regularizer=regularizers.l2(0.5),
            )
            x_in, x_out = wys.in_out_tensors()
            model = Model(inputs=x_in, outputs=x_out)
            model.compile(loss=graph_log_likelihood, optimizer=tf.keras.optimizers.Adam(1e-3))
            batch_size = 64
            train_gen = generator.flow(batch_size=batch_size, num_parallel_calls=10)
            history = model.fit(train_gen, epochs=100, verbose=1, steps_per_epoch=int(len(S.nodes()) // batch_size))
        embeddings = wys.embeddings()
        plot_history(history)
        plt.show()
    else:
        raise ValueError("Unknown embedding generator")

    # print(pca.explained_variance_ratio_)
    embeddings_dict = dict(zip(S.nodes(), embeddings))

    real_data = real_data.loc[real_data["source_id"].isin(S.nodes()) & real_data["target_id"].isin(S.nodes())].reset_index(drop=True)
    embeddings_source = pd.concat((pd.DataFrame(S.nodes(), columns= ["source_id"]), pd.DataFrame(embeddings, columns= [f"source_id_{i}" for i in range(embedding_dim)])), axis=1)
    embeddings_target = pd.concat((pd.DataFrame(S.nodes(), columns=["target_id"]), pd.DataFrame(embeddings, columns=[f"target_id_{i}" for i in range(embedding_dim)])), axis=1)
    real_data = pd.merge(real_data, embeddings_source, left_on= "source_id", right_on= "source_id", how= "left")
    real_data = pd.merge(real_data, embeddings_target, left_on= "target_id", right_on= "target_id", how= "left")

    real_data.to_csv("./working/transformed_df_graph.csv", index=False)