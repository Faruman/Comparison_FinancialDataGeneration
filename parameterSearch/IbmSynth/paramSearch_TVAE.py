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

from modified_sitepackages.sdv.single_table import TVAESynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality, evaluate_similarity

import wandb

## setup wandb
#wandb.login()
wandb_project = "EvalGenerationAlgorithms_graph"

min_number_edges_per_node = 2
embedding_generator = "watchyourstep"
embedding_dim = 5

add_transaction_clusters = True

## replace source_id and target_id with graph structure of ids
if not os.path.exists("./working/transformed_pca_extd_df_graph.csv"):
    real_data = pd.read_csv("../../data/IbmSynth/transformed_df.csv", index_col=0)
    real_data = real_data.reset_index()
    real_data["index"] = pd.to_numeric(real_data["index"]).astype(int)
    real_data = real_data.rename(columns={"index": "timeIndicator"})

    if add_transaction_clusters:
        if real_data.shape[0] > 500000:
            cl_data = StandardScaler().fit_transform(real_data.drop(["source_id", "target_id"], axis=1).sample(100000))
        else:
            cl_data = StandardScaler().fit_transform(real_data.drop(["source_id", "target_id"], axis=1))
        cl = KMeans(n_clusters=10)
        real_data["transaction_clusters"] = cl.fit_predict(cl_data)
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
    elif embedding_generator == "watchyourstep":
        # use watchyourstep to embed nodes (https://arxiv.org/pdf/1710.09599)
        generator = AdjacencyPowerGenerator(S, num_powers=10)
        wys = WatchYourStep(
            generator,
            num_walks=80,
            embedding_dimension= 128,
            attention_regularizer=regularizers.l2(0.5),
        )
        x_in, x_out = wys.in_out_tensors()
        model = Model(inputs=x_in, outputs=x_out)
        model.compile(loss=graph_log_likelihood, optimizer=tf.keras.optimizers.Adam(1e-3))
        batch_size = 64
        train_gen = generator.flow(batch_size=batch_size, num_parallel_calls=10)
        history = model.fit(train_gen, epochs=100, verbose=1, steps_per_epoch=int(len(S.nodes()) // batch_size))
        embeddings = wys.embeddings()
        #plot_history(history)
    else:
        raise ValueError("Unknown embedding generator")

    pca = PCA(n_components=embedding_dim)
    embeddings = pca.fit_transform(embeddings)
    # print(pca.explained_variance_ratio_)
    embeddings_dict = dict(zip(S.nodes(), embeddings))

    real_data = real_data.loc[real_data["source_id"].isin(S.nodes()) & real_data["target_id"].isin(S.nodes())].reset_index(drop= True)
    source_embeddings = pd.DataFrame(real_data["source_id"].progress_apply(lambda x: embeddings_dict[x]).to_list())
    source_embeddings.columns = [f"source_id_{i}" for i in range(embedding_dim)]
    target_embeddings = pd.DataFrame(real_data["target_id"].progress_apply(lambda x: embeddings_dict[x]).to_list())
    target_embeddings.columns = [f"target_id_{i}" for i in range(embedding_dim)]
    real_data = pd.concat((real_data, source_embeddings, target_embeddings), axis=1)

    real_data.to_csv("./working/transformed_pca_extd_df_graph.csv", index=False)

real_data = pd.read_csv("./working/transformed_pca_extd_df_graph.csv")
real_data = real_data.reset_index()
real_data["index"] = pd.to_numeric(real_data["index"]).astype(int)
real_data = real_data.rename(columns={"index": "timeIndicator"})

real_data = real_data.drop(columns= ["timeIndicator"])
real_data = real_data.drop(columns=["source_id", "target_id"])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

## Test CTGAN
sweep_config = {
    "name": "IbmSynth",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "Jensen Shannon Distance"},
    "parameters": {
        "embedding_dim": {"values": [32, 64, 256]},
        "compress_dims": {"values": [(128, 128), (256, 256), (512, 512)]},
        "decompress_dims": {"values": [(128, 128), (256, 256), (512, 512)]},
        "l2scale": {"min": 0.00001, "max": 0.001},
        "loss_factor": {"min": 1, "max": 5},
        "learning_rate": {"min": 0.00001, "max": 0.001},
        "epochs": {"min": 100, "max": 1000},
        "batch_size": {"values": [5000]}
    },
}
#sweep_id = wandb.sweep(sweep=sweep_config, project="FinancialDataGeneration_TVAE_ParamSearch", entity="financialDataGeneration")
sweep_id = "financialDataGeneration/FinancialDataGeneration_TVAE_ParamSearch/bh90ig0h"

### Priority 1
def main():
    wandb.init(project="FinancialDataGeneration_TVAE_ParamSearch", entity="financialDataGeneration")
    synthesizer = TVAESynthesizer(metadata, embedding_dim= wandb.config["embedding_dim"], compress_dims= wandb.config["compress_dims"], decompress_dims= wandb.config["decompress_dims"],
                                    l2scale= wandb.config["l2scale"], loss_factor= wandb.config["loss_factor"], learning_rate= wandb.config["learning_rate"],
                                    epochs= wandb.config["epochs"], batch_size= wandb.config["batch_size"], verbose=True, use_wandb=True)
    synthesizer.fit(data=real_data)
    synthetic_data = synthesizer.sample(num_rows=10000)
    diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
    wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
    wandb.finish()

wandb.agent(sweep_id, function=main, count=30)