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

from modified_sitepackages.sdv.sequential import DOPPELGANGERSynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality, evaluate_similarity

import wandb

## setup wandb
#wandb.login()
wandb_project = "FinancialDataGeneration_DOPPELGANGER_Evaluation"

min_number_edges_per_node = 2
embedding_generator = "watchyourstep"
embedding_dim = 5

add_transaction_clusters = True

data_path = "../data/transformed_pca_extd_df.csv"

## replace source_id and target_id with graph structure of ids
if not os.path.exists("../working/transformed_pca_extd_df_graph.csv"):
    real_data = pd.read_csv(data_path, index_col=0)
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

    real_data.to_csv("../working/transformed_pca_extd_df_graph.csv", index=False)

## load data
real_data = pd.read_csv("../working/transformed_pca_extd_df_graph.csv")
real_data = real_data.drop(columns=["target_id"])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='source_id', sdtype='id')
metadata.update_column(column_name='timeIndicator', sdtype='numerical')
metadata.set_sequence_key(column_name='source_id')
metadata.set_sequence_index(column_name='timeIndicator')
metadata.set_primary_key(None)
context_columns= [f"source_id_{i}" for i in range(embedding_dim)]


## Truncate sequences
def truncate_sequence(group, max_len, min_len, id_column):
    if len(group) <= max_len and len(group) >= min_len:
        group[id_column] = group[id_column].apply(lambda x: f"{x}_0")
        return group
    elif len(group) > max_len:
        out = pd.DataFrame(columns=group.columns)
        for i in range(len(group) // max_len):
            seq = group.sample(min(len(group), max_len))
            seq[id_column] = seq[id_column].apply(lambda x: f"{x}_{i}")
            if out.empty:
                out = seq
            else:
                out = pd.concat((out, seq))
            group = group.drop(seq.index)
        return out
    else:
        return pd.DataFrame(columns=group.columns)
real_data = real_data.groupby(["source_id"] + context_columns).progress_apply(truncate_sequence, max_len= 30, min_len= min_number_edges_per_node, id_column= "source_id").reset_index(drop=True)

## Test CTGAN
### Priority 1
wandb.init(project=wandb_project, entity="financialDataGeneration")
synthesizer = DOPPELGANGERSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, sample_len= 10, feature_noise_dim = 11, attribute_num_layers = 3,
                                      attribute_num_units = 139, feature_num_layers = 5, feature_num_units = 287, gradient_penalty_coef = 9.035,
                                      attribute_gradient_penalty_coef = 8.21, attribute_loss_coef = 2.048, generator_learning_rate = 0.001833, generator_beta1 = 0.3226,
                                      discriminator_learning_rate = 0.002267, discriminator_beta1 = 0.5658, attribute_discriminator_learning_rate = 0.000199,
                                      attribute_discriminator_beta1 = 0.3048, discriminator_rounds = 2, batch_size= 5000,
                                      epochs= 575, verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=10000)
diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
wandb.finish()