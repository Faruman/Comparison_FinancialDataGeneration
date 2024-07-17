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

from modified_sitepackages.sdv.sequential import PARSynthesizer, DOPPELGANGERSynthesizer, BANKSFORMERSynthesizer
from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, WGANGPSynthesizer, WGANGP_DRSSynthesizer, FINDIFFSynthesizer
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
    real_data = pd.read_csv("data/RealBank/transformed_pca_extd_df.csv", index_col=0)
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

## create result df to store values
result_df = pd.DataFrame(columns= ["Algorithm", "Data Validity", "Data Structure", "Column Shapes", "Column Pair Trends"])

## Test WGAN-GP
wandb.init(project=wandb_project, notes= "Performance Evaluation WGAN-GP", tags= ["WGAN-GP", "Priority3"], entity="financialDataGeneration")
wandb.config = {"epochs": 500, "batch_size": 5000}
### Priority 3
synthesizer = WGANGPSynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=10000)
synthetic_data.to_csv("./synth/WGANGP_graph.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "WGANGP"
result_df = pd.concat((result_df, report))
wandb.finish()

## Test WGAN-GP with DRS
wandb.init(project=wandb_project, notes= "Performance Evaluation WGAN-GP with DRS", tags= ["WGAN-GPwDRS", "Priority1"], entity="financialDataGeneration")
wandb.config = {"epochs": 500, "batch_size": 5000}
### Priority 1
synthesizer = WGANGP_DRSSynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=10000)
synthetic_data.to_csv("./synth/WGANGP-DRS_graph.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "WGANGP-DRS"
result_df = pd.concat((result_df, report))
wandb.finish()

## Test CTGAN
wandb.init(project=wandb_project, notes= "Performance Evaluation CTGAN", tags= ["CTGAN", "Priority1"], entity="financialDataGeneration")
wandb.config = {"epochs": 500, "batch_size": 5000}
### Priority 1
synthesizer = CTGANSynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=10000)
synthetic_data.to_csv("./synth/CTGAN_graph.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "CTGAN"
result_df = pd.concat((result_df, report))
wandb.finish()

## Test TVAE
wandb.init(project=wandb_project, notes= "Performance Evaluation TVAE", tags= ["TVAE", "Priority1"], entity="financialDataGeneration")
wandb.config = {"epochs": 500, "batch_size": 5000}
### Priority 1
synthesizer = TVAESynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=10000)
synthetic_data.to_csv("./synth/TVAE_graph.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "TVAE"
result_df = pd.concat((result_df, report))
wandb.finish()

## Test FinDiff
wandb.init(project=wandb_project, notes= "Performance Evaluation FinDiff", tags= ["FinDiff", "Priority1"], entity="financialDataGeneration")
wandb.config = {"epochs": 500, "batch_size": 5000}
### Priority 1
synthesizer = FINDIFFSynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=10000)
synthetic_data.to_csv("./synth/FinDiff_graph.csv", index=False)
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "FinDiff"
result_df = pd.concat((result_df, report))
wandb.finish()


## load data
real_data = pd.read_csv("./working/transformed_pca_extd_df_graph.csv")
real_data = real_data.drop(columns=["target_id"])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='source_id', sdtype='id')
metadata.update_column(column_name='timeIndicator', sdtype='numerical')
metadata.set_sequence_key(column_name='source_id')
metadata.set_sequence_index(column_name='timeIndicator')
context_columns= [f"source_id_{i}" for i in range(embedding_dim)]

## Truncate sequences
def truncate_sequence(group, max_len, min_len, id_column):
    if len(group) <= max_len and len(group) >= min_len:
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
real_data = real_data.groupby("source_id").progress_apply(truncate_sequence, max_len= 30, min_len= 5, id_column= "source_id").reset_index(drop=True)

## Test DoppelGANger
wandb.init(project=wandb_project, notes= "Performance Evaluation DoppelGANger", tags= ["DoppelGANger", "Priority1"], entity="financialDataGeneration")
wandb.config = {"epochs": 500, "batch_size": 5000}
### Priority 1
synthesizer = DOPPELGANGERSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, sample_len= 5, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows= 10000)
synthetic_data.to_csv("./synth/DoppelGANger_graph.csv", index=False)
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "DoppelGANger"
result_df = pd.concat((result_df, report))
wandb.finish()

## Test Banksformer
### Priority 1
#synthesizer = BANKSFORMERSynthesizer(metadata, context_columns= context_columns, amount_column="Open", max_sequence_len= 260, sample_len= 5, epochs= 500, verbose= True)
#synthesizer.fit(data=real_data)
#synthetic_data = synthesizer.sample(num_rows= 500)
#synthetic_data.to_csv("./synth/Banksformer.csv", index=False)
#print(synthetic_data.head())
#diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
#report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
#report["Algorithm"] = "Banksformer"
#result_df = pd.concat((result_df, report))

result_df.to_excel("./working/evaluation_results_graph.xlsx")
print(result_df)