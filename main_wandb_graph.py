import os.path

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

import networkx as nx
import stellargraph as sg
from stellargraph.data import UnsupervisedSampler

from modified_sitepackages.sdv.sequential import PARSynthesizer, DOPPELGANGERSynthesizer, BANKSFORMERSynthesizer
from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, WGANGPSynthesizer, WGANGP_DRSSynthesizer, FINDIFFSynthesizer
from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb


class GraphConstruction:
    """
    This class initializes a networkX graph

     Parameters
    ----------
    nodes : dict(str, iterable)
        A dictionary with keys representing the node type, values representing
        an iterable container of nodes (list, dict, set etc.)
    edges : 2-tuples (u,v) or 3-tuples (u,v,d)
        Each edge given in the container will be added to the graph.
    features: dict(str, (str/dict/list/Dataframe)
        A dictionary with keys representing node type, values representing the node
        data.

    """

    g_nx = None
    node_features = None

    def __init__(self, nodes, edges, features=None):
        self.g_nx = nx.Graph()
        self.add_nodes(nodes)
        self.add_edges(edges)

        if features is not None:
            self.node_features = features

    def add_nodes(self, nodes):

        for key, values in nodes.items():
            self.g_nx.add_nodes_from(values, ntype=key)

    def add_edges(self, edges):

        for edge in edges:
            self.g_nx.add_edges_from(edge)

    def get_stellargraph(self):
        return sg.StellarGraph(self.g_nx, node_type_name="ntype", node_features=self.node_features)

    def get_edgelist(self):
        edgelist = []
        for edge in nx.generate_edgelist(self.g_nx):
            edgelist.append(str(edge).strip('{}'))
        el = []
        for edge in edgelist:
            splitted = edge.split()
            numeric = map(float, splitted)
            el.append(list(numeric))
        return el


## setup wandb
wandb.login(key= "46ccf115437ee40731bb49ccce1e7d4886329fe4", relogin= True)
wandb_project = "EvalGenerationAlgorithms_graph"

## replace source_id and target_id with graph structure of ids
if not os.path.exists("./working/transformed_pca_extd_df_graph.csv"):
    real_data = pd.read_csv("./data/transformed_pca_extd_df.csv", index_col=0)

    real_data = real_data.reset_index()
    real_data["index"] = pd.to_numeric(real_data["index"]).astype(int)
    real_data = real_data.rename(columns={"index": "timeIndicator"})

    real_data_graph = real_data.groupby(by= ["source_id", "target_id"])["timeIndicator"].count().reset_index()
    nodes = {"account": pd.concat([real_data_graph["source_id"], real_data_graph["target_id"]]).unique(), "transaction": real_data_graph.index}
    edges = [zip(real_data_graph["source_id"], real_data_graph.index), zip(real_data_graph["target_id"], real_data_graph.index)]
    account_node_data = pd.DataFrame([1] * len(nodes["account"])).set_index(nodes["account"])
    transaction_node_data = real_data_graph["timeIndicator"]
    features = {"transaction": transaction_node_data, "account": account_node_data}

    G = GraphConstruction(nodes, edges, features)
    # filter nodes with at least 3 connections
    G.g_nx = nx.subgraph(G.g_nx,[x for x in G.nodes() if node_degree_dict[x]>3])
    S = G.get_stellargraph()

    nodes = list(S.nodes())
    number_of_walks = 1
    length = 5
    unsupervised_samples = UnsupervisedSampler(S, nodes=nodes, length=length, number_of_walks=number_of_walks)

    batch_size = 50
    epochs = 4
    num_samples = [10, 5]
    generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)

    layer_sizes = [50, 50]
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
    )

    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    model.fit(train_gen, epochs=epochs, verbose=1, use_multiprocessing=False, workers=4, shuffle=True)

    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    node_ids = node_subjects.index
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)

    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

    real_data.to_csv("./working/transformed_pca_extd_df_graph.csv", index=False)

real_data = pd.read_csv("./working/transformed_pca_extd_df_graph.csv")

def split_single_transactions(row_sender, sender_column, receiver_column):
    row_receiver = row_sender.copy()
    row_sender.drop(receiver_column, inplace=True)
    row_sender["isSender"] = True
    row_sender.rename({sender_column: "Id"}, inplace=True)
    row_receiver.drop(sender_column, inplace=True)
    row_receiver["isSender"] = False
    row_receiver.rename({receiver_column: "Id"}, inplace=True)
    return pd.concat([row_sender, row_receiver], axis= 1).transpose()

## load data
# use split data
split = True
if split:
    if not os.path.exists("./working/transformed_pca_extd_df_split.csv"):
        real_data = pd.read_csv("./data/transformed_pca_extd_df.csv", index_col=0)
        real_data = real_data.reset_index()
        real_data["index"] = pd.to_numeric(real_data["index"]).astype(int)
        real_data = real_data.rename(columns={"index": "timeIndicator"})
        real_data = real_data.progress_apply(lambda row: split_single_transactions(row, "source_id", "target_id"), axis=1)
        real_data = pd.concat(real_data.to_list()).reset_index(drop=True)
        real_data["Id"] = real_data["Id"].astype(int).astype(str)
        real_data.to_csv("./working/transformed_pca_extd_df_split.csv", index=False)
else:
    if not os.path.exists("./working/transformed_pca_extd_df.csv"):
        real_data = real_data.rename(columns={"source_id": "Id"})
        real_data["Id"] = real_data["Id"].astype(int).astype(str)
        real_data["target_id"] = real_data["target_id"].astype(int).astype(str)
        real_data.to_csv("./working/transformed_pca_extd_df.csv", index=False)

if split:
    real_data = pd.read_csv("./working/transformed_pca_extd_df_split.csv")
    real_data = real_data.drop(columns=["timeIndicator"])
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    metadata.update_column(column_name='isSender', sdtype='boolean')
else:
    real_data = pd.read_csv("./working/transformed_pca_extd_df.csv")
    real_data = real_data.drop(columns=["timeIndicator"])
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
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/WGANGP.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log(diagnostic_report.get_properties().set_index("Property")["Score"].to_dict() | quality_report.get_properties().set_index("Property")["Score"].to_dict())
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
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
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/WGANGP-DRS.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log(diagnostic_report.get_properties().set_index("Property")["Score"].to_dict() | quality_report.get_properties().set_index("Property")["Score"].to_dict())
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
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
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/CTGAN.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log(diagnostic_report.get_properties().set_index("Property")["Score"].to_dict() | quality_report.get_properties().set_index("Property")["Score"].to_dict())
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
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
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/TVAE.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log(diagnostic_report.get_properties().set_index("Property")["Score"].to_dict() | quality_report.get_properties().set_index("Property")["Score"].to_dict())
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
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
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/FinDiff.csv", index=False)
wandb.log(diagnostic_report.get_properties().set_index("Property")["Score"].to_dict() | quality_report.get_properties().set_index("Property")["Score"].to_dict())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "FinDiff"
result_df = pd.concat((result_df, report))
wandb.finish()


if split:
    real_data = pd.read_csv("./working/transformed_pca_extd_df_split.csv")
    real_data = real_data.drop(columns=["timeIndicator"])
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    metadata.update_column(column_name='isSender', sdtype='boolean')
else:
    real_data = pd.read_csv("./working/transformed_pca_extd_df.csv")
    real_data = real_data.drop(columns=["timeIndicator"])
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)

metadata.update_column(column_name='Id', sdtype='id')
metadata.set_sequence_key(column_name='Id')
metadata.set_sequence_index(column_name='timeIndicator')
context_columns= []

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
real_data = real_data.groupby("Id").progress_apply(truncate_sequence, max_len= 30, min_len= 5, id_column= "Id").reset_index(drop=True)

## Test DoppelGANger
wandb.init(project=wandb_project, notes= "Performance Evaluation DoppelGANger", tags= ["DoppelGANger", "Priority1"], entity="financialDataGeneration")
wandb.config = {"epochs": 500, "batch_size": 5000}
### Priority 1
synthesizer = DOPPELGANGERSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, sample_len= 5, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows= 500)
synthetic_data.to_csv("./synth/DoppelGANger.csv", index=False)
wandb.log(diagnostic_report.get_properties().set_index("Property")["Score"].to_dict() | quality_report.get_properties().set_index("Property")["Score"].to_dict())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
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

result_df.to_excel("./working/evaluation_results.xlsx")
print(result_df)