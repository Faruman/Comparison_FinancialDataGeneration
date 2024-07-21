import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.sequential import DOPPELGANGERSynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality, evaluate_similarity

import wandb

## setup wandb
#wandb.login()
wandb_project = "FinancialDataGeneration_DOPPELGANGER_Evaluation"

min_number_edges_per_node = 2
embedding_dim = 6


if not os.path.exists("./model/"):
    os.makedirs("./model/")
if not os.path.exists("./synth/"):
    os.makedirs("./synth/")

## load data
real_data = pd.read_csv("./working/transformed_pca_extd_df_graph.csv")
real_data = real_data.drop(columns=["target_id"])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='source_id', sdtype='id')
metadata.update_column(column_name='timeIndicator', sdtype='numerical')
metadata.set_sequence_key(column_name='source_id')
metadata.set_sequence_index(column_name='timeIndicator')
metadata.set_primary_key(None)
if os.path.exists("./working/transformed_pca_extd_df_graph_metadata_series.json"):
    os.remove("./working/transformed_pca_extd_df_graph_metadata_series.json")
metadata.save_to_json("./working/transformed_pca_extd_df_graph_metadata_series.json")
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


wandb.init(project=wandb_project, entity="financialDataGeneration", tags=["RealBank"])
synthesizer = DOPPELGANGERSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, sample_len= 10, feature_noise_dim = 11, attribute_noise_dim= 10, attribute_num_layers = 3,
                                      attribute_num_units = 139, feature_num_layers = 5, feature_num_units = 287, gradient_penalty_coef = 9.035,
                                      attribute_gradient_penalty_coef = 8.21, attribute_loss_coef = 2.048, generator_learning_rate = 0.001833, generator_beta1 = 0.3226,
                                      discriminator_learning_rate = 0.002267, discriminator_beta1 = 0.5658, attribute_discriminator_learning_rate = 0.000199,
                                      attribute_discriminator_beta1 = 0.3048, discriminator_rounds = 2, batch_size= 5000,
                                      epochs= 575, verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthesizer.save("./model/DOPPELGANGER.pkl")
synthesizer.load("./model/DOPPELGANGER.pkl")
synthetic_data = synthesizer.sample(num_rows=100000)
synthetic_data.to_csv("./synth/DOPPELGANGER_synthetic_data.csv", index=False)
diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
wandb.finish()