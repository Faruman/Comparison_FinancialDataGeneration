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
wandb_project = "EvalGenerationAlgorithms_graph"

min_number_edges_per_node = 2
embedding_dim = 6


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
sweep_config = {
    "name": "RealBank",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "Jensen Shannon Distance"},
    "parameters": {
        "sample_len": {"values": [3, 5, 10, 15]},
        "attribute_noise_dim": {"min": 5, "max": 15},
        "feature_noise_dim": {"min": 5, "max": 15},
        "attribute_num_layers": {"min": 2, "max": 5},
        "attribute_num_units": {"min": 128, "max": 512},
        "feature_num_layers": {"min": 2, "max": 5},
        "feature_num_units": {"min": 128, "max": 512},
        "gradient_penalty_coef": {"min": 5.0, "max": 15.0},
        "attribute_gradient_penalty_coef": {"min": 5.0, "max": 15.0},
        "attribute_loss_coef": {"min": 0.5, "max": 3.0},
        "generator_learning_rate": {"min": 0.00001, "max": 0.005},
        "generator_beta1": {"min": 0.2, "max": 1.0},
        "discriminator_learning_rate": {"min": 0.00001, "max": 0.005},
        "discriminator_beta1": {"min": 0.2, "max": 1.0},
        "attribute_discriminator_learning_rate": {"min": 0.00001, "max": 0.005},
        "attribute_discriminator_beta1": {"min": 0.2, "max": 1.0},
        "discriminator_rounds": {"min": 1, "max": 10},
        "epochs": {"min": 100, "max": 1000},
        "batch_size": {"values": [5000]}
    },
}
sweep_id = wandb.sweep(sweep=sweep_config, project="FinancialDataGeneration_DOPPELGANGER_ParamSearch", entity="financialDataGeneration")

### Priority 1
def main():
    wandb.init(project="FinancialDataGeneration_DOPPELGANGER_ParamSearch", entity="financialDataGeneration")
    synthesizer = DOPPELGANGERSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, sample_len= wandb.config["sample_len"], feature_noise_dim = wandb.config["feature_noise_dim"], attribute_noise_dim= wandb.config["attribute_noise_dim"], attribute_num_layers = wandb.config["attribute_num_layers"],
                                          attribute_num_units = wandb.config["attribute_num_units"], feature_num_layers = wandb.config["feature_num_layers"], feature_num_units = wandb.config["feature_num_units"], gradient_penalty_coef = wandb.config["gradient_penalty_coef"],
                                          attribute_gradient_penalty_coef = wandb.config["attribute_gradient_penalty_coef"], attribute_loss_coef = wandb.config["attribute_loss_coef"], generator_learning_rate = wandb.config["generator_learning_rate"], generator_beta1 = wandb.config["generator_beta1"],
                                          discriminator_learning_rate = wandb.config["discriminator_learning_rate"], discriminator_beta1 = wandb.config["discriminator_beta1"], attribute_discriminator_learning_rate = wandb.config["attribute_discriminator_learning_rate"],
                                          attribute_discriminator_beta1 = wandb.config["attribute_discriminator_beta1"], discriminator_rounds = wandb.config["discriminator_rounds"], batch_size= wandb.config["batch_size"],
                                          epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
    synthesizer.fit(data=real_data)
    synthetic_data = synthesizer.sample(num_rows=10000)
    diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
    wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
    wandb.finish()

wandb.agent(sweep_id, function=main, count=30)