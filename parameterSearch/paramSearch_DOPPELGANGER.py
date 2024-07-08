import os.path

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.sequential import DOPPELGANGERSynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb

## setup wandb
wandb.login(key= "46ccf115437ee40731bb49ccce1e7d4886329fe4", relogin= True)
wandb_project = "FinancialDataGeneration_DOPPELGANGER_ParamSearch"

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
if not os.path.exists("../working/transformed_pca_extd_df_split.csv"):
    real_data = pd.read_csv("../data/transformed_pca_extd_df.csv", index_col=0)
    real_data = real_data.reset_index()
    real_data["index"] = pd.to_numeric(real_data["index"]).astype(int)
    real_data = real_data.rename(columns={"index": "timeIndicator"})
    real_data = real_data.progress_apply(lambda row: split_single_transactions(row, "source_id", "target_id"), axis=1)
    real_data = pd.concat(real_data.to_list()).reset_index(drop=True)
    real_data["Id"] = real_data["Id"].astype(int).astype(str)
    real_data.to_csv("../working/transformed_pca_extd_df_split.csv", index=False)

real_data = pd.read_csv("./working/transformed_pca_extd_df_split.csv")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='isSender', sdtype='boolean')
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

## Test CTGAN
sweep_config = {
    "name": "Param Search",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Column Pair Trends"},
    "parameters": {
        "sample_len": {"values": [3, 5, 10, 15]},
        "attribute_noise_dim": {"min": 5, "max": 30},
        "feature_noise_dim": {"min": 5, "max": 30},
        "attribute_num_layers": {"min": 2, "max": 6},
        "attribute_num_units": {"min": 128, "max": 512},
        "feature_num_layers": {"min": 2, "max": 6},
        "feature_num_units": {"min": 128, "max": 512},
        "gradient_penalty_coef": {"min": 3.0, "max": 21.0},
        "attribute_gradient_penalty_coef": {"min": 3.0, "max": 21.0},
        "attribute_loss_coef": {"min": 0.5, "max": 5},
        "generator_learning_rate": {"min": 0.00001, "max": 0.01},
        "generator_beta1": {"min": 0.2, "max": 1},
        "discriminator_learning_rate": {"min": 0.00001, "max": 0.01},
        "discriminator_beta1": {"min": 0.2, "max": 1},
        "attribute_discriminator_learning_rate": {"min": 0.00001, "max": 0.01},
        "attribute_discriminator_beta1": {"min": 0.2, "max": 1},
        "discriminator_rounds": {"min": 1, "max": 10},
        "generator_rounds": {"min": 1, "max": 10},
        "epochs": {"min": 100, "max": 1000},
        "batch_size": {"values": [5000]}
    },
}
sweep_id = wandb.sweep(sweep=sweep_config, project="FinancialDataGeneration_DOPPELGANGER_ParamSearch", entity="financialDataGeneration")

### Priority 1
def main():
    wandb.init(project="FinancialDataGeneration_ParamSearch", entity="financialDataGeneration")
    synthesizer = DOPPELGANGERSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, sample_len= wandb.config["sample_len"], feature_noise_dim = wandb.config["feature_noise_dim"], attribute_num_layers = wandb.config["attribute_num_layers"],
                                          attribute_num_units = wandb.config["attribute_num_units"], feature_num_layers = wandb.config["feature_num_layers"], feature_num_units = wandb.config["feature_num_units"], gradient_penalty_coef = wandb.config["gradient_penalty_coef"],
                                          attribute_gradient_penalty_coef = wandb.config["attribute_gradient_penalty_coef"], attribute_loss_coef = wandb.config["attribute_loss_coef"], generator_learning_rate = wandb.config["generator_learning_rate"], generator_beta1 = wandb.config["generator_beta1"],
                                          discriminator_learning_rate = wandb.config["discriminator_learning_rate"], discriminator_beta1 = wandb.config["discriminator_beta1"], attribute_discriminator_learning_rate = wandb.config["attribute_discriminator_learning_rate"],
                                          attribute_discriminator_beta1 = wandb.config["attribute_discriminator_beta1"], discriminator_rounds = wandb.config["discriminator_rounds"], generator_rounds = wandb.config["generator_rounds"], batch_size= wandb.config["batch_size"],
                                          epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
    synthesizer.fit(data=real_data)
    synthetic_data = synthesizer.sample(num_rows=500)
    diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    wandb.log(diagnostic_report.get_properties().set_index("Property")["Score"].to_dict() | quality_report.get_properties().set_index("Property")["Score"].to_dict())
    wandb.finish()

wandb.agent(sweep_id, function=main, count=10)