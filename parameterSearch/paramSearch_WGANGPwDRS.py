import os.path

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.sequential import PARSynthesizer, DOPPELGANGERSynthesizer, BANKSFORMERSynthesizer
from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, WGANGPSynthesizer, WGANGP_DRSSynthesizer, FINDIFFSynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

import wandb

## setup wandb
wandb.login(key= "46ccf115437ee40731bb49ccce1e7d4886329fe4", relogin= True)
wandb_project = "FinancialDataGeneration_WGANGPwDRS_ParamSearch"

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

real_data = pd.read_csv("../working/transformed_pca_extd_df_split.csv")
real_data = real_data.drop(columns= ["timeIndicator"])
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='isSender', sdtype='boolean')

## Test WGAN-GP with DRS
sweep_config = {
    "name": "Param Search",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Column Pair Trends"},
    "parameters": {
        "embedding_dim": {"values": [32, 64, 256]},
        "generator_dim": {"values": [(128, 256, 512), (256, 512, 512), (256, 512, 1048)]},
        "discriminator_dim": {"values": [(512, 256, 128), (512, 512, 256), (1048, 512, 256)]},
        "generator_lr": {"min": 0.00001, "max": 0.001},
        "generator_decay": {"min": 0.0, "max": 0.05},
        "discriminator_lr": {"min": 0.00001, "max": 0.001},
        "discriminator_decay": {"min": 0.0, "max": 0.05},
        "discriminator_steps": {"min": 1, "max": 15},
        "epochs": {"min": 100, "max": 1000},
        "pac": {"min": 1, "max": 20},
        "batch_size": {"values": [5000]},
        "dsr_epsilon": {"min": 0.00001, "max": 0.001},
        "dsr_gamma_percentile": {"min": 0.7, "max": 0.95}
    },
}
sweep_id = wandb.sweep(sweep=sweep_config, project="FinancialDataGeneration_WGANGPwDRS_ParamSearch", entity="financialDataGeneration")

### Priority 1
def main():
    wandb.init(project="FinancialDataGeneration_ParamSearch", entity="financialDataGeneration")
    synthesizer = WGANGP_DRSSynthesizer(metadata, embedding_dim= wandb.config["embedding_dim"], generator_dim= wandb.config["generator_dim"], discriminator_dim= wandb.config["discriminator_dim"],
                                    generator_lr= wandb.config["generator_lr"], generator_decay= wandb.config["generator_decay"], discriminator_lr= wandb.config["discriminator_lr"], discriminator_decay= wandb.config["discriminator_decay"], batch_size= wandb.config["batch_size"],
                                    epochs= wandb.config["epochs"], discriminator_steps= wandb.config["discriminator_steps"], pac= wandb.config["pac"], dsr_epsilon= wandb.config["dsr_epsilon"], dsr_gamma_percentile= 0.80, verbose=True, use_wandb=True)
    synthesizer.fit(data=real_data)
    synthetic_data = synthesizer.sample(num_rows=500)
    diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    wandb.log(diagnostic_report.get_properties().set_index("Property")["Score"].to_dict() | quality_report.get_properties().set_index("Property")["Score"].to_dict())
    wandb.finish()

wandb.agent(sweep_id, function=main, count=10)