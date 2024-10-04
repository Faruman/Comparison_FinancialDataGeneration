import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.single_table import CTGANSynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality, evaluate_similarity

import wandb

## setup wandb
#wandb.login()
wandb_project = "EvalGenerationAlgorithms_graph"

embedding_dim = 6


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
    "name": "RealBank",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "Jensen Shannon Distance"},
    "parameters": {
        "embedding_dim": {"values": [32, 64, 256]},
        "generator_dim": {"values": [(128, 128), (256, 256), (512, 512)]},
        "discriminator_dim": {"values": [(128, 128), (256, 256), (512, 512)]},
        "generator_lr": {"min": 0.00001, "max": 0.001},
        "generator_decay": {"min": 0.0, "max": 0.05},
        "discriminator_lr": {"min": 0.00001, "max": 0.001},
        "discriminator_decay": {"min": 0.0, "max": 0.05},
        "discriminator_steps": {"min": 1, "max": 15},
        "epochs": {"min": 100, "max": 1000},
        "pac": {"values": [1, 2, 4, 8, 10, 20]},
        "batch_size": {"values": [5000]}
    },
}
sweep_id = wandb.sweep(sweep=sweep_config, project="FinancialDataGeneration_CTGAN_ParamSearch", entity="financialDataGeneration")

### Priority 1
def main():
    wandb.init(project="FinancialDataGeneration_CTGAN_ParamSearch", entity="financialDataGeneration")
    synthesizer = CTGANSynthesizer(metadata, embedding_dim= wandb.config["embedding_dim"], generator_dim= wandb.config["generator_dim"], discriminator_dim= wandb.config["discriminator_dim"],
                                    generator_lr= wandb.config["generator_lr"], generator_decay= wandb.config["generator_decay"], discriminator_lr= wandb.config["discriminator_lr"], discriminator_decay= wandb.config["discriminator_decay"], batch_size= wandb.config["batch_size"],
                                    epochs= wandb.config["epochs"], discriminator_steps= wandb.config["discriminator_steps"], pac= wandb.config["pac"], verbose=True, use_wandb=True)
    synthesizer.fit(data=real_data)
    synthetic_data = synthesizer.sample(num_rows=10000)
    diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
    similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
    wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
    wandb.finish()

wandb.agent(sweep_id, function=main, count=30)