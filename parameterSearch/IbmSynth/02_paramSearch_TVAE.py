import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from modified_sitepackages.sdv.single_table import TVAESynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality, evaluate_similarity

import wandb


## setup wandb
#wandb.login()
wandb_project = "EvalGenerationAlgorithms_graph"

embedding_dim = 6


real_data = pd.read_csv("./working/transformed_pca_extd_df_graph.csv")
real_data = real_data.sample(100000)
real_data["timeIndicator"] = pd.to_datetime(real_data["timeIndicator"])
real_data = real_data.drop(columns=["source_id", "target_id"])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column("transaction_clusters", sdtype='categorical')

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
sweep_id = "financialDataGeneration/FinancialDataGeneration_TVAE_ParamSearch/tl4skgvg"

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