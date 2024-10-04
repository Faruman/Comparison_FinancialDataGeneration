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
wandb_project = "FinancialDataGeneration_CTGAN_Evaluation"

embedding_dim = 6


if not os.path.exists("./model/"):
    os.makedirs("./model/")
if not os.path.exists("./synth/"):
    os.makedirs("./synth/")

real_data = pd.read_csv("./working/transformed_df_graph.csv")
real_data["timeIndicator"] = pd.to_datetime(real_data["timeIndicator"])
real_data = real_data.drop(columns=["source_id", "target_id"])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column("transaction_clusters", sdtype='categorical')
if os.path.exists("./working/transformed_pca_extd_df_graph_metadata_table.json"):
    os.remove("./working/transformed_pca_extd_df_graph_metadata_table.json")
metadata.save_to_json("./working/transformed_pca_extd_df_graph_metadata_table.json")


wandb.init(project= wandb_project, entity="financialDataGeneration", tags= ["IbmSynth"])
synthesizer = CTGANSynthesizer(metadata, embedding_dim= 256, generator_dim= [256,256], discriminator_dim= [512,512],
                                generator_lr= 0.00007396, generator_decay= 0.01233, discriminator_lr= 0.0001012, discriminator_decay= 0.00537, batch_size= 5000,
                                epochs= 457, discriminator_steps= 6, pac= 1, verbose=True, use_wandb=True)
synthesizer.fit(data=real_data)
synthesizer.save("./model/CTGAN.pkl")
synthesizer.load("./model/CTGAN.pkl")
synthetic_data = synthesizer.sample(num_rows=100000)
synthetic_data.to_csv("./synth/CTGAN_synthetic_data.csv", index=False)
diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
wandb.finish()