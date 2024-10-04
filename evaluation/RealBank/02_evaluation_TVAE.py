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
wandb_project = "FinancialDataGeneration_TVAE_Evaluation"

embedding_dim = 6


if not os.path.exists("./model/"):
    os.makedirs("./model/")
if not os.path.exists("./synth/"):
    os.makedirs("./synth/")


real_data = pd.read_csv("./working/transformed_pca_extd_df_graph.csv")
real_data = real_data.reset_index()
real_data["index"] = pd.to_numeric(real_data["index"]).astype(int)
real_data = real_data.rename(columns={"index": "timeIndicator"})

real_data = real_data.drop(columns= ["timeIndicator"])
real_data = real_data.drop(columns=["source_id", "target_id"])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
if os.path.exists("./working/transformed_pca_extd_df_graph_metadata_table.json"):
    os.remove("./working/transformed_pca_extd_df_graph_metadata_table.json")
metadata.save_to_json("./working/transformed_pca_extd_df_graph_metadata_table.json")


wandb.init(project=wandb_project, entity="financialDataGeneration", tags=["RealBank"])
synthesizer = TVAESynthesizer(metadata, embedding_dim= 32, compress_dims= [512,512], decompress_dims= [128,128],
                                l2scale= 0.0006256, loss_factor= 1, learning_rate= 0.0006903,
                                epochs= 544, batch_size= 5000, verbose=True, use_wandb=True)
synthesizer.fit(data=real_data)
synthesizer.save("./model/TVAE.pkl")
synthesizer.load("./model/TVAE.pkl")
synthetic_data = synthesizer.sample(num_rows=100000)
synthetic_data.to_csv("./synth/TVAE_synthetic_data.csv", index=False)
diagnostic_report = run_diagnostic(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
wandb.finish()