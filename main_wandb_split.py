import os.path

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from modified_sitepackages.sdv.sequential import PARSynthesizer, DOPPELGANGERSynthesizer, BANKSFORMERSynthesizer
from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, WGANGPSynthesizer, WGANGP_DRSSynthesizer, FINDIFFSynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality, evaluate_similarity

import wandb

## setup wandb
#wandb.login()
wandb_project = "EvalGenerationAlgorithms_split"

add_transaction_clusters = True

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
if not os.path.exists("./working/transformed_pca_extd_df_split.csv"):
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
        # print(len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0))

    real_data = real_data.progress_apply(lambda row: split_single_transactions(row, "source_id", "target_id"), axis=1)
    real_data = pd.concat(real_data.to_list()).reset_index(drop=True)
    real_data["Id"] = real_data["Id"].astype(int).astype(str)
    real_data.to_csv("./working/transformed_pca_extd_df_split.csv", index=False)


real_data = pd.read_csv("./working/transformed_pca_extd_df_split.csv")
real_data = real_data.drop(columns= ["timeIndicator"])
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='isSender', sdtype='boolean')
metadata.update_column(column_name='Id', sdtype='id')

## create result df to store values
result_df = pd.DataFrame(columns= ["Algorithm", "Data Validity", "Data Structure", "Column Shapes", "Column Pair Trends"])

## Test WGAN-GP
#wandb.init(project=wandb_project, notes= "Performance Evaluation WGAN-GP", tags= ["WGAN-GP", "Priority3"], entity="financialDataGeneration")
#wandb.config = {"epochs": 500, "batch_size": 5000}
## Priority 3
#synthesizer = WGANGPSynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
#synthesizer.fit(data=real_data)
#synthetic_data = synthesizer.sample(num_rows=10000)
#synthetic_data.to_csv("./synth/WGANGP_split.csv", index=False)
#print(synthetic_data.head())
#diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
#report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
#report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
#report["Algorithm"] = "WGANGP"
#result_df = pd.concat((result_df, report))
#wandb.finish()

## Test WGAN-GP with DRS
#wandb.init(project=wandb_project, notes= "Performance Evaluation WGAN-GP with DRS", tags= ["WGAN-GPwDRS", "Priority1"], entity="financialDataGeneration")
#wandb.config = {"epochs": 500, "batch_size": 5000}
## Priority 1
#synthesizer = WGANGP_DRSSynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
#synthesizer.fit(data=real_data)
#synthetic_data = synthesizer.sample(num_rows=10000)
#synthetic_data.to_csv("./synth/WGANGP-DRS_split.csv", index=False)
#print(synthetic_data.head())
#diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
#report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
#report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
#report["Algorithm"] = "WGANGP-DRS"
#result_df = pd.concat((result_df, report))
#wandb.finish()

## Test CTGAN
#wandb.init(project=wandb_project, notes= "Performance Evaluation CTGAN", tags= ["CTGAN", "Priority1"], entity="financialDataGeneration")
#wandb.config = {"epochs": 500, "batch_size": 5000}
## Priority 1
#synthesizer = CTGANSynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
#synthesizer.fit(data=real_data)
#synthetic_data = synthesizer.sample(num_rows=10000)
#synthetic_data.to_csv("./synth/CTGAN_split.csv", index=False)
#print(synthetic_data.head())
#diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
#report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
#report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
#report["Algorithm"] = "CTGAN"
#result_df = pd.concat((result_df, report))
#wandb.finish()

## Test TVAE
#wandb.init(project=wandb_project, notes= "Performance Evaluation TVAE", tags= ["TVAE", "Priority1"], entity="financialDataGeneration")
#wandb.config = {"epochs": 500, "batch_size": 5000}
## Priority 1
#synthesizer = TVAESynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
#synthesizer.fit(data=real_data)
#synthetic_data = synthesizer.sample(num_rows=10000)
#synthetic_data.to_csv("./synth/TVAE_split.csv", index=False)
#print(synthetic_data.head())
#diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
#report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
#report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
#report["Algorithm"] = "TVAE"
#result_df = pd.concat((result_df, report))
#wandb.finish()

## Test FinDiff
#wandb.init(project=wandb_project, notes= "Performance Evaluation FinDiff", tags= ["FinDiff", "Priority1"], entity="financialDataGeneration")
#wandb.config = {"epochs": 500, "batch_size": 5000}
## Priority 1
#synthesizer = FINDIFFSynthesizer(metadata, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
#synthesizer.fit(data=real_data)
#synthetic_data = synthesizer.sample(num_rows=10000)
#synthetic_data.to_csv("./synth/FinDiff_split.csv", index=False)
#diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#similarity_report = evaluate_similarity(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
#wandb.log({**diagnostic_report.get_properties().set_index("Property")["Score"].to_dict(), **quality_report.get_properties().set_index("Property")["Score"].to_dict(), **similarity_report.get_properties().set_index("Property")["Score"].to_dict()})
#report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties(), similarity_report.get_properties())).transpose()
#report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
#report["Algorithm"] = "FinDiff"
#result_df = pd.concat((result_df, report))
#wandb.finish()


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

## Test DoppelGANger
wandb.init(project=wandb_project, notes= "Performance Evaluation DoppelGANger", tags= ["DoppelGANger", "Priority1"], entity="financialDataGeneration")
wandb.config = {"epochs": 500, "batch_size": 5000}
### Priority 1
synthesizer = DOPPELGANGERSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, sample_len= 5, batch_size= wandb.config["batch_size"], epochs= wandb.config["epochs"], verbose= True, use_wandb= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows= 10000)
synthetic_data.to_csv("./synth/DoppelGANger_split.csv", index=False)
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

result_df.to_excel("./working/evaluation_results_split.xlsx")
print(result_df)