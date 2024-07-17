import os.path

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sdv.metadata import SingleTableMetadata


from modified_sitepackages.sdv.sequential import DOPPELGANGERSynthesizer, BANKSFORMERSynthesizer
from modified_sitepackages.sdv.single_table import CTGANSynthesizer, TVAESynthesizer, WGANGPSynthesizer, WGANGP_DRSSynthesizer, FINDIFFSynthesizer

from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality

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
    real_data = real_data.progress_apply(lambda row: split_single_transactions(row, "source_id", "target_id"), axis=1)
    real_data = pd.concat(real_data.to_list()).reset_index(drop=True)
    real_data["Id"] = real_data["Id"].astype(int).astype(str)
    real_data.to_csv("./working/transformed_pca_extd_df_split.csv", index=False)


real_data = pd.read_csv("./working/transformed_pca_extd_df_split.csv")
real_data = real_data.drop(columns= ["timeIndicator"])
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
metadata.update_column(column_name='isSender', sdtype='boolean')

## create result df to store values
result_df = pd.DataFrame(columns= ["Algorithm", "Data Validity", "Data Structure", "Column Shapes", "Column Pair Trends"])

## Test WGAN-GP
### Priority 3
synthesizer = WGANGPSynthesizer(metadata, batch_size=1000, epochs= 400, verbose= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/WGANGP.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "WGANGP"
result_df = pd.concat((result_df, report))

## Test WGAN-GP with DRS
### Priority 1
synthesizer = WGANGP_DRSSynthesizer(metadata, batch_size=1000, epochs= 400, verbose= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/WGANGP-DRS.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "WGANGP-DRS"
result_df = pd.concat((result_df, report))

## Test CTGAN
### Priority 1
synthesizer = CTGANSynthesizer(metadata, batch_size=1000, epochs= 400, verbose= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/CTGAN.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "CTGAN"
result_df = pd.concat((result_df, report))

## Test TVAE
### Priority 1
synthesizer = TVAESynthesizer(metadata, batch_size=1000, epochs= 400, verbose= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/TVAE.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "TVAE"
result_df = pd.concat((result_df, report))

## Test FinDiff
### Priority 1
synthesizer = FINDIFFSynthesizer(metadata, batch_size=1000, epochs= 400, verbose= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("./synth/FinDiff.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "FinDiff"
result_df = pd.concat((result_df, report))


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
### Priority 1
synthesizer = DOPPELGANGERSynthesizer(metadata, context_columns= context_columns, max_sequence_len= 30, sample_len= 5, epochs= 400, batch_size=1000, verbose= True)
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows= 500)
synthetic_data.to_csv("./synth/DoppelGANger.csv", index=False)
print(synthetic_data.head())
diagnostic_report = run_diagnostic(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
quality_report = evaluate_quality(real_data= real_data, synthetic_data= synthetic_data, metadata= metadata)
report = pd.concat((diagnostic_report.get_properties(), quality_report.get_properties())).transpose()
report = pd.DataFrame(report.values[1:], columns=report.iloc[0])
report["Algorithm"] = "DoppelGANger"
result_df = pd.concat((result_df, report))

## Test Banksformer
### Priority 1
#synthesizer = BANKSFORMERSynthesizer(metadata, context_columns= context_columns, amount_column="Open", max_sequence_len= 260, sample_len= 5, epochs= 400, verbose= True)
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