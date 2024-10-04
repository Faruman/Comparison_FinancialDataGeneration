import os.path
import pickle
import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from modified_sitepackages.sdv.evaluation.single_table import evaluate_quality
from sdmetrics.single_table import NewRowSynthesis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.spatial.distance import cdist


models = ['DOPPELGANGER', 'FINDIFF', 'TVAE', 'WGANGPwDRS', 'CTGAN']
keep_col = ['Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering', 'transaction_clusters']

real_data = pd.read_csv("./working/transformed_df_graph.csv")
real_data = real_data.sample(100000)

if not os.path.exists("./results"):
    os.makedirs("./results")

if os.path.exists("./results/evaluation.xlsx"):
    results_df = pd.read_excel("./results/evaluation.xlsx")
else:
    results_df = pd.DataFrame()

for model in models:
    print("Currently Processing Model: {}".format(model))
    with open("./model/{}.pkl".format(model), 'rb') as file:
        synthesizer = pickle.load(file)

    synthetic_data = synthesizer.sample(num_rows= 100000)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data[keep_col])

    # Evaluate Fidelity
    #quality_report = evaluate_quality(real_data=real_data[keep_col], synthetic_data=synthetic_data[keep_col], metadata=metadata)
    #fidelity_dict = {**quality_report.get_properties().set_index("Property")["Score"].to_dict()}

    # Evaluate Synthesis
    #synthesis_dict = NewRowSynthesis.compute_breakdown(
    #    real_data= real_data[keep_col],
    #    synthetic_data= synthetic_data[keep_col],
    #    metadata= metadata,
    #    numerical_match_tolerance=0.01
    #)

    # Evaluate Privacy
    real_data_oh = pd.DataFrame()
    synthetic_data_oh = pd.DataFrame()
    cat_cols = [column_name for column_name, sdtype in metadata.columns.items() if sdtype["sdtype"] == "categorical"]
    cont_cols = [column_name for column_name, sdtype in metadata.columns.items() if sdtype["sdtype"] != "categorical"]
    for column_name, sdtype in metadata.columns.items():
        if sdtype["sdtype"] == 'categorical':
            ohe = OneHotEncoder()
            temp = pd.DataFrame(ohe.fit_transform(real_data[column_name].values.reshape(-1, 1)).todense(), columns= ["{}_{}".format(column_name, i) for i in range(real_data[column_name].nunique())], index= real_data.index)
            real_data_oh = pd.concat((real_data_oh, temp), axis= 1)
            temp = pd.DataFrame(ohe.transform(synthetic_data[column_name].values.reshape(-1, 1)).todense(), columns=["{}_{}".format(column_name, i) for i in range(real_data[column_name].nunique())], index=synthetic_data.index)
            synthetic_data_oh = pd.concat((synthetic_data_oh, temp), axis= 1)
        else:
            scaler = StandardScaler()
            real_data_oh = pd.concat((real_data_oh, pd.DataFrame(scaler.fit_transform(real_data[[column_name]]), columns= [column_name], index= real_data.index)), axis= 1)
            synthetic_data_oh = pd.concat((synthetic_data_oh, pd.DataFrame(scaler.transform(synthetic_data[[column_name]]), columns= [column_name], index= synthetic_data.index)), axis= 1)
    cat_cols = list(set(real_data_oh.columns) - set(cont_cols))
    scaler = StandardScaler()
    sample_size = 100
    i = 0
    min_distances = np.array([])
    second_min_distances = np.array([])
    while i < synthetic_data_oh.shape[0]:
        # Calculate the Euclidean distances
        synthetic_data_chunk = synthetic_data_oh.iloc[i:i + sample_size]
        distances_euclidean = cdist(synthetic_data_chunk[cont_cols].values, real_data_oh[cont_cols].values, metric='euclidean')
        distances_hamming = cdist(synthetic_data_chunk[cat_cols].values, real_data_oh[cat_cols].values, metric='hamming')
        distances = (len(cont_cols) / len(metadata.columns)) * distances_euclidean + (len(cat_cols) / len(metadata.columns)) * distances_hamming
        min_values = np.min(distances, axis=1)
        # To find the second minimum value for each row, we can mask the minimum values and find the minimum of the remaining elements
        masked_distances = np.where(distances == min_values[:, None], np.inf, distances)
        second_min_values = np.min(masked_distances, axis=1)
        # Find the minimum distance for each synthetic record
        min_distances = np.concatenate((min_distances, min_values))
        second_min_distances = np.concatenate((second_min_distances, second_min_values))
        i += sample_size

    ratio = min_distances / second_min_distances

    mean_dcr = np.mean(min_distances)
    median_dcr = np.median(min_distances)
    mean_dcr_5th = np.mean(np.percentile(min_distances, 5))
    median_dcr_5th = np.median(np.percentile(min_distances, 5))
    mean_nnrn = np.mean(ratio)
    median_nnrn = np.median(ratio)
    mean_nnrn_5th = np.mean(np.percentile(ratio, 5))
    median_nnrn_5th = np.median(np.percentile(ratio, 5))
    privacy_dict = {"Mean DCR": mean_dcr, "Median DCR": median_dcr, "Mean 5th DCR": mean_dcr_5th, "Median 5th DCR": median_dcr_5th,
                    "Mean NNRN": mean_nnrn, "Median NNRN": median_nnrn, "Mean NNRN 5th DCR": mean_nnrn_5th, "Median NNRN 5th DCR": median_nnrn_5th}

    # Store the results per model
    #results_df = pd.concat((results_df, pd.DataFrame({**fidelity_dict, **synthesis_dict, **privacy_dict}, index=[model])))
    results_df = pd.concat((results_df, pd.DataFrame(privacy_dict, index=[model])))

    results_df.drop_duplicates()
    results_df.to_excel("./results/evaluation.xlsx", index=True)

results_df.to_excel("./results/evaluation.xlsx", index=True)