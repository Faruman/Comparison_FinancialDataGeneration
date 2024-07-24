import pickle
import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from modified_sitepackages.sdv.evaluation.single_table import evaluate_quality
from sdmetrics.single_table import NewRowSynthesis
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


models = ['DOPPELGANGER', 'FINDIFF', 'TVAE', 'WGAN', 'CTGAN']
keep_col = ['Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering', 'transaction_clusters']

results_df = pd.DataFrame()

for model in models:
    print("Currently Processing Model: {}".format(model))
    with open("./model/{}.pkl".format(model), 'rb') as file:
        synthesizer = pickle.load(file)

    synthetic_data = synthesizer.sample(num_rows=100000)

    real_data = pd.read_csv("./working/transformed_df_graph.csv")

    # Evaluate Fidelity
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data[keep_col])
    quality_report = evaluate_quality(real_data=real_data[keep_col], synthetic_data=synthetic_data[keep_col], metadata=metadata)
    fidelity_dict = {**quality_report.get_properties().set_index("Property")["Score"].to_dict()}



    # Evaluate Synthesis
    synthesis_dict = NewRowSynthesis.compute_breakdown(
        real_data[keep_col],
        synthetic_data[keep_col],
        metadata,
        numerical_match_tolerance=0.01
    )

    # Evaluate Privacy
    scaler = StandardScaler()
    real_data_scaled = pd.DataFrame(scaler.fit_transform(real_data[keep_col]), columns=keep_col)
    synthetic_data_scaled = pd.DataFrame(scaler.transform(synthetic_data[keep_col]), columns=keep_col)
    sample_size = 100
    i = 0
    min_distances = np.array([])
    while i < synthetic_data_scaled.shape[0]:
        # Calculate the Euclidean distances
        synthetic_data_chunk = synthetic_data_scaled.iloc[i:i + sample_size]
        distances = cdist(synthetic_data_chunk.values, real_data_scaled.values, metric='euclidean')
        # Find the minimum distance for each synthetic record
        min_distances = np.concatenate((min_distances, distances.min(axis=1)))
        i += sample_size

    # DCR
    mean_dcr = np.mean(min_distances)
    median_dcr = np.median(min_distances)
    privacy_dict = {"Privacy Mean": mean_dcr, "Privacy Median": median_dcr}

    results_df = pd.concat((results_df, pd.DataFrame({**fidelity_dict, **synthesis_dict, **privacy_dict}, index=[model])))

