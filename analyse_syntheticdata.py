# Fidelity
# libraries
import scipy
import boto3
import numpy
import pandas as pd
import dill as pickle
import io
from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality, evaluate_similarity
from sdmetrics.reports.single_table.quality_report import QualityReport
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport

# sdv
bucket_name = 'adi-aicoe-bucket'
s3_file_key = 'WGAN.pkl'
s3_resource = boto3.resource('s3')
obj = s3_resource.Object(bucket_name, s3_file_key).get()['Body'].read()
synthesizer = pickle.loads(obj)

WGAN_synthetic_data = synthesizer.sample(num_rows=100000)

df = spark.read.csv('s3://adi-aicoe-bucket/transformed_full_pca_data_graph_min3.csv', header=True,
                    inferSchema=True).toPandas()

keep_col = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
synthetic_data = WGAN_synthetic_data[keep_col]
real_data = df[keep_col]

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
# in quality_report
# Column Shapes -> Column Fidelity
# Column Pair Trends -> Row Fidelty

quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
fidelity_dict = {**quality_report.get_properties().set_index("Property")["Score"].to_dict()}
fidelity_df = pd.DataFrame.from_dict(fidelity_dict, orient='index', columns=['Score'])

bucket = 'adi-aicoe-bucket'
filename = "WGAN_fidelity.csv"
csv_buffer = io.StringIO()
fidelity_df.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, filename).put(Body=csv_buffer.getvalue())

# SDMetrics for plots
# Column Shapes -> Column Fidelity -> KSComplement
# Column Pair Trends -> Row Fidelty -> Correlation

metadata = {
    'columns': {
        'PC1': {'sdtype': 'numerical'},
        'PC2': {'sdtype': 'numerical'},
        'PC3': {'sdtype': 'numerical'},
        'PC4': {'sdtype': 'numerical'},
        'PC5': {'sdtype': 'numerical'},
        'PC6': {'sdtype': 'numerical'},
        'PC7': {'sdtype': 'numerical'},
        'PC8': {'sdtype': 'numerical'},
        'PC9': {'sdtype': 'numerical'},
        'PC10': {'sdtype': 'numerical'},
        'primary_key': 'source_id'}
}

my_report = QualityReport()
my_report.generate(real_data, synthetic_data, metadata)

my_report.get_visualization(property_name='Column Pair Trends')

my_report.get_visualization(property_name='Column Shapes')


# Synthesis
# import libraries
import scipy
import boto3
import pandas as pd
import numpy as np
from tqdm import tqdm
import dill as pickle
import io
from modified_sitepackages.sdv.evaluation.single_table import run_diagnostic, evaluate_quality, evaluate_similarity
from sdv.metadata import SingleTableMetadata
from sdmetrics.single_table import NewRowSynthesis

# load model
bucket_name = 'adi-aicoe-bucket'
s3_file_key = 'TVAE.pkl'
s3_resource = boto3.resource('s3')
obj = s3_resource.Object(bucket_name, s3_file_key).get()['Body'].read()
synthesizer = pickle.loads(obj)

# load data
TVAE_synthetic_data = synthesizer.sample(num_rows=100000)
df = spark.read.csv('s3://adi-aicoe-bucket/transformed_full_pca_data_graph_min3.csv', header=True,
                    inferSchema=True).toPandas()

keep_col = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
synthetic_data = TVAE_synthetic_data[keep_col]
real_data = df[keep_col]

# SDMetrics
metadata = {
    'columns': {
        'PC1': {'sdtype': 'numerical'},
        'PC2': {'sdtype': 'numerical'},
        'PC3': {'sdtype': 'numerical'},
        'PC4': {'sdtype': 'numerical'},
        'PC5': {'sdtype': 'numerical'},
        'PC6': {'sdtype': 'numerical'},
        'PC7': {'sdtype': 'numerical'},
        'PC8': {'sdtype': 'numerical'},
        'PC9': {'sdtype': 'numerical'},
        'PC10': {'sdtype': 'numerical'},
    },
    'primary_key': 'source_id'
}

synthesis_dict = NewRowSynthesis.compute_breakdown(
    real_data,
    synthetic_data,
    metadata,
    numerical_match_tolerance=0.01
)
synthesis_df = pd.DataFrame.from_dict(synthesis_dict, orient='index', columns=['Score'])

# save results
bucket = 'adi-aicoe-bucket'
filename = "TVAE_synthesis.csv"
csv_buffer = io.StringIO()
synthesis_df.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, filename).put(Body=csv_buffer.getvalue())

# # codes based on Sattarov et al.,
# # Use vectorized operations in pandas for optimisation
# def calculate_synthesis_score(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, tolerance=0.01):
#     def numeric_match(real_col, synthetic_col, tolerance):
#         return np.abs(real_col - synthetic_col) <= tolerance * real_col

#     Snm = 0

#     # Iterate over each row in synthetic_data
#     for syn_index, syn_row in tqdm(synthetic_data.iterrows(), total=len(synthetic_data), desc="Processing rows"):
#         match_found = False
#         mask = pd.Series([True] * len(real_data))

#         for col in synthetic_data.columns:
#             if pd.api.types.is_numeric_dtype(real_data[col]):
#                 mask &= numeric_match(real_data[col], syn_row[col], tolerance)
#             else:
#                 mask &= (real_data[col] == syn_row[col])

#         if mask.any():
#             match_found = True

#         if match_found:
#             Snm += 1

#     N = len(synthetic_data)
#     S = 1 - Snm / N

#     return S

# score = calculate_synthesis_score(real_data, synthetic_data)
# print("Synthesis Score:", score)


# Privacy metrics
# libraries
import scipy
import boto3
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import dill as pickle
# from sdmetrics.single_table import privacy
# from sdmetrics.single_table.privacy import NumericalRadiusNearestNeighbor
from sdmetrics.single_table.privacy.radius_nearest_neighbor import NumericalRadiusNearestNeighbor
from scipy.spatial.distance import cdist

bucket_name = 'adi-aicoe-bucket'
s3_resource = boto3.resource('s3')
df = spark.read.csv('s3://adi-aicoe-bucket/transformed_full_pca_data_graph_min3.csv', header=True, inferSchema=True).toPandas()

# Distance to Closest Record (DCR) is the Euclidean distance between the record of the synthetic data and the  nearest record in the original table. DCR equal to zero means that the synthetic record will leak the real information, while higher DCR values mean less risk of privacy leakage.
models = ['DOPPELGANGER.pkl', 'FINDIFF.pkl', 'TVAE.pkl', 'WGAN.pkl', 'CTGAN.pkl']
keep_col = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
real_data = df[keep_col]
score_list_median = []
score_list_mean = []
for s3_file_key in models:
    print(s3_file_key.split('.')[0])
    obj = s3_resource.Object(bucket_name, s3_file_key).get()['Body'].read()
    synthesizer = pickle.loads(obj)
    synthetic_data = synthesizer.sample(num_rows=100000)
    synthetic_data = synthetic_data[keep_col]

    # Scale the data
    scaler = StandardScaler()
    real_data_scaled = pd.DataFrame(scaler.fit_transform(real_data), columns=keep_col)
    synthetic_data_scaled = pd.DataFrame(scaler.transform(synthetic_data), columns=keep_col)
    sample_size = 100
    i = 0
    min_distances = np.array([])
    while i < synthetic_data_scaled.shape[0]:
        # Calculate the Euclidean distances
        synthetic_data_chunk = synthetic_data_scaled.iloc[i:i+sample_size]
        distances = cdist(synthetic_data_chunk.values, real_data_scaled.values, metric='euclidean')
        # Find the minimum distance for each synthetic record
        min_distances = np.concatenate((min_distances,distances.min(axis=1)))
        i += sample_size

    # DCR
    mean_dcr = np.mean(min_distances)
    median_dcr = np.median(min_distances)

    score_list_mean.append(mean_dcr)
    score_list_median.append(median_dcr)
    print(f'Mean DCR: {mean_dcr}')
    print(f'Median DCR: {median_dcr}')  # Sattarov used the median.

score_list_mean = score_list_mean
score_list_median = score_list_median
models = models
report = pd.DataFrame({'algorithm': models, 'mean': score_list_mean, 'median': score_list_median})
report