import pickle

import numpy as np
import pandas as pd


models = ['DOPPELGANGER', 'FINDIFF', 'TVAE', 'WGAN', 'CTGAN']
keep_col = ['Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering', 'transaction_clusters']

results_df = pd.DataFrame()

for model in models:
    print("Currently Processing Model: {}".format(model))
    with open("./model/{}.pkl".format(model), 'rb') as file:
        synthesizer = pickle.load(file)

    synthetic_data = synthesizer.sample(num_rows=100000)

    real_data = pd.read_csv("./working/transformed_df_graph.csv")

    # Restore Graph Structure
    ## Get from embeddings to grpah nodes
