# load data
if not os.path.exists('s3://adi-aicoe-bucket/transformed_full_pca_data_ready.csv'):
    real_data_raw = pd.read_feather('/Workspace/Shared/syn_gen_data/transformed_full_pca_data.feather')
    real_data = real_data_raw.copy()
    real_data = real_data.reset_index()
    real_data["index"] = pto_numeric(real_data["index"]).astype(int)
    real_data = real_data.rename(columns={"index": "timeIndicator"})
    # isSender = True
    real_data_1 = real_data.copy()
    real_data_1.rename(columns={"source_id": "Id", }, inplace=True)
    real_data_1.drop(["target_id"], axis=1, inplace=True)
    real_data_1["isSender"] = True
    real_data_1
    # isSender = False
    real_data_2 = real_data.copy()
    real_data_2.rename(columns={"target_id": "Id", }, inplace=True)
    real_data_2.drop(["source_id"], axis=1, inplace=True)
    real_data_2["isSender"] = False
    real_data_2
    # combine both isSender (True, False)
    real_data_combined = pd.concat([real_data_1, real_data_2], axis=0)
    real_data.to_csv('s3://adi-aicoe-bucket/transformed_full_pca_data_ready.csv', index=False)