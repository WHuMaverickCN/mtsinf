from sktime.datasets import load_from_tsfile_to_dataframe

# 读取 .ts 文件
train_df, y_train = load_from_tsfile_to_dataframe('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/mfqc_L350727/converted.ts')
test_df, y_test = load_from_tsfile_to_dataframe()
# df = pd.read_csv('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/mfqcSample/mfqcSample_TEST.ts', sep=',')  # 假设文件以制表符分隔
print(type(train_df),train_df.shape)
print(type(y_train),y_train)