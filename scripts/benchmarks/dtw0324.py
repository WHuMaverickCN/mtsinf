from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import ColumnEnsembleClassifier

# 假设您的文件路径是 'path/to/your_dataset.ts'
# train_file_path = '//home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id32_sample/RealMFQC_StageIII_id32_sample_TRAIN.ts'
# test_file_path = '/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id32_sample/RealMFQC_StageIII_id32_sample_TEST.ts'
train_file_path = '//home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id31/RealMFQC_StageIII_id31_TRAIN.ts'
test_file_path = '/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id31/RealMFQC_StageIII_id31_TEST.ts'
# 读取数据
X_train, y_train = load_from_tsfile_to_dataframe(train_file_path)
X_test, y_test = load_from_tsfile_to_dataframe(test_file_path)

# DTW 分类器使用 KNN
dtw_classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='dtw')

# 创建一个 ColumnEnsembleClassifier，对每个时间序列维度应用独立的 KNN-DTW 分类器
ensemble_classifier = ColumnEnsembleClassifier(
    estimators=[
        (f"dtw_knn_{i}", dtw_classifier, [i]) for i in range(X_train.shape[1])
    ]
)

# 训练模型
ensemble_classifier.fit(X_train, y_train)

# 进行预测
y_pred = ensemble_classifier.predict(X_test)

# 计算准确率
from sklearn.metrics import accuracy_score, precision_score

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算精确率
precision = precision_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")