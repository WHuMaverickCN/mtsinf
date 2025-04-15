import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
'''
# 读取数据
data = pd.read_csv('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id21/RealMFQC_StageIII_id21_TEST.csv')

# 分离特征和标签
X = data.drop('af', axis=1)
y = data['af'] 
'''

# 定义要排除的列
# excluded_columns = ['FOV_distance2', 'FOV_distance3', 'FOV_distance4', 'FOV_distance5', 'fovd']
excluded_columns = ['FOV_distance2', 'FOV_distance3', 'FOV_distance4', 'FOV_distance5']

# 读取训练集和测试集，排除指定的列
# train_data = pd.read_csv('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id21/RealMFQC_StageIII_id21_TRAIN.csv', usecols=lambda x: x not in excluded_columns)
# test_data = pd.read_csv('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id21/RealMFQC_StageIII_id21_TEST.csv', usecols=lambda x: x not in excluded_columns)

# train_data = pd.read_csv('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id31/RealMFQC_StageIII_id31_TRAIN.csv', usecols=lambda x: x not in excluded_columns)
# test_data = pd.read_csv('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id31/RealMFQC_StageIII_id31_TEST.csv', usecols=lambda x: x not in excluded_columns)

train_data = pd.read_csv('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id45pt/RealMFQC_StageIII_id45pt_TRAIN.csv', usecols=lambda x: x not in excluded_columns)
test_data = pd.read_csv('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id45pt/RealMFQC_StageIII_id45pt_TEST.csv', usecols=lambda x: x not in excluded_columns)


# 分离训练集的特征和标签
X_train = train_data.drop('af', axis=1)
y_train = train_data['af']

# 分离测试集的特征和标签
X_test = test_data.drop('af', axis=1)
y_test = test_data['af']

# 存储多次运行的准确率
svm_accuracies = []
xgb_accuracies = []

# 运行多次以获取统计结果
n_runs = 5
for i in range(n_runs):
    # 训练 SVM 模型
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_accuracies.append(accuracy_score(y_test, svm_predictions))

    # 训练 XGBoost 模型
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_accuracies.append(accuracy_score(y_test, xgb_predictions))

# 输出统计结果
print("\nSVM Statistics:")
print(f"Mean Accuracy: {np.mean(svm_accuracies):.4f}")
print(f"Std Accuracy: {np.std(svm_accuracies):.4f}")
print("Classification Report for last run:")
print(classification_report(y_test, svm_predictions))

print("\nXGBoost Statistics:")
print(f"Mean Accuracy: {np.mean(xgb_accuracies):.4f}")
print(f"Std Accuracy: {np.std(xgb_accuracies):.4f}")
print("Classification Report for last run:")
print(classification_report(y_test, xgb_predictions))
