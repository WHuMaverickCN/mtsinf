from dtaidistance import dtw_ndim
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sktime.datasets import load_from_tsfile_to_dataframe

# 创建样本时间序列数据（假设每个系列是二维时间序列）
series1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1], [0, 0]], dtype=np.double)
series2 = np.array([[0, 0], [2, 1], [0, 1], [0, 0.5], [0, 0]], dtype=np.double)
series3 = np.array([[1, 0], [1, 1], [3, 1], [1, 1], [1, 0]], dtype=np.double)
series4 = np.array([[0, 0], [3, 1], [1, 1], [0, 1], [0, 0]], dtype=np.double)

# 假设这些系列有对应的标签，0 和 1 作为二分类标签
X = [series1, series2, series3, series4]  # 时间序列数据列表
y = [0, 1, 0, 1]  # 标签，假设2个类别

# 假设有一个新的测试样本series5和对应标签
series5 = np.array([[1, 0], [2, 1], [1, 1], [1, 0], [0, 0]], dtype=np.double)
series6 = np.array([[0, 0], [1, 1], [2, 1], [1, 0.5], [0, 0]], dtype=np.double)
series7 = np.array([[1, 0], [2, 2], [3, 1], [1, 1], [0, 0]], dtype=np.double)
series8 = np.array([[0, 0], [0.5, 1], [1, 1], [0, 0.5], [0, 0]], dtype=np.double)

# Combine all test samples
X_test = [series5, series6, series7, series8]
y_test = [0, 1, 0, 1]  # Assumed ground truth labels

# X.shape

X_,y_ = load_from_tsfile_to_dataframe('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC/RealMFQC_TRAIN.ts')

def convert_dataframe_to_series_list(df):
    num_instances = len(df)
    num_dims = len(df.columns)
    series_list = []
    
    for i in range(num_instances):
        print(i)
        # 获取当前实例的所有维度数据
        series = []
        for col in df.columns:
            values = df[col].iloc[i].values
            series.append(values)
        
        # 转置数组使其成为 [timesteps, dimensions] 的形状
        series = np.array(series).T
        series_list.append(series)
    print("done")
    return series_list

# 转换数据
X_series = convert_dataframe_to_series_list(X_)

# 计算每个时间序列之间的DTW距离（构建一个DTW距离矩阵）
def compute_distance_matrix(X):
    n = len(X)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        print(f"{i}/{n}")
        for j in range(i, n):
            distance = dtw_ndim.distance(X[i], X[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    return distance_matrix

# 计算训练集的DTW距离矩阵
distance_matrix = compute_distance_matrix(X_series)

# 将距离矩阵作为特征，训练KNN分类器
X_train = distance_matrix  # 使用DTW距离矩阵作为输入特征
y_train = np.array(y)  # 标签

# 打印数据结构
print("X_train shape:", X_train.shape)
print("X_train data:\n", X_train,type(y_train))
print("\ny_train shape:", y_train.shape)
print("y_train data:", y_train,type(y_train))

# 使用KNN进行分类
knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
knn.fit(X_train, y_train)


# 计算测试样本与训练集样本的DTW距离
distances_to_test = [dtw_ndim.distance(series5, train_sample) for train_sample in X]
print("DTW distances to test sample:", distances_to_test)

# 使用KNN进行预测
# Generate more test samples



# Calculate DTW distances for all test samples
test_distances = []
for test_sample in X_test:
    distances = [dtw_ndim.distance(test_sample, train_sample) for train_sample in X]
    test_distances.append(distances)

# Make predictions for all test samples
y_pred = knn.predict([distances_to_test])
print("Predicted Class:", y_pred)

# 计算准确率
accuracy = (y_pred == y_test).mean()
print("Prediction Accuracy:", accuracy)
