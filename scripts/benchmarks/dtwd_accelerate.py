from dtaidistance import dtw_ndim
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sktime.datasets import load_from_tsfile_to_dataframe
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
# from dtwd import convert_dataframe_to_series_list


X_train,y_train = load_from_tsfile_to_dataframe('/home/gyx/projects/shapeformer/Dataset/CQC/Multivariate_ts/RealMFQC_StageIII_id31/RealMFQC_StageIII_id31_TRAIN.ts')
series_list = convert_dataframe_to_series_list(X_train)
print(type(series_list))
print(series_list[0].__len__())