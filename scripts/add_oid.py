import os
import pandas as pd
import json

# 指定目录路径
directory_path = '/home/gyx/projects/shapeformer/Dataset/raw/serinf2jag/modisf/20250304_2125_l35_809'

# 定义文件路径
geojson_path = os.path.join(directory_path, 'train_all_vec.geojson')
csv_path = os.path.join(directory_path, 'train_converted_data_5D.csv')

# 读取GeoJSON文件
with open(geojson_path, 'r') as geojson_file:
    geojson_data = json.load(geojson_file)
    print("GeoJSON data loaded successfully.")

id_to_oid = {}
for feature in geojson_data['features']:
    # 提取id和oid2
    feature_id = feature['properties'].get('id', 'No ID provided')
    feature_oid = feature['properties'].get('oid', 'No OID provided')
    if feature_id and feature_oid:
        id_to_oid[feature_id] = feature_oid
    # 打印id和oid
    # print(f"Feature ID: {feature_id}, OID: {feature_oid}")

# 读取CSV文件
csv_data = pd.read_csv(csv_path)

if 'id' in csv_data.columns:
    # 创建一个新列，用于存放OID
    csv_data['oid'] = csv_data['id'].map(id_to_oid)
else:
    print("Error: No 'id' column found in the CSV.")

csv_data.to_csv(csv_path.replace('.csv','_full.csv'), index=False)
print("CSV data loaded successfully.")

# 打印输出数据示例（可选）
# print(json.dumps(geojson_data, indent=4))  # 打印GeoJSON数据的部分内容
# print(csv_data.head())  # 打印CSV数据的前几行
