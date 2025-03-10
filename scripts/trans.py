import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import shutil

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()
    
    def load_data(self):
        return pd.read_csv(self.file_path)
    
    def get_y_distribution(self):
        return self.df['y'].describe()
    
    def get_y_intervals(self, bins=10):
        # 按区间统计y值
        min_y = self.df['y'].min()
        max_y = self.df['y'].max()
        intervals = pd.cut(self.df['y'], bins=bins, precision=2)
        interval_counts = intervals.value_counts().sort_index()
        return interval_counts
    
    def replace_y_with_af(self,target_distance = 1.0):
        """Replace y column with af column based on conditions"""
        self.df['af'] = np.where(self.df['y'] < target_distance, 1, 0)
        print(self.df.shape)
        print(self.df[self.df['y']<target_distance].shape)
        # input("Press Enter to continue...")
        self.df = self.df.drop('y', axis=1)
        # Save to a new file with '_processed' suffix
        new_file_path = self.file_path.rsplit('.', 1)[0] + '_class_mode.csv'
        self.df.to_csv(new_file_path, index=False)
        return self.df
    
    def plot_y_distribution(self):
        plt.figure(figsize=(8,6))
        self.df['y'].hist(bins=50)
        plt.title('Distribution of y')
        plt.xlabel('y')
        plt.ylabel('Frequency')
        plt.show()
    
    def __csv_to_arff(self, relation_name="Dataset", seq_length=50, step_length=8):
        """Convert the loaded CSV file to ARFF format with time series features."""
        step_len = seq_length // step_length
        # Select relevant features
        output_path = self.file_path.replace(".csv",f"_{step_len}.ts")
        # titles = [f'id','utc',f'posi_type',f'numsv',f'dv',f'fovd',f'speed',f'altitude',f'roll',f'y'] 
        features = ['posi_type','numsv', 'dv', 'fovd', 'speed','altitude','roll']
        
        # Create time series features
        window_features = []
        # for feature in features:
        #     for i in range(seq_length):
        #         window_features.append(f"@attribute {feature}_{i} numeric")
        
        # Add class attribute
        # window_features.append("@attribute af {0,1}")
        
        # Write ARFF file
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"@relation {relation_name}\n\n")
            
            # Write metadata
            f.write("@problemName ArticularyWordRecognition\n")
            f.write("@timeStamps false\n")
            f.write("@missing false\n")
            f.write("@univariate false\n")
            f.write(f"@dimensions {len(features)}\n")
            f.write("@equalLength true\n")
            f.write(f"@seriesLength {seq_length}\n")
            f.write("@classLabel true 0.0 1.0\n\n")
            
            # Write attributes
            # for attr in window_features:
            #     f.write(f"{attr}\n")
            
            # Write data section
            f.write("\n@data\n")
            
            # Create sliding windows
            features_data = self.df[features].values
            num_samples = len(self.df) - seq_length
            
            for i in range(0,num_samples,step_len):
                window = features_data[i:i+seq_length]
                # Create feature sequences for each feature type
                feature_sequences = []
                for j in range(len(features)):
                    sequence = ",".join(str(val) for val in window[:, j])
                    feature_sequences.append(sequence)
                
                # Add target label
                target = self.df['af'].iloc[i+seq_length-1]
                # Join sequences with ":" and add label
                row_str = ":".join(feature_sequences) + ":" + str(float(target))
                f.write(f"{row_str}\n")

    def __csv_to_arff_stage_2(self, relation_name="Dataset", seq_length=50, step_length=4):
        """Convert the loaded CSV file to ARFF format with time series features."""
        step_len = seq_length // step_length
        # Select relevant features
        output_path = self.file_path.replace(".csv",f"_{step_len}_stage2.ts")
        print(output_path)
        # titles = [f'id','utc',f'posi_type',f'numsv',f'dv',f'fovd',f'speed',f'altitude',f'roll',f'y'] 
        # features_stage1 = ['posi_type','numsv', 'dv', 'fovd', 'speed','altitude','roll']
        features = ['posi_type', 'dv','altitude','roll'] # stage2 feature selection
        # Create time series features
        window_features = []
        # for feature in features:
        #     for i in range(seq_length):
        #         window_features.append(f"@attribute {feature}_{i} numeric")
        
        # Add class attribute
        # window_features.append("@attribute af {0,1}")
        
        # Write ARFF file
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"@relation {relation_name}\n\n")
            
            # Write metadata
            f.write("@problemName ArticularyWordRecognition\n")
            f.write("@timeStamps false\n")
            f.write("@missing false\n")
            f.write("@univariate false\n")
            f.write(f"@dimensions {len(features)}\n")
            f.write("@equalLength true\n")
            f.write(f"@seriesLength {seq_length}\n")
            f.write("@classLabel true 0.0 1.0\n\n")
            
            # Write attributes
            # for attr in window_features:
            #     f.write(f"{attr}\n")
            
            # Write data section
            f.write("\n@data\n")
            
            # Create sliding windows
            features_data = self.df[features].values
            num_samples = len(self.df) - seq_length
            
            for i in range(0,num_samples,step_len):
                window = features_data[i:i+seq_length]
                # Create feature sequences for each feature type
                feature_sequences = []
                for j in range(len(features)):
                    sequence = ",".join(str(val) for val in window[:, j])
                    feature_sequences.append(sequence)
                
                # Add target label
                target = self.df['af'].iloc[i+seq_length-1]
                # Join sequences with ":" and add label
                row_str = ":".join(feature_sequences) + ":" + str(float(target))
                f.write(f"{row_str}\n")

    def csv_to_arff_switch_window(self, relation_name="Dataset", seq_length=50, step_length=4,start_point=3):
        """Convert the loaded CSV file to ARFF format with time series features."""
        step_len = int(seq_length // step_length)
        # input(step_len)
        # Select relevant features
        output_path = self.file_path.replace(".csv",f"_{step_len}_stage3.ts")
        print(output_path)
        # titles = [f'id','utc',f'posi_type',f'numsv',f'dv',f'fovd',f'speed',f'altitude',f'roll',f'y'] 
        # features_stage1 = ['posi_type','numsv', 'dv', 'fovd', 'speed','altitude','roll']
        
        
        # features = ['posi_type', 'dv','altitude','roll',f'fovd',\
        #             f'FOV_distance2',f'FOV_distance3',f'FOV_distance4',f'FOV_distance5'] # stage2 feature selection
        
        features = ['posi_type', 'dv','altitude','roll']
        
        # Create time series features
        window_features = []
        # for feature in features:
        #     for i in range(seq_length):
        #         window_features.append(f"@attribute {feature}_{i} numeric")
        
        # Add class attribute
        # window_features.append("@attribute af {0,1}")
        
        # Write ARFF file
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"@relation {relation_name}\n\n")
            
            # Write metadata
            f.write("@problemName ArticularyWordRecognition\n")
            f.write("@timeStamps false\n")
            f.write("@missing false\n")
            f.write("@univariate false\n")
            f.write(f"@dimensions {len(features)}\n")
            f.write("@equalLength true\n")
            f.write(f"@seriesLength {seq_length}\n")
            f.write("@classLabel true 0.0 1.0\n\n")
            
            # Write attributes
            # for attr in window_features:
            #     f.write(f"{attr}\n")
            
            # Write data section
            f.write("\n@data\n")
            
            # Create sliding windows
            features_data = self.df[features].values
            num_samples = len(self.df) - seq_length
            
            for i in range(start_point+seq_length,num_samples,step_len):
                # window = features_data[i:i+seq_length]
                window = features_data[i-seq_length:i]
                # Create feature sequences for each feature type
                feature_sequences = []
                for j in range(len(features)):
                    sequence = ",".join(str(val) for val in window[:, j])
                    feature_sequences.append(sequence)
                
                # Add target label
                # target = self.df['af'].iloc[i+seq_length-1]
                # print(window[-1])
                # print(self.df[['posi_type', 'dv','altitude','roll']].iloc[i-1])
                # print(window[-1][-1] - self.df['roll'].iloc[i-1],\
                #       window[-1][-2] - self.df['altitude'].iloc[i-1])
                # input()
                target = self.df['af'].iloc[i-1]
                # Join sequences with ":" and add label
                row_str = ":".join(feature_sequences) + ":" + str(float(target))
                f.write(f"{row_str}\n")
            # output_path
            parent_dir = os.path.dirname(output_path)
            file = os.path.basename(output_path)
            # print(file)
            # input()
            temp_dir = os.path.join(parent_dir, 'temp')
            target_file = os.path.join(temp_dir, file)

            # 检查目录存在
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # 检查文件存在
            if os.path.exists(target_file):
                os.remove(target_file)
            shutil.move(output_path, temp_dir)
            print(f"Moved {output_path} to {temp_dir}")
            return target_file
    def csv_to_csv_switch(self, seq_length=50, step_length=4,start_point=0):
        """Downsample the CSV data by selecting rows at regular intervals."""
        num_samples = len(self.df) - seq_length
        step_len = int(seq_length // step_length)
        output_path = self.file_path.replace(".csv", f"_{step_len}_stage3.csv")
        print(output_path)
        
        # features = ['id','posi_type','numsv', 'dv', 'fovd', 'speed','altitude','roll']
        features = ['id','posi_type','numsv', 'dv', 'fovd', 'speed','altitude','roll',\
                    f'FOV_distance2',f'FOV_distance3',f'FOV_distance4',f'FOV_distance5']
        
        # Select rows at regular intervals using step_len
        downsampled_df = self.df.iloc[start_point+seq_length-1:num_samples:step_len]
        
        # Save the downsampled data to CSV
        columns_to_save = features + ['af']
        downsampled_df[columns_to_save].to_csv(output_path, index=False)

        parent_dir = os.path.dirname(output_path)
        file = os.path.basename(output_path)

        temp_dir = os.path.join(parent_dir, 'temp')
        target_file = os.path.join(temp_dir, file)

        # 检查目录存在
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 检查文件存在
        if os.path.exists(target_file):
            os.remove(target_file)
        shutil.move(output_path, temp_dir)
        print(f"Moved {output_path} to {temp_dir}")
        return target_file

def __trans(file_path,step_length,target_d_precision,start_point,sample_sequence_length= 30):
    # sample_sequence_length = 30


    processor = DataProcessor(file_path)
    
    processor.replace_y_with_af(target_d_precision)

    # extract_target_rows
    csv_target_file = processor.csv_to_csv_switch(seq_length=sample_sequence_length,\
                                step_length = step_length,\
                                start_point = start_point)

    # Convert to ARFF format
    return processor.csv_to_arff_switch_window(relation_name="CQCDataset_Stage2",\
                        seq_length=sample_sequence_length,\
                        step_length = step_length,
                        start_point = start_point),csv_target_file
    
with open('config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

root = config['paths']  # Assuming 'file_path' is defined in your yaml
step_length = config['step_length']
target_d_precision = config['target_d_precision']
test_case_path = config['test_case_name']
start_point = config['start_point']
sample_sequence_length = config['sample_sequence_length']

for i in root:
    file_path = root[i]
    ts_target_file,csv_target_file = __trans(file_path,step_length,target_d_precision,start_point,sample_sequence_length)
    print(i,ts_target_file,csv_target_file)
    # 获取 test_case_path 最底层文件夹的名称
    base_folder_name = os.path.basename(os.path.normpath(test_case_path))

    # input()
    # 根据条件移动并重命名文件
    if i == 'train':
        # 构造新文件名
        new_file_name = f"{base_folder_name}_TRAIN.ts"
        new_file_path = os.path.join(test_case_path, new_file_name)
        # 移动并重命名文件

        new_csv_file_name = f"{base_folder_name}_TRAIN.csv"
        new_csv_file_path = os.path.join(test_case_path, new_csv_file_name)

        shutil.move(ts_target_file, new_file_path)
        shutil.move(csv_target_file, new_csv_file_path)
        print(f"Moved and renamed '{i}' file to {new_file_path}")

    elif i == 'test':
        # 构造新文件名
        new_file_name = f"{base_folder_name}_TEST.ts"
        new_file_path = os.path.join(test_case_path, new_file_name)
        
        new_csv_file_name = f"{base_folder_name}_TEST.csv"
        new_csv_file_path = os.path.join(test_case_path, new_csv_file_name)

        # 移动并重命名文件
        shutil.move(ts_target_file, new_file_path)
        shutil.move(csv_target_file, new_csv_file_path)
        print(f"Moved and renamed '{i}' file to {new_file_path}")

# 定义文件路径
file_path = 'config.yaml'
# test_case_path = input('请输入目标路径: ').strip()
if not os.path.exists(test_case_path):
    os.makedirs(test_case_path)

# 复制并重命名文件
new_file_path = os.path.join(test_case_path, 'config.log')
shutil.copy2(file_path, new_file_path)

    # print(file_path)
# input()
# __trans(file_path,step_length,target_d_precision)

'''
processor = DataProcessor(file_path)
processor.replace_y_with_af(target_d_precision)

# Convert to ARFF format
processor.csv_to_arff_stage_3(relation_name="CQCDataset_Stage2",\
                      seq_length=200,\
                    step_length = step_length)
                    '''
