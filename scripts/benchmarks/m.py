from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_UCR_UEA_dataset
from sktime.transformations.panel import channel_selection
from sktime.transformations.panel.rocket import Rocket
from sktime.datasets import load_from_tsfile_to_dataframe

# cs = channel_selection.ElbowClassSum()  # ECS
cs = channel_selection.ElbowClassPairwise()  # ECP
rocket_pipeline = make_pipeline(cs, Rocket(), RidgeClassifierCV())

data = "BasicMotions"
X_train, y_train = load_UCR_UEA_dataset(data, split="train", return_X_y=True)
X_test, y_test = load_UCR_UEA_dataset(data, split="test", return_X_y=True)
print(type(X_train),\
        type(y_train),\
            X_train.shape,\
                y_train.shape)
train_df, train_df_y = load_from_tsfile_to_dataframe('/home/gyx/projects/shapeformer/Dataset/raw/serinf2jag/allinone/20250220_l35-0809/train_converted_data_25.ts')

rocket_pipeline.fit(X_train, y_train)
rocket_pipeline.score(X_test, y_test)

rocket_pipeline.steps[0][1].channels_selected_
rocket_pipeline.steps[0][1].distance_frame_

print(type(train_df),\
        type(train_df_y),\
            train_df.shape,\
                train_df_y.shape)
print("done")