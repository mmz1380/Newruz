import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from IPython.display import display


def undummify(df : pd.DataFrame, prefix_sep=" ") -> pd.DataFrame:
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def dataframe_to_dataset(dataframe:pd.DataFrame) -> tf.data.Dataset:
    dataframe = dataframe.copy()
    labels = dataframe.pop("Cover_Type")
    labels = keras.utils.to_categorical(labels)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))

    return ds

data = pd.read_csv("covtype.data",header=None)
data.columns = ["Elevation","Aspect","Slope","HDH","VDH","HDR","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","HDFP",
                "Wilderness_Area Rawah","Wilderness_Area Neota","Wilderness_Area Comanche_Peak",
                "Wilderness_Area Cache_la_Poudre","Soil_Type USFS2702","Soil_Type USFS2703","Soil_Type USFS2704",
                "Soil_Type USFS2705","Soil_Type USFS2706","Soil_Type USFS2717","Soil_Type USFS3501","Soil_Type USFS3502",
                "Soil_Type USFS4201","Soil_Type USFS4703","Soil_Type USFS4704","Soil_Type USFS4744","Soil_Type USFS4758",
                "Soil_Type USFS5101","Soil_Type USFS5151","Soil_Type USFS6101","Soil_Type USFS6102","Soil_Type USFS6731",
                "Soil_Type USFS7101","Soil_Type USFS7102","Soil_Type USFS7103","Soil_Type USFS7201","Soil_Type USFS7202",
                "Soil_Type USFS7700","Soil_Type USFS7701","Soil_Type USFS7702","Soil_Type USFS7709","Soil_Type USFS7710",
                "Soil_Type USFS7745","Soil_Type USFS7746","Soil_Type USFS7755","Soil_Type USFS7756","Soil_Type USFS7757",
                "Soil_Type USFS7790","Soil_Type USFS8703","Soil_Type USFS8707","Soil_Type USFS8708","Soil_Type USFS8771",
                "Soil_Type USFS8772","Soil_Type USFS8776","Cover_Type"]


data = undummify(data)

df = data.sample(frac=0.2, random_state=1337)
train_df = df.sample(frac=0.7, random_state=1337)
val_df = df.drop(train_df.index)
test_df = data.drop(df.index)

train_ds = dataframe_to_dataset(train_df)
val_ds = dataframe_to_dataset(val_df)
test_ds = dataframe_to_dataset(test_df)

from keras.utils import FeatureSpace

feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        # "sex": FeatureSpace.integer_categorical(num_oov_indices=0),
        # Categorical feature encoded as string
        "Wilderness_Area": FeatureSpace.string_categorical(num_oov_indices=0),
        "Soil_Type": FeatureSpace.string_categorical(num_oov_indices=0),
        # Numerical features to normalize
        # "age": FeatureSpace.float_discretized(num_bins=30),
        # Numerical features to normalize
        "Elevation": FeatureSpace.float_normalized(),
        "Aspect": FeatureSpace.float_normalized(),
        "Slope": FeatureSpace.float_normalized(),
        "HDH": FeatureSpace.float_normalized(),
        "VDH": FeatureSpace.float_normalized(),
        "HDR": FeatureSpace.float_normalized(),
        "Hillshade_9am": FeatureSpace.float_normalized(),
        "Hillshade_Noon": FeatureSpace.float_normalized(),
        "Hillshade_3pm": FeatureSpace.float_normalized(),
        "HDFP": FeatureSpace.float_normalized(),

    },
    # Specify feature cross with a custom crossing dim.
    # crosses=[
    #     FeatureSpace.cross(feature_names=("Soil_Type", "Wilderness_Area"), crossing_dim=32),
    # ],
    output_mode="concat",
)

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_test_ds = test_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_test_ds = preprocessed_test_ds.prefetch(tf.data.AUTOTUNE)


"this is my best model for this data at 85% accuracy for val_ds"

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(16, activation="relu")(encoded_features)
# x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(32, activation="relu")(x)
# x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(64, activation="relu")(x)
# x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(128, activation="relu")(x)
# x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(64, activation="relu")(x)
# x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(32, activation="relu")(x)
# x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(16, activation="relu")(x)
# x = keras.layers.Dropout(0.3)(x)
predictions = keras.layers.Dense(8, activation="softmax")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)



training_model.fit(
    preprocessed_train_ds, epochs=20, validation_data=preprocessed_val_ds, verbose=1
)