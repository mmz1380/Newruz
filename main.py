"""

I used sample code which I added it to this git's repository to make this also used my codes in my Cheat-sheet and Q1 repository to modify it to work on my Dataset

"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.core.display_functions import display
import matplotlib.pyplot as plt

EMBEDDING_SIZE = 50
"""
Also Leart about using low level coding Tensorflow without Keras in this section
"""
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.item_embedding = layers.Embedding(
            num_items,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_bias = layers.Embedding(num_items, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

"""This part was a integarted piece of code but I seperated it and make it a independent function"""

def df_encoding(df: pd.DataFrame) -> pd.DataFrame:
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    item_ids = df["itemId"].unique().tolist()
    item2item_encoded = {x: i for i, x in enumerate(item_ids)}
    item_encoded2item = {i: x for i, x in enumerate(item_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["item"] = df["itemId"].map(item2item_encoded)

    num_users = len(user2user_encoded)
    num_items = len(item_encoded2item)
    df["rating"] = df["rating"].values.astype(np.float32)
    # min and max ratings will be used to normalize the ratings later
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    return df, num_users, num_items, min_rating, max_rating

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

df_train, num_users, num_items, min_rating, max_rating = df_encoding(df_train)
data = df_train.sample(frac=0.15, random_state=1111)

# Assuming training on 80% of the data and validating on 20%.
train_df = data.sample(frac=0.5, random_state=1111)
val_df = df_train.drop(train_df.index)

x_train,x_val = (train_df[["user", "item"]].values,val_df[["user", "item"]].values)
# Normalize the targets between 0 and 1. Makes it easy to train.
y_train,y_val = (train_df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values,val_df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values)


model = RecommenderNet(num_users, num_items, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.0007),
)


history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=3,
    verbose=1,
    validation_data=(x_val, y_val),
)

"""We can use this part and sample df_test to inferce some answers"""

df_test, num_users, num_items, min_rating, max_rating = df_encoding(df_test)

#TODO: later have to do something here to make it ragression model to predict number instead of sigmoid and tranform it to 1 to 5 number

