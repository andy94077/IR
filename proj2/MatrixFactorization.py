import os, sys, argparse
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, CuDNNGRU, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf

import utils

class MatrixFactorization(Model):
    def __init__(self, num_users, num_items, latent_dim, regularizer=None, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.user_embedding = Embedding(num_users, latent_dim, input_length=1, embeddings_regularizer=regularizer, name='user_embedding')
        self.item_embedding = Embedding(num_items, latent_dim, embeddings_regularizer=regularizer, name='item_embedding')

    def call(self, inputs):
        user_embed = self.user_embedding(inputs[:, 0:1])
        item_embed = self.item_embedding(inputs[:, 1:])

        x = tf.matmul(user_embed, item_embed, transpose_a=True)
        if inputs.shape[1] == 3:  # Bayesian personalized ranking
            x = x[:, 0:1] - x[:, 1:2]
        x = tf.nn.sigmoid(x)

        return x

    def predict_topk(self, topk):
        return np.argsort(self.user_embedding.get_weights()[0] @ self.item_embedding.get_weights()[0].T, axis=1)[:, :-topk + 1:-1]
