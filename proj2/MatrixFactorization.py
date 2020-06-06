import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf

class MatrixFactorization(Model):
    def __init__(self, num_users, num_items, latent_dim, regularizer=None, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.user_embedding = Embedding(num_users, latent_dim, embeddings_regularizer=regularizer, name='user_embedding')
        self.item_embedding = Embedding(num_items, latent_dim, embeddings_regularizer=regularizer, name='item_embedding')

    def call(self, inputs):
        # print(inputs.shape)
        user_embed = self.user_embedding(inputs[:, 0:1])
        item_embed = self.item_embedding(inputs[:, 1:])
        # print(user_embed.shape, item_embed.shape)
        x = tf.reduce_sum(tf.math.multiply(user_embed, item_embed), axis=-1)
        if inputs.shape[1] == 3:  # Bayesian personalized ranking
            x = x[:, 0:1] - x[:, 1:2]
        x = tf.nn.sigmoid(x)

        return x

    def predict_topk(self, topk, positive):
        matrix = self.user_embedding.get_weights()[0] @ self.item_embedding.get_weights()[0].T
        matrix[positive[:, 0], positive[:, 1]] = 0
        return np.argsort(matrix, axis=1)[:, :-topk + 1:-1]
