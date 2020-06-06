import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf

class MatrixFactorization(Model):
    def __init__(self, num_users, num_items, latent_dim, input_shape, regularizer=None, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.user_embedding = Embedding(num_users, latent_dim, embeddings_regularizer=regularizer, name='user_embedding')
        self.item_embedding = Embedding(num_items, latent_dim, embeddings_regularizer=regularizer, name='item_embedding')
        self.item_count = input_shape[0]
        self.build((None,) + input_shape)

    def call(self, inputs):
        user_embed = self.user_embedding(inputs[:, 0:1])
        item_embed = self.item_embedding(inputs[:, 1:])
        if self.item_count == 3:  # Bayesian personalized ranking
            item_embed = item_embed[:, 0:1] - item_embed[:, 1:2]
        x = tf.reduce_sum(tf.math.multiply(user_embed, item_embed), axis=-1)
        x = tf.nn.sigmoid(x)

        return x

    def predict_topk(self, topk, positive):
        matrix = self.user_embedding.get_weights()[0] @ self.item_embedding.get_weights()[0].T
        matrix[positive[:, 0], positive[:, 1]] = 0
        return np.argsort(matrix, axis=1)[:, :-topk + 1:-1]
