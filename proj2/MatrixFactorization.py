import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf

class MatrixFactorization(Model):
    def __init__(self, num_users, num_items, latent_dim, regularizer=None, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
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

    # def get_config(self):
    #     config = super(MatrixFactorization, self).get_config()
    #     config.update(dict(num_users=self.num_users, num_items=self.num_items, latent_dim=self.latent_dim, predict_topk=self.predict_topk))

def predict_topk(model, topk, positive):
    matrix = tf.matmul(model.user_embedding.weights[0], model.item_embedding.weights[0], transpose_b=True).numpy()
    matrix[positive[:, 0], positive[:, 1]] = 0
    return np.argsort(matrix, axis=1)[:, :-topk + 1:-1]
