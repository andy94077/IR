import os, sys, argparse
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split

import utils
from MatrixFactorization import MatrixFactorization

def bpr_loss(y_true, y_pred):
    return -tf.math.log(y_pred)

def bce_generator(positiveX_pair, negativeX_pair, batch_size=128, epochs=None, seed=None):
    ds_positive = tf.data.Dataset.from_tensor_slices(positiveX_pair)
    ds_negative = tf.data.Dataset.from_tensor_slices(negativeX_pair).shuffle(positiveX_pair.shape[0], seed=seed, reshuffle_each_iteration=True).take(positiveX_pair.shape[0])
    dsX = ds_positive.concatenate(ds_negative)

    dsY = tf.data.Dataset.from_tensor_slices([1.] * positiveX_pair.shape[0] + [0.] * positiveX_pair.shape[0])
    ds = tf.data.Dataset.zip((dsX, dsY)).shuffle(2 * positiveX_pair.shape[0], seed=seed).batch(2 * batch_size).repeat(epochs).prefetch(10)

    return ds

def bpr_generator(positiveX_pair, negativeX, batch_size=128, epochs=None, seed=None):
    ds_positive = tf.data.Dataset.from_tensor_slices(positiveX_pair).shuffle(positiveX_pair.shape[0], seed=seed).batch(batch_size).repeat(epochs).prefetch(10)
    for pos in ds_positive:
        pos = pos.numpy()
        neg = np.array([[np.random.choice(negativeX[u])] for u in pos[:, 0]])
        yield np.concatenate([pos, neg], axis=1), np.zeros(pos.shape[0])

def prepare_training(mode, positiveX, positive, num_users, num_items):
    if mode == 'bce':
        negative = np.concatenate([np.stack([np.array([u] * (num_items - len(positiveX[u]))),
                                             np.setdiff1d(np.arange(num_items), positiveX[u], assume_unique=True)], axis=1) for u in range(num_users)], axis=0)
        negative, valid_negative = train_test_split(negative, test_size=0.1, random_state=880301)
        generator = bce_generator
    else:
        negative = [np.setdiff1d(np.arange(num_items), positiveX[u], assume_unique=True) for u in range(num_users)]
        np.random.seed(880301)
        valid_negative = []
        for i, ll in enumerate(negative):
            train, valid = train_test_split(ll, test_size=0.1, random_state=np.random.randint(2147483647))
            negative[i] = train
            valid_negative.append(valid)
        generator = bpr_generator
    
    positive, valid_positive = train_test_split(positive, test_size=0.1, random_state=880301)
    print(f'\033[32;1mpositive: {positive.shape}, valid_positive: {valid_positive.shape}, negative: {negative.shape if mode == "bce" else len(negative)}, valid_negative: {valid_negative.shape if mode == "bce" else len(valid_negative)}\033[0m')
    return positive, valid_positive, negative, valid_negative, generator
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_file', help='training file')
    parser.add_argument('model_path')
    parser.add_argument('mode', choices=['bce', 'bpr'], help="model mode")
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='output file')
    parser.add_argument('-g', '--gpu', type=str, default='')
    args = parser.parse_args()

    trainX_path = args.training_file
    model_path = args.model_path
    mode = args.mode
    training = not args.no_training
    test = args.test

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    positiveX, num_users, num_items = utils.load_data(trainX_path)
    print(f'\033[32;1mnum_users: {num_users}, num_items: {num_items}\033[0m')
    positive = np.array([[u, i] for u in range(num_users) for i in positiveX[u]])

    if mode == 'bce':
        model = MatrixFactorization(num_users, num_items, latent_dim=128, input_shape=(2,))
        model.compile(Adam(1e-3), loss='binary_crossentropy', metrics=['acc'])
    else:
        model = MatrixFactorization(num_users, num_items, latent_dim=128, input_shape=(3,), regularizer=l2(1e-6))
        model.compile(Adam(1e-4), loss=bpr_loss, metrics=[bpr_loss])
    model.summary()

    if training:
        positive, valid_positive, negative, valid_negative, generator = prepare_training(mode, positiveX, positive, num_users, num_items)

        checkpoint = ModelCheckpoint(model_path, 'val_bpr_loss' if mode == 'bpr' else 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_bpr_loss' if mode == 'bpr' else 'val_loss', 0.8, 20, verbose=1, min_lr=1e-6)
        #logger = CSVLogger(model_path+'.csv', append=True)
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, write_images=True, update_freq=512)
        batch_size = 256
        print('start fitting')
        model.fit(generator(positive, negative, batch_size=batch_size), validation_data=generator(valid_positive, valid_negative, batch_size=batch_size), epochs=300, steps_per_epoch=positive.shape[0] // batch_size, validation_steps=valid_positive.shape[0] // batch_size, callbacks=[checkpoint, reduce_lr])
    else:
        print('\033[32;1mLoading Model\033[0m')

    model.load_weights(model_path)
    if test:
        pred = model.predict_topk(50, positive)
        utils.generate_csv(pred, test)
    else:
        if not training:
            positive, valid_positive, negative, valid_negative, generator = prepare_training(mode, positiveX, positive, num_users, num_items)
        
        print(f'\033[32;1mTraining score: {model.evaluate(generator(positive, negative, epochs=1, batch_size=512), verbose=0)}\033[0m')
        print(f'\033[32;1mValidaiton score: {model.evaluate(generator(valid_positive, valid_negative, epochs=1, batch_size=512), verbose=0)}\033[0m')
