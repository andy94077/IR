import os, sys, argparse
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, CuDNNGRU, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf

import utils
from MatrixFactorization import MatrixFactorization

def bpr_loss(y_true, y_pred):
    return tf.math.log(y_pred)

def bce_generator(positiveX, num_users, num_items, batch_size=128, seed=None):
    positiveX_pair = np.array([[u, i] for u in range(num_users) for i in positiveX[u]])
    negativeX_pair = np.array([[u, i] for u in range(num_users) for i in np.setdiff1d(np.arange(num_items), positiveX[i], assume_unique=True)])
    if seed is not None:
        np.random.seed(seed)

    while True:
        positive_idx = np.random.permutation(positiveX_pair.shape[0])
        negative_idx = np.random.choice(max(negativeX_pair.shape[0], positiveX_pair.shape[0]), size=positiveX_pair.shape[0], replace=False) % negativeX_pair.shape[0]
        for i in range(0, positive_idx.shape[0], batch_size):
            batch_positiveX_pair = positiveX_pair[positive_idx[i:i+batch_size]]
            batch_negativeX_pair = negativeX_pair[negative_idx[i:i+batch_size]]
            yield np.concatenate([batch_positiveX_pair, batch_negativeX_pair], axis=0), np.concatenate([np.ones(batch_positiveX_pair.shape[0]), np.zeros(batch_negativeX_pair.shape[0])], axis=0)

def bpr_generator(positiveX, num_users, num_items, batch_size=128, seed=None):
    positiveX_pair = np.array([[u, i] for u in range(num_users) for i in positiveX[u]])
    negativeX = [np.setdiff1d(np.arange(num_items), positiveX[i], assume_unique=True) for u in range(num_users)]
    if seed is not None:
        np.random.seed(seed)

    while True:
        positive_idx = np.random.permutation(positiveX_pair.shape[0])
        for i, _ in enumerate(negativeX):
            np.random.shuffle(negativeX[i])
        for i in range(0, positive_idx.shape[0], batch_size):
            batch_positiveX_pair = positiveX_pair[positive_idx[i:i+batch_size]]
            batch_negativeX = np.array([[negativeX[u][n % len(negativeX[u])]] for u, n in zip(batch_positiveX_pair[:, 0], np.random.choice(num_items, size=batch_positiveX_pair.shape[0], replace=True))])
            yield np.concatenate([batch_positiveX_pair, batch_negativeX], axis=1), np.zeros(batch_positiveX_pair.shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('mode', help="model mode = ['bce', 'bpr']")
    parser.add_argument('-t', '--training-file', help='training file')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='output file')
    parser.add_argument('-g', '--gpu', type=str, default='')
    args = parser.parse_args()

    model_path = args.model_path
    mode = args.mode
    trainX_path = args.training_file
    training = not args.no_training
    test = args.test

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    max_seq_len = 32
    w2v_model = Word2Vec().load(word2vec_model_path)
    word2idx = w2v_model.get_word2idx()
    embedding = w2v_model.get_embedding()
    vocabulary_size = len(word2idx)
    print(f'\033[32;1mvocabulary_size: {vocabulary_size}\033[0m')

    if function not in globals():
        globals()[function] = getattr(importlib.import_module(function[:function.rfind('.')]), function.split('.')[-1])

    if training:
        positiveX, num_users, num_items = utils.load_data(trainX_path)
        
        #trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
        #print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

        if mode == 'bce':
            model = MatrixFactorization(num_users, num_items, latent_dim=128)
            model.compile(Adam(1e-3), loss='binary_crossentropy', metrics=['acc'])
            generator = bce_generator
        else:
            model = MatrixFactorization(num_users, num_items, latent_dim=128, regularizer=l2(1e-3))
            model.compile(Adam(1e-3), loss=bpr_loss)
            generator = bpr_generator
        model.summary()

        checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_loss', 0.8, 2, verbose=1, min_lr=1e-5)
        #logger = CSVLogger(model_path+'.csv', append=True)
        #tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', histogram_freq=1, batch_size=1024, write_grads=True, write_images=True, update_freq=512)

        model.fit(generator(positiveX, num_users, num_items, batch_size=256), epochs=10, steps_per_epoch=positiveX.shape[0]//batch_size, callbacks=[checkpoint, reduce_lr])
        model.load_weights(model_path)
        model.save(model_path)
    else:
        print('\033[32;1mLoading Model\033[0m')
        load_model(model_path)


    if test:
        testX = utils.load_test_data(test[0], word2idx, max_seq_len)
        pred = model.predict(testX, batch_size=256)
        if ensemble:
            np.save(test[1], pred)
        else:
            utils.generate_csv(pred, test[1])
    else:
        if not training:
            trainX, trainY = utils.load_train_data(labeled_path, word2idx, max_seq_len)
            trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, split_ratio=0.1)
            print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')
        print(f'\033[32;1mTraining score: {model.evaluate(trainX, trainY, batch_size=256, verbose=0)}\033[0m')
        print(f'\033[32;1mValidaiton score: {model.evaluate(validX, validY, batch_size=256, verbose=0)}\033[0m')
