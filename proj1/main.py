import pickle
import os, sys, argparse
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
import scipy.sparse.linalg

import utils

VOCAB_ALL_SIZE = 29907
BIGRAM_SIZE = 1171247
WORD_SIZE = VOCAB_ALL_SIZE + BIGRAM_SIZE
FILE_NUM = 46972

def get_file_id(model_dir, document_dir):
    '''
    Return:
        file_id: a list of file ids (string).
    '''
    with open(os.path.join(model_dir, 'file-list'), 'r') as f:
        file_id = [ET.parse(os.path.join(document_dir, name)).getroot().find('.//id').text.lower() for name in f.read().split()]
    return file_id

def get_tf_idf(row, col, value, doc_len, idf, k1=1.5, b=0.75):
    print(f'k1: {k1:.1f}')
    ave_doc_len = np.mean(doc_len)
    value = value * (k1 + 1) / (value + k1 * (1 - b + b * doc_len[row] / ave_doc_len)) * idf[col]
    tf_idf = csr_matrix((value, (row, col)), shape=(FILE_NUM, WORD_SIZE))
    return tf_idf
    
def get_query(query_file, word2idx, k3):
    '''
    Return:
        query_id: a list of query ids (string), query_vector: a csr_matrix(WORD_SIZE, QUERY_SIZE) containing tf of every query.

    query_vector = [ |  |  ... | 
                     |  |  ... |
                     q0 q1 ... qn
                     |  |  ... |
                     |  |  ... | ]
    '''
    print(f'k3: {k3:.1f}')
    root = ET.parse(query_file).getroot()
    query_id = []
    row_col_value = []
    for i, topic in enumerate(root.iter('topic')):
        query_id.append(topic.find('number').text[-3:])
        query = topic.find('title').text + '\n' + topic.find('question').text + '\n' + topic.find('narrative').text + '\n' + topic.find('concepts').text 
        words = list(query) + [w1 + w2 for w1, w2 in zip(query[:-1], query[1:])]
        counter = Counter([w for w in words if w in word2idx])
        row_col_value.extend([[word2idx[w], i, counter[w]] for w in counter])

    row_col_value = np.array(row_col_value, np.int32)
    row, col, value = row_col_value[:, 0], row_col_value[:, 1], row_col_value[:, 2].astype(np.float32)
    value *= (k3 + 1) / (k3 + value)
    return query_id, csr_matrix((value, (row, col)), shape=(WORD_SIZE, len(query_id)), dtype=np.float32)

def predict(cos_sim):
    pred_ranks = [np.argsort(cos_sim[:, i])[:-101:-1] for i in range(cos_sim.shape[1])]
    return pred_ranks

def load_ans_file(path, file_id2idx):
    return [[file_id2idx[file_id] for file_id in ans.split(' ')] for ans in pd.read_csv(path)['retrieved_docs']]

def generate_csv(pred_ranks, query_id, file_id, output_file):
    df = [[query_id[i], ' '.join([file_id[j] for j in pred_rank])] for i, pred_rank in enumerate(pred_ranks)]
    df = pd.DataFrame(df, columns=['query_id', 'retrieved_docs'])
    df.to_csv(output_file, index=False)

def relevance_feedback(queries, alpha, beta, gamma, relevant_set, irrelevant_set):
    print(f'alpha: {alpha:.2f}, beta: {beta:.2f}, gamma: {gamma:.2f}')
    queries = queries.T.toarray()
    for i in range(queries.shape[0]):
        queries[i] = alpha * queries[i] + beta / relevant_set[i].shape[0] * np.array(relevant_set[i].sum(axis=0)).ravel() - gamma / irrelevant_set[i].shape[0] * np.array(irrelevant_set[i].sum(axis=0)).ravel()
    return csr_matrix(queries.T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--relevance-feedback', action='store_true')
    parser.add_argument('-q', '--query-file')
    parser.add_argument('-o', '--ranked-list')
    parser.add_argument('-m', '--model-dir')
    parser.add_argument('-d', '--NTCIR-dir')
    parser.add_argument('-a', '--ans-file')
    args = parser.parse_args()

    is_relevance_feedback = args.relevance_feedback
    query_file = args.query_file
    output_file = args.ranked_list
    model_dir = args.model_dir
    document_dir = args.NTCIR_dir
    ans_file = args.ans_file

    with open(os.path.join('preprocessed', 'word2idx.pickle'), 'rb') as f:
        word2idx = pickle.load(f)
    row, col, value, doc_len, idf = utils.load_npys('preprocessed', ['row.npy', 'col.npy', 'value.npy', 'doc_len.npy', 'idf.npy'])
    tf_idf = get_tf_idf(row, col, value, doc_len, idf, k1=1.8)
    file_id = get_file_id(model_dir, document_dir)
    file_id2idx = {x: i for i, x in enumerate(file_id)}
    query_id, queries = get_query(query_file, word2idx, 5)
    if ans_file:
        trainY = load_ans_file(ans_file, file_id2idx)

    norm = scipy.sparse.linalg.norm(tf_idf, axis=1)
    cos_sim = (tf_idf * queries).toarray() / (scipy.sparse.linalg.norm(tf_idf, axis=1).reshape(-1, 1) + 1e-8) / scipy.sparse.linalg.norm(queries, axis=0)

    if is_relevance_feedback:
        argpartition = np.argpartition(cos_sim, 100, axis=0)
        relevant_set, irrelevant_set = [tf_idf[argpartition[:100, i]] for i in range(argpartition.shape[1])], [tf_idf[argpartition[100:, i]] for i in range(argpartition.shape[1])]
        queries = relevance_feedback(queries, 0.75, 0.2, 0.05, relevant_set, irrelevant_set)
        cos_sim = (tf_idf * queries).toarray() / (scipy.sparse.linalg.norm(tf_idf, axis=1).reshape(-1, 1) + 1e-8) / scipy.sparse.linalg.norm(queries, axis=0)

    pred_ranks = predict(cos_sim)
    if ans_file:
        print(f'map: {utils.mean_average_precision(pred_ranks, trainY):.5f}')
    generate_csv(pred_ranks, query_id, file_id, output_file)
