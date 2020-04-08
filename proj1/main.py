import pickle
import os, sys, argparse
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
import scipy.sparse.linalg

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--relevance-feedback', action='store_true')
    parser.add_argument('-q', '--query-file')
    parser.add_argument('-o', '--ranked-list')
    parser.add_argument('-m', '--model-dir')
    parser.add_argument('-d', '--NTCIR-dir')
    args = parser.parse_args()

    is_relevance_feedback = args.relevance_feedback
    query_file = args.query_file
    output_file = args.ranked_list
    model_dir = args.model_dir
    document_dir = args.NTCIR_dir

    with open(os.path.join('preprocessed', 'word2idx.pickle'), 'rb') as f:
        word2idx = pickle.load(f)
    tf_idf = load_npz(os.path.join('preprocessed', 'tf_idf.npz'))
    file_id = get_file_id(model_dir, document_dir)
    query_id, queries = get_query(query_file, word2idx, 5)

    #print(queries[word2idx['流浪'], 0])
    norm = scipy.sparse.linalg.norm(tf_idf, axis=1)
    cos_sim = (tf_idf * queries).toarray() / (scipy.sparse.linalg.norm(tf_idf, axis=1).reshape(-1, 1) + 1e-8) / scipy.sparse.linalg.norm(queries, axis=0)
    #print(cos_sim[496, 0])
    #print(cos_sim[21256, 0])
    #exit()
    threshold = 0.1
    df = []
    for i in range(cos_sim.shape[1]):
        rank = np.argsort(cos_sim[:, i])[:-101:-1]
        df.append([query_id[i], ' '.join([file_id[j] for j in rank])])
    df = pd.DataFrame(df, columns=['query_id', 'retrieved_docs'])
    df.to_csv(output_file, index=False)
