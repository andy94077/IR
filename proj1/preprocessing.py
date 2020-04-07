import pickle
import os, sys, argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

VOCAB_ALL_SIZE = 29907
BIGRAM_SIZE = 1171247
WORD_SIZE = VOCAB_ALL_SIZE + BIGRAM_SIZE
FILE_NUM = 46972

def load_npys(data_dir, args):
    npys = [np.load(os.path.join(data_dir, i)) for i in args]
    if len(args) == 1:
        return npys[0]
    return npys

def preprocessing(model_dir, document_dir):
    '''
    Return:
        idx2word: list(WORD_SIZE), word2idx: dict, row: np.array, col: np.array, value: np.array, doc_len: np.array(FILE_NUM), idf: np.array(WORD_SIZE).
    '''
    with open(os.path.join(model_dir, 'vocab.all'), 'r') as f:
        idx2word = [w.rstrip() for w in f.readlines()[1:]]
    word2idx = {w: i for i, w in enumerate(idx2word)}

    with open(os.path.join(model_dir, 'file-list'), 'r') as f:
        doc_len = np.array([get_doc_len(os.path.join(document_dir, name)) for name in f.read().split()])

    row, col, value = [], [], []
    idf = np.zeros(WORD_SIZE, np.float32)
    with open(os.path.join(model_dir, 'inverted-file'), 'r') as f:
        f_all = f.readlines() 
    i = 0
    while i < len(f_all):
        w1, w2, n = [int(j) for j in f_all[i].split()]
        i += 1
        if w2 == -1:  # unigram
            word_id = w1 - 1
        else:  # bigram
            word_id = len(idx2word) - 1
            idx2word.append(idx2word[w1 - 1] + ' ' + idx2word[w2 - 1])
            word2idx[idx2word[w1 - 1] + ' ' + idx2word[w2 - 1]] = word_id

        idf[word_id] += n
        for _ in range(n):
            file_i, count = [int(j) for j in f_all[i].split()]
            row.append(file_i)
            col.append(word_id)
            value.append(count)
            i += 1
        print(i, end='\r')
    print(np.mean(value))
    idf = np.log((FILE_NUM - idf + 0.5) / (idf + 0.5))

    return idx2word, word2idx, np.array(row, np.int32), np.array(col, np.int32), np.array(value, np.float32), doc_len, idf

def get_doc_len(filename):
    root = ET.parse(filename).getroot()
    title = root.find('./doc/title').text
    doc_len = len(title.strip()) if title is not None else 0
    doc_len += np.sum([len(p.text.strip()) for p in root.findall('.//p')])
    return doc_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('document_dir')
    args = parser.parse_args()

    model_dir = args.model_dir
    document_dir = args.document_dir
    
    if not os.path.exists(os.path.join('preprocessed', 'idx2word.pickle')) or not os.path.exists(os.path.join('preprocessed', 'word2idx.pickle')) or not os.path.exists(os.path.join('preprocessed', 'row.npy')) or not os.path.exists(os.path.join('preprocessed', 'col.npy')) or not os.path.exists(os.path.join('preprocessed', 'value.npy')) or not os.path.exists(os.path.join('preprocessed', 'doc_len.npy')) or not os.path.exists(os.path.join('preprocessed', 'idf.npy')):
        idx2word, word2idx, row, col, value, doc_len, idf = preprocessing(model_dir, document_dir)
        with open(os.path.join('preprocessed', 'idx2word.pickle'), 'wb') as f:
            pickle.dump(idx2word, f)
        with open(os.path.join('preprocessed', 'word2idx.pickle'), 'wb') as f:
            pickle.dump(word2idx, f)
        np.save(os.path.join('preprocessed', 'row.npy'), row)
        np.save(os.path.join('preprocessed', 'col.npy'), col)
        np.save(os.path.join('preprocessed', 'value.npy'), value)
        np.save(os.path.join('preprocessed', 'doc_len.npy'), doc_len)
        np.save(os.path.join('preprocessed', 'idf.npy'), idf)
    else:
        with open(os.path.join('preprocessed', 'idx2word.pickle'), 'rb') as f:
            idx2word = pickle.load(f)
        with open(os.path.join('preprocessed', 'word2idx.pickle'), 'rb') as f:
            word2idx = pickle.load(f)
        row, col, value, doc_len, idf = load_npys('preprocessed', ['row.npy', 'col.npy', 'value.npy', 'doc_len.npy', 'idf.npy'])

    k1, b = 1.5, 0.75
    ave_doc_len = np.mean(doc_len)
    value *= (k1 + 1) / (value + k1 * (1 - b + b * doc_len[row] / ave_doc_len)) * idf[col]
    tf_idf = csr_matrix((value, (row, col)), shape=(FILE_NUM, len(idx2word)))
    
    save_npz(os.path.join('preprocessed', 'tf_idf.npz'), tf_idf, compressed=False)

