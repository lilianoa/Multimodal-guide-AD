from __future__ import print_function
import os
import json

import numpy as np
from datasets.dataset import Dictionary
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
文本预处理流程：
1.读入文本
2.分词
3.建立字典，将每个词映射到一个唯一的索引(index)
4.将文本从词的序列转换为索引的序列，方便输入模型
"""

def create_dictionary(dataroot):
    dictionary = Dictionary()
    files = [
            'test_captions.json',
            'train_captions.json',
            'val_captions.json'
    ]
    for path in files:
        caption_path = os.path.join(dataroot, path)
        caps = json.load(open(caption_path))['captions']
        for cap in caps:
            dictionary.tokenize(cap['caption'], True)

    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    print("number of glove", len(entries))
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            print("cannot fin in GloVe:", word)
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    dataroot = "../data/gas_data/caption_s_json"
    data_root = "../data"

    dictionary_path = os.path.join(dataroot, 'dictionary.pkl')

    d = create_dictionary(dataroot)
    d.dump_to_file(dictionary_path)

    d = Dictionary.load_from_file(dictionary_path)
    emb_dim = 300
    # zip_file_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    # r = requests.get(zip_file_url)
    # z = zipfile.ZipFile(io.BytesIO(r.content))
    # z.extractall(data_root)
    glove_file = '../data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(os.path.join(data_root, 'glove6b_init_%dd.npy' % emb_dim), weights)
