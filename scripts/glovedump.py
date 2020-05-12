import pickle
import bcolz
import numpy as np
words = []
idx = 0
word2idx = {}
glove_path = './yelp'
vectors = bcolz.carray(np.zeros(1), rootdir=f'6B.300.dat', mode='w')

with open(f'glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400001, 300)), rootdir=f'6B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'6B.300words.pkl', 'wb'))
pickle.dump(word2idx, open(f'6B.300_idx.pkl', 'wb'))


vectors = bcolz.open(f'6B.300.dat')[:]
words = pickle.load(open(f'6B.300words.pkl', 'rb'))
word2idx = pickle.load(open(f'6B.300_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

