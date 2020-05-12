import os
import torch
import numpy as np
import random

import math
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import datetime
import shutil

PAD_WORD="<pad>"
EOS_WORD="<eos>"
BOS_WORD="<bos>"
UNK="<unk>"

'''
Methods for File Output Consistency in train scripts
'''
# create default output file name in train scripts
def output_file_name(data_path, outf, script):
    # get current date time
    currentDT = datetime.datetime.now()
    date = currentDT.strftime("%m%d-%H%M")

    if(outf=="_output"):
        outf = data_path + outf + "/" + script + "_" + date
    os.makedirs(outf, exist_ok=True)
    return outf

# make output directory if it doesn't already exist
def make_output_directory(data_path, outf):
    if os.path.isdir(outf):
        files = [ f for f in os.listdir(outf) if os.path.isfile(outf+'/'+f) ]
        oldoutf = outf + "/" + "old_output"
        os.makedirs(oldoutf,exist_ok=True)
        for file in files:
            print("moving from {} to {}".format(outf + "/" + file, oldoutf+"/"+file) )
            shutil.move(outf + "/" + file, oldoutf)
    # make output directory if it doesn't already exist
    else:
        os.makedirs(outf)


def load_kenlm():
    global kenlm
    import kenlm


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.word2idx[PAD_WORD] = 0
            self.word2idx[BOS_WORD] = 1
            self.word2idx[EOS_WORD] = 2
            self.word2idx[UNK] = 3
            self.wordcounts = {}
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                    {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("Original vocab {}; Pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, datafiles, maxlen, vocab_size=11000, lowercase=False, vocab=None, debug=False):
        self.dictionary = Dictionary(vocab)
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.datafiles = datafiles
        self.forvocab = []
        self.data = {}

        if vocab is None:
            for path, name, fvocab in datafiles:
                if fvocab or debug:
                    self.forvocab.append(path)
            self.make_vocab()

        for path, name, _ in datafiles:
            self.data[name] = self.tokenize(path)


    def make_vocab(self):
        for path in self.forvocab:
            assert os.path.exists(path)
            # Add words to the dictionary
            with open(path, 'r',encoding="utf-8") as f:
                for line in f:
                    #print(line)
                    #print(type(line))
                    L = line.lower() if self.lowercase else line
                    words = L.strip().split(" ")
                    for word in words:
                        self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r', encoding="utf-8") as f:
        
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                L = line.lower() if self.lowercase else line
                words = L.strip().split(" ")
                if self.maxlen > 0 and len(words) > self.maxlen:
                    dropped += 1
                    continue
                words = [BOS_WORD] + words + [EOS_WORD]
                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab[UNK]
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines


def batchify(data, bsz, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)

    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        batch = data[i*bsz:(i+1)*bsz]
        
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        words = batch
        lengths = [len(x)-1 for x in words]

        # sort items by length (decreasing)
        batch, lengths = length_sort(batch, lengths)
        words = batch

        # source has no end symbol
        source = [x[:-1] for x in words]
        # target has no start symbol
        target = [x[1:] for x in words]

        # find length to pad to
        maxlen = max(lengths)
        for x, y in zip(source, target):
            zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros

        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)

        batches.append((source, target, lengths))
    print('{} batches'.format(len(batches)))
    return batches


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


def truncate(words):
    # truncate sentences to first occurrence of <eos>
    truncated_sent = []
    for w in words:
        if w != EOS_WORD:
            truncated_sent.append(w)
        else:
            break
    sent = " ".join(truncated_sent)
    return sent


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    
    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

    load_kenlm()
    # create language model
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10**-(total_nll/total_wc)
    return ppl

"""below is stuff added from a5 utils.py """

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()` 
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal 
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21 

    ### YOUR CODE HERE for part 1f
    ### TODO:
    ###     Perform necessary padding to the sentences in the batch similar to the pad_sents() 
    ###     method below using the padding character from the arguments. You should ensure all 
    ###     sentences have the same number of words and each word has the same number of 
    ###     characters. 
    ###     Set padding words to a `max_word_length` sized vector of padding characters.  
    ###
    ###     You should NOT use the method `pad_sents()` below because of the way it handles 
    ###     padding and unknown words.

    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    max_length = 0
    for sent in sents:
        # find which longest sentence
        if len(sent) > max_length:
            max_length = len(sent)
    
    for sent in sents:
        # pad each sentence to be length max_length
        diff = max_length - len(sent)
        if diff > 0: 
            sents_padded.append(sent + ([[]] * diff))
        else:
            sents_padded.append(sent)
    
    for sent in sents_padded:
        for i in range(len(sent)):
            # pad each word to max_word_length
            diff = max_word_length - len(sent[i])
            if diff < 0:
                sent[i] = sent[i][0:max_word_length] # truncate
            else:
                sent[i] = sent[i] + ([char_pad_token] * diff) # pad
    
    ### END YOUR CODE

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents