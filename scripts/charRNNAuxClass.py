from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np 

class Highway(nn.Module):
    def __init__(self,embed_size):
        super(Highway,self).__init__()
        self.e_word = embed_size
        self.Wproject = nn.Linear(embed_size,embed_size, bias=True)
        self.Wgate = nn.Linear(embed_size,embed_size, bias=True)

    def forward(self,x_convout):
        projection = F.relu(self.Wproject(x_convout))
        gate = torch.sigmoid(self.Wgate(x_convout))
        x_highway = torch.mul(projection,gate)+torch.mul(1-gate,x_convout)
        return x_highway
    
class CNN(nn.Module):
    def __init__(self,embed_size, embed_char_size,k = 5):
        super(CNN,self).__init__()
        self.e_word = embed_size
        self.e_char = embed_char_size
        self.m_word = 21
        self.conv = nn.Conv1d(self.e_char, self.e_word, k,  bias=True)
        self.pool = nn.MaxPool1d(self.m_word-k+1)
    def forward(self,x_reshaped):
        s,b,w,h = x_reshaped.size()
        x_reshaped = x_reshaped.contiguous().view(s*b,w,h)
        conv = self.conv(x_reshaped)
        c = self.pool(F.relu(conv))
        convout = c.contiguous().view(c.size(0),conv.size(1))
        convout = convout.contiguous().view(s,b,self.e_word)
        return convout

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab,dropout_prob = 0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.e_char = 50
        self.embed_size =embed_size
        self.vocab = vocab
        self.dropout_prob = dropout_prob
        self.char_embeddings = nn.Embedding(len(vocab.char2id),self.e_char,padding_idx=vocab.char2id['<pad>'])
        self.cnn = CNN(self.embed_size,self.e_char)
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(dropout_prob)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """

        ### YOUR CODE HERE for part 1j
        output = self.char_embeddings(input)
        output = output.contiguous().permute(0,1,3,2)
        cnn_out = self.cnn(output)
        highway_out = self.dropout(self.highway(cnn_out))
        return highway_out

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.e_char = char_embedding_size
        self.v_char = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(self.e_char,self.hidden_size,1,bias = True)
        self.char_output_projection = nn.Linear(self.hidden_size,self.v_char, bias = True)
        self.decoderCharEmb = nn.Embedding(self.v_char,self.e_char,padding_idx = target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, 
        batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, 
        batch, hidden_size)
        """
        emb = self.decoderCharEmb(input)
        #print('embedding',emb)
        output, states = self.charDecoder(emb, dec_hidden)
        #print('output',output.shape,output)
        scores = self.char_output_projection(output)
        #print('scores',scores,scores.shape,self.target_vocab)
        return scores, states
        
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        scores, states = self.forward(char_sequence[:-1],dec_hidden) 
        celoss = nn.CrossEntropyLoss(ignore_index = self.target_vocab.char2id['<pad>'])
        length, batch = char_sequence[1:].shape
        scores = scores.contiguous().view(length*batch,-1)
        target = char_sequence[1:].contiguous().view(-1)
        #scores[target == self.target_vocab.char2id['<pad>'],self.target_vocab.char2id['<pad>']]= 0
        softmax = celoss(scores,target)
        return softmax
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        _,batch,hidden_size = initialStates[0].shape
        output_word = [[] for _ in range(batch)]
        start_char = self.target_vocab.start_of_word
        start_char_arr = np.repeat(start_char, (batch))
        curr_char = torch.tensor(start_char_arr,device = device)
        states = initialStates
        for t in range(max_length):
            curr_emb = self.decoderCharEmb(curr_char)
            output, states = self.charDecoder(curr_emb.contiguous().view(1,batch,self.e_char),states)
            scores = self.char_output_projection(states[0].contiguous().view(1,batch,self.hidden_size))
            softmax = nn.Softmax(2)
            curr_char = torch.argmax(softmax(scores),2).contiguous().view(batch)
            for i in range(batch):
                output_word[i].append(self.target_vocab.id2char[int(curr_char[i])])
        output_words = [ ''.join(o) for o in output_word]
        output_words = [ w[:w.find('}')] if w.find('}') else w for w in output_words ]
        return output_words