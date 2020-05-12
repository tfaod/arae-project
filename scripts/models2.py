import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import to_gpu
import json
import os
import numpy as np
import pdb

from charRNNAuxClass import ModelEmbeddings, CharDecoder


class MLP_Classify(nn.Module):

    def __init__(self, ninput, noutput, layers,activation=nn.ReLU(), gpu=False):
        super(MLP_Classify, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            # No batch normalization in first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i + 1])
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.sigmoid(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq2Decoder(nn.Module):

    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_r=0.2,\
                 share_decoder_emb=False, hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq2Decoder, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder1 = nn.Embedding(ntokens, emsize)
        self.embedding_decoder2 = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,\
                               hidden_size=nhidden,\
                               num_layers=nlayers,\
                               dropout=dropout,\
                               batch_first=True)

        decoder_input_size = emsize + nhidden
        self.decoder1 = nn.LSTM(input_size=decoder_input_size,\
                               hidden_size=nhidden,\
                               num_layers=1,\
                               dropout=dropout,\
                               batch_first=True)
        self.decoder2 = nn.LSTM(input_size=decoder_input_size,\
                               hidden_size=nhidden,\
                               num_layers=1,\
                               dropout=dropout,\
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

        if share_decoder_emb:
            self.embedding_decoder2.weight = self.embedding_decoder1.weight

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder1.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder2.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder1.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder2.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, whichdecoder, indices, lengths, noise=False, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        if encode_only:
            return hidden

        decoded = self.decode(whichdecoder, hidden, batch_size, maxlen,indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        #pdb.set_trace()
        #packed_embeddings = pack_padded_sequence(input=embeddings,lengths=lengths,batch_first=True)
        packed_embeddings = pack_padded_sequence(embeddings,lengths,True)
        # Encode
        packed_output, state = self.encoder(packed_embeddings)
        hidden, cell = state

        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)

        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        # hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(self.noise_r,torch.zeros(hidden.size()))
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, whichdecoder, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        if whichdecoder == 1:
            embeddings = self.embedding_decoder1(indices)
        else:
            embeddings = self.embedding_decoder2(indices)

        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(augmented_embeddings,lengths = lengths,batch_first = True)

        if whichdecoder == 1:
            packed_output, state = self.decoder1(packed_embeddings, state)
        else:
            packed_output, state = self.decoder2(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, whichdecoder, hidden, maxlen, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)
        self.start_symbols = to_gpu(self.gpu, self.start_symbols)

        if whichdecoder == 1:
            embedding = self.embedding_decoder1(self.start_symbols)
        else:
            embedding = self.embedding_decoder2(self.start_symbols)

        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            if whichdecoder == 1:
                output, state = self.decoder1(inputs, state)
            else:
                output, state = self.decoder2(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
                indices = indices.unsqueeze(1)
            else:
                assert 1 == 0
                # sampling
                probs = F.softmax(overvocab / temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            if whichdecoder == 1:
                embedding = self.embedding_decoder1(indices)
            else:
                embedding = self.embedding_decoder2(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices
    
class CharSeq2Seq2Decoder(nn.Module):

    def __init__(self, emsize, nhidden, ntokens, vocab, nlayers, noise_r=0.2,\
                 share_decoder_emb=False, hidden_init=False, dropout=0, gpu=False):
        super(CharSeq2Seq2Decoder, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.vocab = vocab
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.model_embeddings_source = ModelEmbeddings(emsize, vocab) # todo link vocab.py when calling
        self.model_embeddings_target = ModelEmbeddings(emsize, vocab)
        
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder1 = nn.Embedding(ntokens, emsize)
        self.embedding_decoder2 = nn.Embedding(ntokens, emsize)

        
        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,\
                               hidden_size=nhidden,\
                               num_layers=nlayers,\
                               dropout=dropout,\
                               batch_first=True)

        decoder_input_size = emsize + nhidden
        self.decoder1 = nn.LSTM(input_size=decoder_input_size,\
                               hidden_size=nhidden,\
                               num_layers=1,\
                               dropout=dropout,\
                               batch_first=True)
        self.decoder2 = nn.LSTM(input_size=decoder_input_size,\
                               hidden_size=nhidden,\
                               num_layers=1,\
                               dropout=dropout,\
                               batch_first=True)

        self.charDecoder1 = CharDecoder(nhidden, target_vocab=vocab) 
        self.charDecoder2 = CharDecoder(nhidden, target_vocab=vocab) 
        
        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

        if share_decoder_emb:
            self.embedding_decoder2.weight = self.embedding_decoder1.weight

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder1.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder2.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder1.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder2.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, source, target, whichdecoder, indices, lengths, noise=False, encode_only=False):
        """
        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`
        """
        batch_size, maxlen = indices.size()

        source_padded_chars = self.vocab.to_input_tensor_char(source, device=self.device) # shape (max sent length, b, m_word)
        target_padded_chars = self.vocab.to_input_tensor_char(target, device=self.device)
                
        hidden = self.encode(source_padded_chars, indices, lengths, noise)

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        if encode_only:
            return hidden

        decoded = self.decode(whichdecoder, hidden, batch_size, maxlen,indices=indices, lengths=lengths)

        return decoded

    def encode(self, source_padded, indices, lengths, noise):
        """
        source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        """
        embeddings = self.embedding(indices)
        #pdb.set_trace()
        #packed_embeddings = pack_padded_sequence(input=embeddings,lengths=lengths,batch_first=True)
        packed_embeddings = pack_padded_sequence(embeddings,lengths,True)
        
        X = self.model_embeddings_source(source_padded)
        X_packed = pack_padded_sequence(X, lengths, True)
        # Encode
        packed_output, state = self.encoder(X_packed)
        hidden, cell = state

        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)

        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        # hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(self.noise_r,torch.zeros(hidden.size()))
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, whichdecoder, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        if whichdecoder == 1:
            embeddings = self.embedding_decoder1(indices)
            
        else:
            embeddings = self.embedding_decoder2(indices)

        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(augmented_embeddings,lengths = lengths,batch_first = True)

        if whichdecoder == 1:
            packed_output, state = self.decoder1(packed_embeddings, state)
        else:
            packed_output, state = self.decoder2(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, whichdecoder, hidden, maxlen, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)
        self.start_symbols = to_gpu(self.gpu, self.start_symbols)

        if whichdecoder == 1:
            embedding = self.embedding_decoder1(self.start_symbols)
        else:
            embedding = self.embedding_decoder2(self.start_symbols)

        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            if whichdecoder == 1:
                output, state = self.decoder1(inputs, state)
            else:
                output, state = self.decoder2(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
                indices = indices.unsqueeze(1)
            else:
                assert 1 == 0
                # sampling
                probs = F.softmax(overvocab / temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            if whichdecoder == 1:
                embedding = self.embedding_decoder1(indices)
            else:
                embedding = self.embedding_decoder2(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices

class CNN_D(torch.nn.Module):

    def __init__(self, ninput, noutput, filter_sizes='1-2-3-4-5', n_filters=128, activation=nn.LeakyReLU(0.2), gpu=False):
        super(CNN_D, self).__init__()

        self.n_filters = n_filters
        self.ninput = ninput
        self.noutput = noutput
        #FIX THIS
        self.filter_sizes =  [int(x) for x in filter_sizes.split('-')]
        print(self.filter_sizes)
        print(ninput)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=n_filters, padding=int((size-1)/2), kernel_size=size) for size in self.filter_sizes])
        self.activation=nn.LeakyReLU(.2)
        self.dropout = nn.Dropout(0.3)
        # flatten before feeding into fully connected
        self.fc = nn.Linear(ninput, noutput, bias=True)  # calculate size here,1 bias = True)

   
    def forward(self, x):
        # x shape is (N, C_in)
        print(x.shape)
        c_in = x.shape[-1]
        x = x.unsqueeze(1)
        # x is shape (N, C_in, 1)
        outputs = []
        for i in range(len(self.convs)):
            h = self.activation(self.convs[i](x))
            # output shape is 
            print("for i: {}, output shape is: {}".format(i, str(h.shape)))
            pooled, pooled_idx = torch.max(h, dim=1)
            print(pooled.shape)
            pooled = pooled.view(-1, self.n_filters).contiguous()
            print("reshaped pool is: {}".format(pooled.shape))
            outputs.append(pooled)
        outputs=torch.cat(outputs, 1)

        outputs=self.dropout(x)
        print("outputs shape is: {}".format(outputs.shape))
#         outputs=outputs.view(-1).contiguous()
        logits=self.fc(outputs)
        print("logits shape is: {}".format(logits.shape))

        return logits



    
    
    
# class CNN_D(torch.nn.Module):

#     def __init__(self, ninput, noutput, layers, activation=nn.LeakyReLU(0.2), gpu=False):
#         #pdb.set_trace()
#         super(CNN_D, self).__init__()
#         self.ninput = ninput
#         self.noutput = noutput
#         self.conv1 = nn.Conv2d(self.ninput, 1, kernel_size=4,\
#                                stride=1, padding=2, bias=True)
#         self.activation = activation
#         self.dropout = nn.Dropout(0.3)
#         #pdb.set_trace()
#         # flatten before feeding into fully connected
#         self.fc1 = nn.Linear(256, noutput, bias=True)  # calculate size here,1 bias = True)
#         #256 is palce holder
#     def forward(self, x):
#         #pdb.set_trace()
#         # get last item
#         x = x.unsqueeze(-1)
#         x = x.unsqueeze(-1)
#         #print(x.shape)
#         #pdb.set_trace()
#         x=self.conv1(x)
#         x=self.dropout(self.activation(x))
#         x=x.view(-1)
#         #print(self.noutput,'after',x.shape)
#         logits=self.fc1(x)
#         #print(logits.shape)
#         #pdb.set_trace()
#         return logits



class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput=ninput
        self.noutput=noutput

        layer_sizes=[ninput] + [int(x) for x in layers.split('-')]
        self.layers=[]

        for i in range(len(layer_sizes) - 1):
            layer=nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn=nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer=nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x=layer(x)
        return x

    def init_weights(self):
        init_std=0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_r=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden=nhidden
        self.emsize=emsize
        self.ntokens=ntokens
        self.nlayers=nlayers
        self.noise_r=noise_r
        self.hidden_init=hidden_init
        self.dropout=dropout
        self.gpu=gpu

        self.start_symbols=to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding=nn.Embedding(ntokens, emsize)
        self.embedding_decoder=nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder=nn.LSTM(input_size=emsize,hidden_size=nhidden,num_layers=nlayers,dropout=dropout,batch_first=True)

        decoder_input_size=emsize + nhidden
        self.decoder=nn.LSTM(input_size=decoder_input_size,\
                               hidden_size=nhidden,\
                               num_layers=1,\
                               dropout=dropout,\
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear=nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange=0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1=Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2=Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros=Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm=torch.norm(grad, 2, 1)
        self.grad_norm=norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen=indices.size()

        hidden=self.encode(indices, lengths, noise)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded=self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings=self.embedding(indices)
        packed_embeddings=pack_padded_sequence(input=embeddings,\
                                                 lengths=lengths,\
                                                 batch_first=True)

        # Encode
        packed_output, state=self.encoder(packed_embeddings)

        hidden, cell=state
        # batch_size x nhidden
        hidden=hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms=torch.norm(hidden, 2, 1)

        # For older versions of PyTorch use:
        hidden=torch.div(hidden, norms.expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        # hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_r > 0:
            gauss_noise=torch.normal(means=torch.zeros(hidden.size()),
                                       std=self.noise_r)
            hidden=hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden=hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state=(hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state=self.init_hidden(batch_size)

        embeddings=self.embedding_decoder(indices)
        augmented_embeddings=torch.cat([embeddings, all_hidden], 2)
        packed_embeddings=pack_padded_sequence(input=augmented_embeddings,\
                                                 lengths=lengths,\
                                                 batch_first=True)

        packed_output, state=self.decoder(packed_embeddings, state)
        output, lengths=pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded=self.linear(output.contiguous().view(-1, self.nhidden))
        decoded=decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size=hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state=(hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state=self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding=self.embedding_decoder(self.start_symbols)
        inputs=torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices=[]
        for i in range(maxlen):
            output, state=self.decoder(inputs, state)
            overvocab=self.linear(output.squeeze(1))

            if not sample:
                vals, indices=torch.max(overvocab, 1)
            else:
                # sampling
                probs=F.softmax(overvocab / temp)
                indices=torch.multinomial(probs, 1)

            all_indices.append(indices)

            embedding=self.embedding_decoder(indices)
            inputs=torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices=torch.cat(all_indices, 1)

        return max_indices

def load_models(load_path, epoch, twodecoders=False):
    model_args=json.load(open("{}/args.json".format(load_path), "r"))
    word2idx=json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word={v: k for k, v in word2idx.items()}

    if not twodecoders:
        autoencoder=Seq2Seq(emsize=model_args['emsize'],\
                              nhidden=model_args['nhidden'],\
                              ntokens=model_args['ntokens'],\
                              nlayers=model_args['nlayers'],\
                              hidden_init=model_args['hidden_init'])
    else:
        autoencoder=Seq2Seq2Decoder(emsize=model_args['emsize'],\
                              nhidden=model_args['nhidden'],\
                              ntokens=model_args['ntokens'],\
                              nlayers=model_args['nlayers'],\
                              hidden_init=model_args['hidden_init'])

    gan_gen=MLP_G(ninput=model_args['z_size'],\
                    noutput=model_args['nhidden'],\
                    layers=model_args['arch_g'])
    gan_disc=MLP_D(ninput=model_args['nhidden'],\
                     noutput=1,\
                     layers=model_args['arch_d'])

    print('Loading models from' + load_path)
    ae_path=os.path.join(load_path, "autoencoder_model_{}.pt".format(epoch))
    gen_path=os.path.join(load_path, "gan_gen_model_{}.pt".format(epoch))
    disc_path=os.path.join(load_path, "gan_disc_model_{}.pt".format(epoch))

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc


def generate(autoencoder, gan_gen, z, vocab, sample, maxlen):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise=z
    elif type(z) == torch.FloatTensor or type(z) == torch.cuda.FloatTensor:
        noise=Variable(z, volatile=True)
    elif type(z) == np.ndarray:
        noise=Variable(torch.from_numpy(z).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden=gan_gen(noise)
    max_indices=autoencoder.generate(hidden=fake_hidden,\
                                       maxlen=maxlen,\
                                       sample=sample)

    max_indices=max_indices.data.cpu().numpy()
    sentences=[]
    for idx in max_indices:
        # generated sentence
        words=[vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent=[]
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent=" ".join(truncated_sent)
        sentences.append(sent)

    return sentences
