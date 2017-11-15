import sys
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, Function, Variable
from chainer import Link, Chain
import vsmlib


class CNN(Chain):
    
    def __init__(self, n_vocab, input_channel, output_channel, n_label, embed_dim, position_dims=50, freeze=True, train=True):
        super(CNN, self).__init__(
            embed=L.EmbedID(n_vocab, embed_dim), 
            dist1=L.EmbedID(n_vocab, position_dims),
            dist2=L.EmbedID(n_vocab, position_dims),
            conv1=L.Convolution2D(
                input_channel, output_channel, (3, embed_dim + 2*position_dims)),
            l1=L.Linear(output_channel, n_label)
        )
        self.train = train
        self.freeze = freeze

    def load_embeddings(self, emb_path, vocab):
        assert self.embed != None
        sys.stderr.write("loading word embedddings...")
        m = vsmlib.model.load_from_dir(emb_path)

        emb_ids = m.vocabulary.dic_words_ids.keys()
        dataset_ids = vocab.keys()

        for word in vocab.keys():
            if m.has_word(word.lower()):
                self.embed.W.data[vocab[word]] = m.get_row(word.lower())
        #print(self.embed.W.data.shape)
        sys.stderr.write("done\n")

    def __call__(self, token, pos1, pos2):
        
        if self.freeze:
            self.embed.disable_update()
        xs1 = self.embed(token)
        xs2 = self.dist1(pos1)
        xs3 = self.dist2(pos2)
        xs = F.concat([xs1, xs2, xs3],axis=-1) 
        batchsize, height, width = xs.shape
        xs = F.reshape(xs, (batchsize, 1, height, width))
        conv1_xs = self.conv1(xs)
        h1 = F.max_pooling_2d(F.tanh(conv1_xs), conv1_xs.shape[2])
        y = self.l1(F.dropout(F.tanh(h1)))
        return y