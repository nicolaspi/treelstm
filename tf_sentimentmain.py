import tf_data_utils as utils

import sys
import numpy as np
import tensorflow as tf
import random

import tf_tree_lstm
import nary_tree_lstm

DIR = 'data/sst/'
GLOVE_DIR ='data/glove/'

import time

#from tf_data_utils import extract_tree_data,load_sentiment_treebank

class Config(object):

    num_emb=None

    emb_dim = 300
    hidden_dim = 150
    output_dim=None
    degree = 2
    num_labels = 3
    num_epochs = 100
    early_stopping = 2
    dropout = 0.0
    lr = 0.2
    emb_lr = 0.0
    reg=0.00001

    batch_size = 200
    #num_steps = 10
    maxseqlen = None
    maxnodesize = None
    fine_grained=False
    trainable_embeddings=False
    nonroot_labels=True
    #dependency=True not supported
    embeddings = None

def train2():
    config = Config()

    data, vocab = utils.load_sentiment_treebank(DIR, GLOVE_DIR, config.fine_grained)
   # data, vocab = utils.load_sentiment_treebank(DIR, None, config.fine_grained)
    config.embeddings = vocab.embed_matrix

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    print 'test', len(test_set)

    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)
    print 'num emb', num_emb
    print 'num labels', num_labels

    config.num_emb = num_emb
    config.output_dim = num_labels

    config.maxseqlen = utils.get_max_len_data(data)
    config.maxnodesize = utils.get_max_node_size(data)

    print config.maxnodesize, config.maxseqlen, " maxsize"
    # return
    random.seed()
    np.random.seed()

    train_set, dev_set, test_set
    train_set = utils.build_labelized_batch_trees(train_set, config.batch_size)
    dev_set = utils.build_labelized_batch_trees(dev_set, config.batch_size)
    test_set = utils.build_labelized_batch_trees(test_set, config.batch_size)

    with tf.Graph().as_default():

        #model = tf_seq_lstm.tf_seqLSTM(config)
        model = nary_tree_lstm.SoftMaxNarytreeLSTM(config, train_set + dev_set + test_set)

        init=tf.global_variables_initializer()
        best_valid_score=0.0
        best_valid_epoch=0
        dev_score=0.0
        test_score=0.0
        with tf.Session() as sess:

            sess.run(init)


            for epoch in range(config.num_epochs):
                start_time = time.time()
                print 'epoch', epoch
                avg_loss=0.0
                model.train_epoch(train_set[:],sess)
                model.test(dev_set,sess)

                print "time per epoch is {0}".format(
                    time.time()-start_time)
            test_score = evaluate(model,test_set,sess)
            print test_score,'test_score'


def train(restore=False):

    config=Config()

    data,vocab = utils.load_sentiment_treebank(DIR,GLOVE_DIR,config.fine_grained)
    config.embeddings = vocab.embed_matrix

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    print 'test', len(test_set)

    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)
    print 'num emb', num_emb
    print 'num labels', num_labels

    config.num_emb=num_emb
    config.output_dim = num_labels

    config.maxseqlen=utils.get_max_len_data(data)
    config.maxnodesize=utils.get_max_node_size(data)

    print config.maxnodesize,config.maxseqlen ," maxsize"
    #return 
    random.seed()
    np.random.seed()


    with tf.Graph().as_default():

        #model = tf_seq_lstm.tf_seqLSTM(config)
        model = tf_tree_lstm.tf_NarytreeLSTM(config)

        init=tf.global_variables_initializer()
        saver = tf.train.Saver()
        best_valid_score=0.0
        best_valid_epoch=0
        dev_score=0.0
        test_score=0.0
        with tf.Session() as sess:

            sess.run(init)


            if restore:saver.restore(sess,'./ckpt/tree_rnn_weights')
            for epoch in range(config.num_epochs):
                start_time = time.time()
                print 'epoch', epoch
                avg_loss=0.0
                avg_loss = train_epoch(model, train_set,sess)
                print 'avg loss', avg_loss

                dev_score=evaluate(model,dev_set,sess)
                print 'dev-score', dev_score

                if dev_score > best_valid_score:
                    best_valid_score=dev_score
                    best_valid_epoch=epoch
                    #saver.save(sess,'./ckpt/tree_rnn_weights')

                if epoch -best_valid_epoch > config.early_stopping:
                    break

                print "time per epochis {0}".format(
                    time.time()-start_time)
            test_score = evaluate(model,test_set,sess)
            print test_score,'test_score'

def train_epoch(model,data,sess):

    loss=model.train(data,sess)
    return loss

def evaluate(model,data,sess):
    acc=model.evaluate(data,sess)
    return acc

if __name__ == '__main__':
    if len(sys.argv) > 1:
        restore=True
    else:restore=False
    train2()

