import tf_data_utils as utils

import sys
import numpy as np
import tensorflow as tf
import random
import copy

import tf_tree_lstm
import nary_tree_lstm
from nary_tree_lstm_autoencoder import NarytreeLSTMAutoEncoder

DIR = 'data/sst/'
GLOVE_DIR ='data/glove/'

import time

#from tf_data_utils import extract_tree_data,load_sentiment_treebank

class Config(object):

    num_emb=None

    emb_dim = 300
    hidden_dim = 300
    output_dim = None
    degree = 2
    num_labels = 3
    num_epochs = 50
    nb_hidden_layers = 0

    maxseqlen = None
    maxnodesize = None
    fine_grained=False
    trainable_embeddings=True
    nonroot_labels=True

    embeddings = None


def train2():

    config = Config()
    config.batch_size = 25
    config.lr = 0.05
    config.dropout = 1.0
    config.reg = 0.000001
    config.emb_lr = 0.02
    config.pretrain = False
    config.pretrain_num_epochs = 50
    config.pretrain_batch_size = 100
    config.pretrain_lr = 0.05
    config.pretrain_dropout = 1.0
    config.pretrain_train_sub_trees = False

    import collections
    import numpy as np
    from sklearn import metrics

    def test(model, data, session):
        relevant_labels = [0, 2]
        ys_true = collections.deque([])
        ys_pred = collections.deque([])
        for batch in data:
            y_pred = model.get_output()
            y_true = batch[0].root_labels/2
            feed_dict = {model.labels: batch[0].root_labels}
            feed_dict.update(model.tree_lstm.get_feed_dict(batch[0]))
            y_pred_ = session.run([y_pred], feed_dict=feed_dict)
            y_pred_ = np.argmax(y_pred_[0][:,relevant_labels], axis=1)
            ys_true += y_true.tolist()
            ys_pred += y_pred_.tolist()
        ys_true = list(ys_true)
        ys_pred = list(ys_pred)
        score = metrics.accuracy_score(ys_true, ys_pred)
        print "Accuracy", score
        #print "Recall", metrics.recall_score(ys_true, ys_pred)
        #print "f1_score", metrics.f1_score(ys_true, ys_pred)
        print "confusion_matrix"
        print metrics.confusion_matrix(ys_true, ys_pred)
        return score

    fine_data, fine_vocab, pretrain_config = None, None, None
    pretrain_train_set, pretrain_dev_set, pretrain_test_set = None, None, None
    if config.pretrain:
        fine_data, fine_vocab = utils.load_sentiment_treebank(DIR, GLOVE_DIR, True)
        pretrain_config = copy.copy(config)
        pretrain_config.embeddings = fine_vocab.embed_matrix
        pretrain_config.batch_size = config.pretrain_batch_size
        pretrain_config.lr = config.pretrain_lr
        pretrain_config.trainable_embeddings = False
        pretrain_config.dropout = config.pretrain_dropout
        pretrain_config.train_sub_trees = config.pretrain_train_sub_trees
        pretrain_config.nb_hidden_layers = 0

        pretrain_train_set, pretrain_dev_set, pretrain_test_set = fine_data['train'], fine_data['dev'], fine_data[
            'test']

    data, vocab = utils.load_sentiment_treebank(DIR, GLOVE_DIR, False)
   # data, vocab = utils.load_sentiment_treebank(DIR, None, config.fine_grained)
    config.embeddings = vocab.embed_matrix


    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    if pretrain_test_set is not None:
        print 'pretrain train', len(pretrain_train_set)
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

    random.seed()
    np.random.seed()

    from random import shuffle
    shuffle(train_set)

    if pretrain_train_set:
        shuffle(pretrain_train_set)
        pretrain_train_set,_ = zip(*utils.build_labelized_batch_trees(pretrain_train_set, pretrain_config.batch_size))
        pretrain_dev_set,_ = zip(*utils.build_labelized_batch_trees(pretrain_dev_set, 500))
        pretrain_test_set,_ = zip(*utils.build_labelized_batch_trees(pretrain_test_set, 500))

    train_set = utils.build_labelized_batch_trees(train_set, config.batch_size)
    dev_set = utils.build_labelized_batch_trees(dev_set, 500)
    test_set = utils.build_labelized_batch_trees(test_set, 500)

    with tf.Graph().as_default():

        pretrain_model = None
        if config.pretrain:
            with tf.variable_scope("Pretrain"):
                pretrain_model = NarytreeLSTMAutoEncoder(pretrain_config)

        model = nary_tree_lstm.SoftMaxNarytreeLSTM(config, pretrain_model.tree_lstm if pretrain_model is not None else None)

        init=tf.global_variables_initializer()
        best_valid_score=0.0
        best_valid_epoch=0
        dev_score=0.0
        test_score=0.0
        with tf.Session() as sess:

            sess.run(init)
            tf.Graph.finalize(sess.graph)

            if config.pretrain:
                with tf.variable_scope("Pretrain"):
                    for epoch in range(config.pretrain_num_epochs):
                        start_time = time.time()

                        pretrain_model.train_epoch(pretrain_train_set[:], sess)
                        print "Pretraining time per epoch is {0}".format(
                            time.time() - start_time)
                        e = pretrain_model.test(pretrain_dev_set[:], sess)
                        print "dev error", e

            for epoch in range(config.num_epochs):
                start_time = time.time()
                print 'epoch', epoch
                avg_loss=0.0
                model.train_epoch(train_set[:], sess)

                print "Training time per epoch is {0}".format(
                    time.time() - start_time)

                print 'validation score'
                score = test(model,dev_set,sess)
                #print 'train score'
                #test(model, train_set[:40], sess)
                if score >= best_valid_score:
                    best_valid_score = score
                    best_valid_epoch = epoch
                    test_score = test(model,test_set,sess)
                print 'test score :', test_score, 'updated', epoch - best_valid_epoch, 'epochs ago with validation score', best_valid_score



def train(restore=False):

    config=Config()
    config.batch_size = 5
    config.lr = 0.05
    data,vocab = utils.load_sentiment_treebank(DIR,GLOVE_DIR,config.fine_grained)
    config.embeddings = vocab.embed_matrix
    config.early_stopping = 2
    config.reg = 0.0001
    config.dropout = 1.0
    config.emb_lr = 0.1

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

                print "Training time per epoch is {0}".format(
                    time.time() - start_time)


                dev_score=evaluate(model,dev_set,sess)
                print 'dev-score', dev_score

                if dev_score >= best_valid_score:
                    best_valid_score=dev_score
                    best_valid_epoch=epoch
                    #saver.save(sess,'./ckpt/tree_rnn_weights')
                    test_score = evaluate(model, test_set, sess)
                    print 'test score :', test_score, 'updated', epoch - best_valid_epoch, 'epochs ago with validation score', best_valid_score



def train_epoch(model,data,sess):

    loss=model.train(data,sess)
    return loss

def evaluate(model,data,sess):
    acc=model.evaluate(data,sess)
    return acc

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if(sys.argv[1] == "-optimized"):
            print "running optimized version"
            train2()
        else:
            print "running not optimized version"
            train()
    else:
        print "running not optimized version, run with option -optimized for the optimized one"
        train()

