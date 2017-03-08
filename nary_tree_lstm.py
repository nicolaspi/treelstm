import tensorflow as tf
from batch_tree import BatchTree, BatchTreeSample
import numpy as np
from sklearn import metrics
import collections


class BatchSample(object):
    def __init__(self, tree):
        o, m, f, p, s, c = tree.build_batch_tree_sample()
        self.prefixes = p
        self.suffixes = s
        self.observables = o
        self.masks = m
        self.flows = f
        self.children_offsets = c


class NarytreeLSTM(object):
    def __init__(self, config=None):
        self.config = config

        with tf.variable_scope("Embed", regularizer=None):

            if config.embeddings is not None:
                initializer = config.embeddings
            else:
                initializer = tf.random_uniform((config.num_emb, config.emb_dim), -0.05, 0.05)
            self.embedding = tf.Variable(initial_value=initializer, trainable=config.trainable_embeddings,
                                         dtype='float32')

        with tf.variable_scope("Node",
                               initializer=
                               # tf.ones_initializer(),
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=None
                               # tf.contrib.layers.l2_regularizer(self.config.reg)
                               ):

            def calc_wt_init(self, fan_in=300):
                eps = 1.0 / np.sqrt(fan_in)
                return eps

            self.U = tf.get_variable("U", [config.hidden_dim * config.degree , config.hidden_dim * (3 + config.degree)], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))
            self.W = tf.get_variable("W", [config.emb_dim, config.hidden_dim], initializer=tf.random_uniform_initializer(-calc_wt_init(config.emb_dim),calc_wt_init(config.emb_dim)))
            self.b = tf.get_variable("b", [config.hidden_dim*3], initializer=tf.constant_initializer(0.0))
            self.bf = tf.get_variable("bf", [config.hidden_dim], initializer=tf.constant_initializer(0.0))



            self.observables = tf.placeholder(tf.int32, shape=[None, None])
            self.scatter_indices = tf.placeholder(tf.int32, shape=[None, None])
            self.masks = tf.placeholder(tf.float32, shape=[None, None])
            self.children_offsets = tf.placeholder(tf.int32, shape=[None, None])
            self.flows = tf.placeholder(tf.int32, shape=[None])
            self.prefixes = tf.placeholder(tf.int32, shape=[None])
            self.suffixes = tf.placeholder(tf.int32, shape=[None])
            self.children_offsets = tf.placeholder(tf.int32, shape=[None])
            self.tree_size = tf.placeholder(tf.int32, shape=[])
            self.batch_size = tf.placeholder(tf.int32, shape=[])


            self.input_embed = tf.nn.embedding_lookup(self.embedding, self.observables)


            # error when one node only in the graph
            self.training_variables = [self.U, self.W, self.b, self.bf]

    def get_feed_dict(self, batch_sample):
        #print batch_sample.scatter_indices
        #print batch_sample.prefixes
        return {
        self.observables : batch_sample.observables,
        self.masks : batch_sample.masks,
        self.children_offsets : batch_sample.children_offsets,
        self.flows : batch_sample.flows,
        self.prefixes : batch_sample.prefixes,
        self.suffixes : batch_sample.suffixes,
        self.tree_size : len(batch_sample.flows),
        self.batch_size: batch_sample.flows[-1],
        self.scatter_indices : batch_sample.scatter_indices
        }

    def get_output(self):
        nodes_h, _ = self.get_outputs()
        return nodes_h

    def get_output_unscattered(self):
        _, nodes_h_unscattered = self.get_outputs()
        return nodes_h_unscattered

    def get_outputs(self):
        with tf.variable_scope("Node", reuse=True):
            W = tf.get_variable("W", [self.config.emb_dim, self.config.hidden_dim])
            U = tf.get_variable("U", [self.config.hidden_dim * self.config.degree , self.config.hidden_dim * (3 + self.config.degree)])
            b = tf.get_variable("b", [3 * self.config.hidden_dim])
            bf = tf.get_variable("bf", [self.config.hidden_dim])

            child_indices_range = tf.constant(np.arange(self.config.degree), dtype=tf.int32)
            nbf = tf.tile(bf, [self.config.degree])

            nodes_h_unscattered = tf.TensorArray(tf.float32, size=self.tree_size, clear_after_read=False)
            nodes_h = tf.TensorArray(tf.float32, size = self.tree_size, clear_after_read=False)
            nodes_c = tf.TensorArray(tf.float32, size = self.tree_size, clear_after_read=False)

            const0f = tf.constant([0], dtype=tf.float32)
            idx_var = tf.constant(0, dtype=tf.int32)

            def _recurrence(nodes_h, nodes_c, nodes_h_unscattered, idx_var):
                out_ = tf.concat([nbf, b], axis=0)
                idx_var_dim1 = tf.expand_dims(idx_var, 0)
                idxvar_0 = tf.concat([idx_var_dim1, [0]],0)

                flow = tf.to_int32(tf.gather(self.flows, idx_var))
                one_flow = tf.concat([[1], tf.expand_dims(flow, 0)], 0)
                flow_slice = tf.concat([[-1],  tf.expand_dims(flow, 0), [-1]], 0)
                flow_slice_2 = tf.concat([tf.expand_dims(flow, 0), [-1]], 0)

                # Child indices:
                children_offset = tf.to_int32(tf.gather(self.children_offsets, idx_var))
                gather_offsets = child_indices_range + idx_var - children_offset - self.config.degree

                conc_hs = tf.cond(tf.less(children_offset,0),
                                  lambda: const0f,
                                  lambda : tf.reshape(tf.concat(tf.split(tf.slice(
                                                                         nodes_h.gather(gather_offsets)
                                                                         , [0,0,0], flow_slice)
                                      , self.config.degree), axis=2), flow_slice_2)
                                  )
                conc_cs = tf.cond(tf.less(children_offset,0),
                                  lambda: const0f,
                                  lambda : tf.reshape(tf.concat(tf.split(tf.slice(
                                                                         nodes_c.gather(gather_offsets)
                                                                         , [0,0,0], flow_slice)
                                      , self.config.degree), axis=2), flow_slice_2)
                                  )

                mask = tf.slice(self.masks, idxvar_0, one_flow)
                mask_sum = tf.to_int32(tf.reduce_sum(mask))
                mask = tf.transpose(tf.to_float(tf.tile(mask,[self.config.emb_dim, 1])))

                observable = tf.squeeze(tf.slice(self.observables, idxvar_0, one_flow))

                observable = tf.Print(observable, [flow, observable, mask], None, None, 300)
                input_embed = tf.multiply(tf.nn.embedding_lookup(self.embedding, observable), mask)

                out_ += tf.cond(tf.less(children_offset,0),
                                  lambda: const0f,
                                  lambda: tf.matmul(conc_hs, U)
                                  )
                out_ += tf.cond(tf.less(0, mask_sum),
                               lambda: tf.tile(tf.matmul(input_embed, W), [1,3 + self.config.degree]),
                               lambda: const0f)


                v = tf.split(out_, 3 + self.config.degree, axis=1)
                vf = tf.sigmoid(tf.concat(v[:self.config.degree], axis=1))
                c = tf.cond(tf.less(children_offset,0),
                             lambda: tf.tanh(v[self.config.degree+2]),
                             lambda: tf.multiply(tf.sigmoid(v[self.config.degree]),tf.tanh(v[self.config.degree+2])) + tf.reduce_sum(
                                 tf.stack(tf.split(tf.multiply(vf, conc_cs), self.config.degree, axis=1)), axis=0)
                             )
                h = tf.multiply(tf.sigmoid(v[self.config.degree + 1]),tf.tanh(c))
                nodes_h_unscattered = nodes_h_unscattered.write(idx_var, h)
                #prefix = tf.to_int32(tf.expand_dims(tf.gather(self.prefixes, idx_var),0))
                scatters = tf.reshape(tf.slice(self.scatter_indices, idxvar_0, one_flow), flow_slice_2)
                #h = tf.Print(h, [v[self.config.degree], v[self.config.degree+2]], None, 300, 300)
                h = tf.scatter_nd(scatters, h, [self.batch_size, self.config.hidden_dim], name=None)
                c = tf.scatter_nd(scatters, c, [self.batch_size, self.config.hidden_dim], name=None)


                nodes_h = nodes_h.write(idx_var, h)
                nodes_c = nodes_c.write(idx_var, c)
                idx_var = tf.add(idx_var, 1)

                return nodes_h, nodes_c, nodes_h_unscattered, idx_var
            loop_cond = lambda x, y, z, id: tf.less(id, self.tree_size)

            loop_vars = [nodes_h, nodes_c, nodes_h_unscattered, idx_var]
            nodes_h, nodes_c, nodes_h_unscattered, idx_var = tf.while_loop(loop_cond, _recurrence, loop_vars,
                                                              parallel_iterations=1)
            return nodes_h.stack, nodes_h_unscattered

class SoftMaxNarytreeLSTM(object):
    def __init__(self, config, data):
        def calc_wt_init(self, fan_in=300):
            eps = 1.0 / np.sqrt(fan_in)
            return eps
        self.config = config
        with tf.variable_scope("Predictor",
                               initializer=
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=None
                               ):
            self.tree_lstm = NarytreeLSTM(config)
            self.W = tf.get_variable("W", [config.hidden_dim, config.num_labels], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))
            self.b = tf.get_variable("b", [config.num_labels], initializer=tf.constant_initializer(0.0))
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            self.training_variables = [self.W, self.b] + self.tree_lstm.training_variables
            self.optimizer = tf.train.AdagradOptimizer(self.config.lr)
            self.cross_entropy = self.get_loss()
            self.gv = self.optimizer.compute_gradients(self.cross_entropy, self.training_variables)
            self.opt = self.optimizer.apply_gradients(self.gv)
            self.output = self.get_root_output()

    def get_root_output(self):
        nodes_h = self.tree_lstm.get_output_unscattered()
        roots_h = nodes_h.read(nodes_h.size()-1)
        out = tf.matmul(roots_h, self.W) + self.b
        return out

    def get_output(self):
        return self.output

    def get_loss(self):
        h = self.tree_lstm.get_output_unscattered().concat()
        out = tf.matmul(h, self.W) + self.b
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=out))

    def get_root_loss(self):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=out))
    def train(self, batch_tree, batch_labels, session):

        feed_dict = {self.labels: batch_tree.labels}
        feed_dict.update(self.tree_lstm.get_feed_dict(batch_tree))
        ce,_ = session.run([self.cross_entropy, self.opt], feed_dict=feed_dict)
        print("cross_entropy " + str(ce))

    def train_epoch(self, data, session):
        from random import shuffle
        #shuffle(data)
        for batch in data:
            self.train(batch[0], batch[1], session)

    def test(self, data, session):
        ys_true = collections.deque([])
        ys_pred = collections.deque([])
        for batch in data:
            y_pred = tf.argmax(self.get_output(), 1)
            y_true = self.labels
            feed_dict = {self.labels: batch[0].root_labels}
            feed_dict.update(self.tree_lstm.get_feed_dict(batch[0]))
            y_pred, y_true = session.run([y_pred, y_true], feed_dict=feed_dict)
            ys_true += y_true.tolist()
            ys_pred += y_pred.tolist()
            print("computing... ")
        ys_true = list(ys_true)
        ys_pred = list(ys_pred)
        print "Accuracy", metrics.accuracy_score(ys_true, ys_pred)
        #print "Recall", metrics.recall_score(ys_true, ys_pred)
        #print "f1_score", metrics.f1_score(ys_true, ys_pred)
        print "confusion_matrix"
        print metrics.confusion_matrix(ys_true, ys_pred)
        # fpr, tpr, tresholds = sk.metrics.roc_curve(ys_true, ys_pred)


def test_lstm_model():
    class Config(object):
        num_emb = 10
        emb_dim = 3
        hidden_dim = 4
        output_dim = None
        degree = 2
        num_epochs = 3
        early_stopping = 2
        dropout = 0.5
        lr = 1.0
        emb_lr = 0.1
        reg = 0.0001
        fine_grained = False
        trainable_embeddings = True
        embeddings = None
        batch_size=7

    tree = BatchTree.empty_tree()
    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(-1, 1, 0)
    tree.root.expand_or_add_child(1, 1, 1)
    tree.root.children[0].expand_or_add_child(1, 0, 0)
    tree.root.children[0].expand_or_add_child(1, 0,  1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(2, 1, 0)
    tree.root.expand_or_add_child(2, 1, 1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(-1, 1, 0)
    tree.root.expand_or_add_child(3, 1, 1)
    tree.root.children[0].expand_or_add_child(3, 0, 0)
    tree.root.children[0].expand_or_add_child(3, 0, 1)

    sample = BatchTreeSample(tree)

    model = NarytreeLSTM(Config())
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    v = sess.run(model.get_output_debug(),feed_dict=model.get_feed_dict(sample))
    print(v)
    return 0


def test_softmax_model():
    class Config(object):
        num_emb = 10
        emb_dim = 3
        hidden_dim = 8
        output_dim = None
        degree = 2
        num_epochs = 3
        early_stopping = 2
        dropout = 0.5
        lr = 1.0
        emb_lr = 0.1
        reg = 0.0001
        fine_grained = False
        trainable_embeddings = True
        num_labels = 2
        embeddings = None

    tree = BatchTree.empty_tree()
    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(-1, 1, 0)
    tree.root.expand_or_add_child(1, 1, 1)
    tree.root.children[0].expand_or_add_child(1, 0, 0)
    tree.root.children[0].expand_or_add_child(1, 0, 1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(2, 1, 0)
    tree.root.expand_or_add_child(2, 1, 1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(-1, 1, 0)
    tree.root.expand_or_add_child(3, 1, 1)
    tree.root.children[0].expand_or_add_child(3, 0, 0)
    tree.root.children[0].expand_or_add_child(3, 0, 1)

    # tree.root.add_sample(1)
    # labels = np.array([[0, 1]])
    batch_sample = BatchTreeSample(tree)

    labels = np.array([0,1,0,1,0])

    model = SoftMaxNarytreeLSTM(Config(), [tree])
    sess = tf.InteractiveSession()
    summarywriter = tf.summary.FileWriter('/tmp/tensortest', graph=sess.graph)
    tf.global_variables_initializer().run()
    sample = [(batch_sample, labels)]
    for i in range(1000):
        model.train(batch_sample, labels, sess)
        model.test(sample, sess)
    return 0


if __name__ == '__main__':
    test_softmax_model()
    #test_lstm_model()