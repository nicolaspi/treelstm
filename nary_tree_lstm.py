import tensorflow as tf
from batch_tree import BatchTree, BatchTreeSample
import numpy as np
from sklearn import metrics
import collections


class NarytreeLSTM(object):
    def __init__(self, config=None):
        self.config = config

        with tf.variable_scope("Embed", regularizer=None):

            if config.embeddings is not None:
                initializer = config.embeddings
            else:
                initializer = tf.random_uniform((config.num_emb, config.emb_dim))
            self.embedding = tf.Variable(initial_value=initializer, trainable=config.trainable_embeddings,
                                         dtype='float32')

        with tf.variable_scope("Node",
                               initializer=
                               # tf.ones_initializer(),
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)
                               ):

            def calc_wt_init(self, fan_in=300):
                eps = 1.0 / np.sqrt(fan_in)
                return eps

            self.U = tf.get_variable("U", [config.hidden_dim * config.degree , config.hidden_dim * (3 + config.degree)], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))
            self.W = tf.get_variable("W", [config.emb_dim, config.hidden_dim], initializer=tf.random_uniform_initializer(-calc_wt_init(config.emb_dim),calc_wt_init(config.emb_dim)))
            self.b = tf.get_variable("b", [config.hidden_dim*3], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))#, regularizer=tf.contrib.layers.l2_regularizer(0.0))
            self.bf = tf.get_variable("bf", [config.hidden_dim], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))#, regularizer=tf.contrib.layers.l2_regularizer(0.0))



            self.observables = tf.placeholder(tf.int32, shape=[None])
            self.flows = tf.placeholder(tf.int32, shape=[None])
            self.input_scatter = tf.placeholder(tf.int32, shape=[None])
            self.observables_indices = tf.placeholder(tf.int32, shape=[None])
            self.out_indices = tf.placeholder(tf.int32, shape=[None])
            self.scatter_out = tf.placeholder(tf.int32, shape=[None])
            self.scatter_in = tf.placeholder(tf.int32, shape=[None])
            self.scatter_in_indices = tf.placeholder(tf.int32, shape=[None])
            self.batch_size = tf.placeholder(tf.int32, shape=[])
            self.tree_height = tf.placeholder(tf.int32, shape=[])
            self.dropout = tf.placeholder(tf.float32, shape=[])
            self.child_scatter_indices = tf.placeholder(tf.int32, shape=[None])
            self.nodes_count = tf.placeholder(tf.int32, shape=[None])
            self.input_embed = tf.nn.embedding_lookup(self.embedding, self.observables)
            self.nodes_count_per_indice = tf.placeholder(tf.float32, shape=[None])

            self.training_variables = [self.U, self.W, self.b, self.bf]
            if config.trainable_embeddings:
                self.training_variables.append( self.embedding)

    def get_feed_dict(self, batch_sample, dropout = 1.0):
        #print batch_sample.scatter_in
        #print batch_sample.scatter_in_indices
        #print batch_sample.nodes_count_per_indice, "nodes_count_per_indice"
        return {
        self.observables : batch_sample.observables,
        self.flows : batch_sample.flows,
        self.input_scatter : batch_sample.input_scatter,
        self.observables_indices : batch_sample.observables_indices,
        self.out_indices: batch_sample.out_indices,
        self.tree_height: len(batch_sample.out_indices)-1,
        self.batch_size: batch_sample.flows[-1],#batch_sample.out_indices[-1] - batch_sample.out_indices[-2],
        self.scatter_out: batch_sample.scatter_out,
        self.scatter_in: batch_sample.scatter_in,
        self.scatter_in_indices: batch_sample.scatter_in_indices,
        self.child_scatter_indices: batch_sample.child_scatter_indices,
        self.nodes_count: batch_sample.nodes_count,
        self.dropout : dropout,
        self.nodes_count_per_indice : batch_sample.nodes_count_per_indice
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

            nbf = tf.tile(bf, [self.config.degree])

            nodes_h_scattered = tf.TensorArray(tf.float32, size=self.tree_height, clear_after_read=False)
            nodes_h = tf.TensorArray(tf.float32, size = self.tree_height, clear_after_read=False)
            nodes_c = tf.TensorArray(tf.float32, size = self.tree_height, clear_after_read=False)

            const0f = tf.constant([0], dtype=tf.float32)
            idx_var = tf.constant(0, dtype=tf.int32)
            hidden_shape = tf.constant([-1, self.config.hidden_dim * self.config.degree], dtype=tf.int32)
            out_shape = tf.stack([-1,self.batch_size, self.config.hidden_dim], 0)

            def _recurrence(nodes_h, nodes_c, nodes_h_scattered, idx_var):
                out_ = tf.concat([nbf, b], axis=0)
                idx_var_dim1 = tf.expand_dims(idx_var, 0)
                prev_idx_var_dim1 = tf.expand_dims(idx_var-1, 0)

                observables_indice_begin, observables_indice_end = tf.split(tf.slice(self.observables_indices, idx_var_dim1, [2]), 2)
                observables_size = observables_indice_end - observables_indice_begin
                out_indice_begin, out_indice_end = tf.split(
                    tf.slice(self.out_indices, idx_var_dim1, [2]), 2)
                out_size = out_indice_end - out_indice_begin
                flow = tf.slice(self.flows, idx_var_dim1, [1])
                w_scatter_shape = tf.concat([flow, [self.config.hidden_dim]], axis=0)
                u_scatter_shape = tf.concat([flow, [self.config.hidden_dim * (3 + self.config.degree)]], axis=0)
                c_scatter_shape = tf.concat([flow, [self.config.hidden_dim * self.config.degree]],axis=0)


                def compute_indices():
                    prev_level_indice_begin, prev_level_indice_end = tf.split(
                        tf.slice(self.out_indices, prev_idx_var_dim1, [2]), 2)
                    prev_level_indice_size = prev_level_indice_end - prev_level_indice_begin
                    scatter_indice_begin, scatter_indice_end = tf.split(
                        tf.slice(self.scatter_in_indices, prev_idx_var_dim1, [2]), 2)
                    scatter_indice_size = scatter_indice_end - scatter_indice_begin
                    child_scatters = tf.slice(self.child_scatter_indices, prev_level_indice_begin, prev_level_indice_size)
                    child_scatters = tf.reshape(child_scatters, tf.concat([prev_level_indice_size, [-1]], 0))
                    return scatter_indice_begin, scatter_indice_size, child_scatters

                def hs_compute():
                    scatter_indice_begin, scatter_indice_size, child_scatters = compute_indices()

                    h = nodes_h.read(idx_var - 1)
                    hs = tf.scatter_nd(child_scatters,h,tf.shape(h), name=None)
                    hs = tf.reshape(hs, hidden_shape)
                    out = tf.matmul(hs, U)

                    scatters_in = tf.slice(self.scatter_in, scatter_indice_begin, scatter_indice_size)
                    scatters_in = tf.reshape(scatters_in, tf.concat([scatter_indice_size, [-1]], 0))
                    #scatters_in = tf.Print(scatters_in, [idx_var, tf.shape(hs), u_scatter_shape, scatters_in], "hs", 300, 300)
                    out = tf.scatter_nd(scatters_in, out, u_scatter_shape, name=None)
                    return out

                def cs_compute():
                    scatter_indice_begin, scatter_indice_size, child_scatters = compute_indices()

                    c = nodes_c.read(idx_var - 1)
                    cs = tf.scatter_nd(child_scatters, c, tf.shape(c), name=None)
                    cs = tf.reshape(cs, hidden_shape)

                    scatters_in = tf.slice(self.scatter_in, scatter_indice_begin, scatter_indice_size)
                    scatters_in = tf.reshape(scatters_in, tf.concat([scatter_indice_size, [-1]], 0))
                    #scatters_in = tf.Print(scatters_in, [idx_var, tf.shape(cs), c_scatter_shape, scatters_in], "cs",
                    #                       300, 300)
                    cs = tf.scatter_nd(scatters_in, cs, c_scatter_shape, name=None)
                    return cs

                out_ += tf.cond(tf.less(0,idx_var),
                             lambda: hs_compute(),
                             lambda: const0f
                             )
                cs = tf.cond(tf.less(0,idx_var),
                             lambda: cs_compute(),
                             lambda: const0f
                             )


                observable = tf.squeeze(tf.slice(self.observables, observables_indice_begin, observables_size))


                input_embed = tf.reshape(tf.nn.embedding_lookup(self.embedding, observable),[-1,self.config.emb_dim])

                def compute_input():
                    out = tf.matmul(input_embed, W)

                    input_scatter = tf.slice(self.input_scatter, observables_indice_begin, observables_size)
                    input_scatter = tf.reshape(input_scatter, tf.concat([observables_size, [-1]], 0))
                    out = tf.scatter_nd(input_scatter, out, w_scatter_shape, name=None)
                    out = tf.tile(out, [1, 3 + self.config.degree])
                    return out

                out_ += tf.cond(tf.less(0, tf.squeeze(observables_size)),
                               lambda: compute_input(),
                               lambda: const0f)

                v = tf.split(out_, 3 + self.config.degree, axis=1)
                vf = tf.sigmoid(tf.concat(v[:self.config.degree], axis=1))

                c = tf.cond(tf.less(0,idx_var),
                                         lambda: tf.multiply(tf.sigmoid(v[self.config.degree]),tf.tanh(v[self.config.degree+2])) + tf.reduce_sum(
                                             tf.stack(tf.split(tf.multiply(vf, cs), self.config.degree, axis=1)), axis=0),
                                         lambda: tf.multiply(tf.sigmoid(v[self.config.degree]),tf.tanh(v[self.config.degree+2]))
                                         )

                h = tf.multiply(tf.sigmoid(v[self.config.degree + 1]),tf.tanh(c))
                h = tf.nn.dropout(h, self.dropout)
                slice = tf.slice(self.embedding, [32,0], [1,10])
                #h = tf.Print(h, [slice], "the DOT embed", 300, 300)
                nodes_h = nodes_h.write(idx_var, h)
                nodes_c = nodes_c.write(idx_var, c)

                scatters = tf.reshape(tf.slice(self.scatter_out, out_indice_begin, out_size), tf.concat([out_size, [-1]], 0))

                node_count = tf.slice(self.nodes_count, idx_var_dim1, [1])
                scatter_out_lenght = node_count * self.batch_size
                scatter_out_shape = tf.stack([tf.squeeze(scatter_out_lenght), self.config.hidden_dim], 0)
                h = tf.reshape(tf.scatter_nd(scatters, h, scatter_out_shape, name=None), out_shape)
                nodes_h_scattered = nodes_h_scattered.write(idx_var, h)
                idx_var = tf.add(idx_var, 1)

                return nodes_h, nodes_c, nodes_h_scattered, idx_var
            loop_cond = lambda x, y, z, id: tf.less(id, self.tree_height)

            loop_vars = [nodes_h, nodes_c, nodes_h_scattered, idx_var]
            nodes_h, nodes_c, nodes_h_scattered, idx_var = tf.while_loop(loop_cond, _recurrence, loop_vars,
                                                              parallel_iterations=1)
            return nodes_h_scattered.concat(), nodes_h

class SoftMaxNarytreeLSTM(object):

    def __init__(self, config, data):
        def calc_wt_init(self, fan_in=300):
            eps = 1.0 / np.sqrt(fan_in)
            return eps
        self.config = config
        with tf.variable_scope("Predictor",
                               initializer=
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)
                               ):
            self.tree_lstm = NarytreeLSTM(config)
            self.W = tf.get_variable("W", [config.hidden_dim, config.num_labels], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))
            self.b = tf.get_variable("b", [config.num_labels], initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),calc_wt_init(config.hidden_dim)))#, regularizer=tf.contrib.layers.l2_regularizer(0.0))
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            self.training_variables = [self.W, self.b] + self.tree_lstm.training_variables
            self.optimizer = tf.train.AdagradOptimizer(self.config.lr)
            self.embed_optimizer = tf.train.AdagradOptimizer(self.config.emb_lr)
            self.loss = self.get_loss()
            #self.gv = self.optimizer.compute_gradients(self.loss, self.training_variables)
            self.gv = zip(tf.gradients(self.loss, self.training_variables),self.training_variables)
            if config.trainable_embeddings:
                self.opt = self.optimizer.apply_gradients(self.gv[:-1])
                self.embed_opt = self.embed_optimizer.apply_gradients(self.gv[-1:])
            else :
                self.opt = self.optimizer.apply_gradients(self.gv)
                self.embed_opt = tf.no_op()

            self.output = self.get_root_output()

    def get_root_output(self):
        nodes_h = self.tree_lstm.get_output_unscattered()
        roots_h = nodes_h.read(nodes_h.size()-1)
        out = tf.matmul(roots_h, self.W) + self.b
        return out

    def get_output(self):
        return self.output

    def get_loss(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart = tf.add_n(reg_losses)
        #regpart = tf.Print(regpart, [regpart])
        h = self.tree_lstm.get_output_unscattered().concat()
        out = tf.matmul(h, self.W) + self.b
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=out)
        return tf.reduce_sum(tf.divide(loss, tf.to_float(self.tree_lstm.batch_size))) + regpart

    def train(self, batch_tree, batch_labels, session):

        feed_dict = {self.labels: batch_tree.labels}
        feed_dict.update(self.tree_lstm.get_feed_dict(batch_tree, self.config.dropout))
        ce,_,_ = session.run([self.loss, self.opt, self.embed_opt], feed_dict=feed_dict)
        #v = session.run([self.output], feed_dict=feed_dict)
        #print("cross_entropy " + str(ce))
        return ce
        #print v

    def train_epoch(self, data, session):
        #from random import shuffle
        #shuffle(data)
        total_error = 0.0
        for batch in data:
            total_error += self.train(batch[0], batch[1], session)
        print 'average error :', total_error/len(data)

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
        ys_true = list(ys_true)
        ys_pred = list(ys_pred)
        score = metrics.accuracy_score(ys_true, ys_pred)
        print "Accuracy", score
        #print "Recall", metrics.recall_score(ys_true, ys_pred)
        #print "f1_score", metrics.f1_score(ys_true, ys_pred)
        print "confusion_matrix"
        print metrics.confusion_matrix(ys_true, ys_pred)
        return score


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
        trainable_embeddings = False
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
    v = sess.run(model.get_output(),feed_dict=model.get_feed_dict(sample))
    print(v)
    return 0


def test_softmax_model():
    class Config(object):
        num_emb = 10
        emb_dim = 3
        hidden_dim = 1
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

    tree.root.add_sample(7, 1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(-1, 1, 0)
    tree.root.expand_or_add_child(-1, 1, 1)
    tree.root.children[0].expand_or_add_child(3, 0, 0)
    tree.root.children[0].expand_or_add_child(3, 0, 1)
    tree.root.children[1].expand_or_add_child(3, 0, 0)
    tree.root.children[1].expand_or_add_child(3, 0, 1)

    # tree.root.add_sample(1)
    # labels = np.array([[0, 1]])
    batch_sample = BatchTreeSample(tree)

    observables, flows, mask, scatter_out, scatter_in, scatter_in_indices, labels, observables_indices, out_indices, childs_transpose_scatter, nodes_count, nodes_count_per_indice = tree.build_batch_tree_sample()
    print observables, "observables"
    print observables_indices, "observables_indices"
    print flows, "flows"
    print mask, "input_scatter"
    print scatter_out, "scatter_out"
    print scatter_in, "scatter_in"
    print scatter_in_indices, "scatter_in_indices"
    print labels, "labels"
    print out_indices, "out_indices"
    print childs_transpose_scatter, "childs_transpose_scatter"
    print nodes_count, "nodes_count"
    print nodes_count_per_indice, "nodes_count_per_indice"

    labels = np.array([0,1,0,1,0])

    model = SoftMaxNarytreeLSTM(Config(), [tree])
    sess = tf.InteractiveSession()
    summarywriter = tf.summary.FileWriter('/tmp/tensortest', graph=sess.graph)
    tf.global_variables_initializer().run()
    sample = [(batch_sample, labels)]
    for i in range(100):
        model.train(batch_sample, labels, sess)
        model.test(sample, sess)
    return 0


if __name__ == '__main__':
    test_softmax_model()
    #test_lstm_model()