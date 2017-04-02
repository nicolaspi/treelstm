import tensorflow as tf
from batch_tree import BatchTree, BatchTreeSample
from nary_tree_lstm import NarytreeLSTM
import numpy as np
import time
from itertools import izip


class NarytreeLSTMAutoEncoder(object):
    def __init__(self, config=None):
        def calc_wt_init(self, fan_in=300):
            eps = 1.0 / np.sqrt(fan_in)
            return eps
        self.config = config
        self.tree_lstm = NarytreeLSTM(config)

        with tf.variable_scope("Decoder",
                               initializer=
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)
                               ):
            #self.tree_lstm = NarytreeLSTM(config)
            self.U = tf.get_variable("U", [config.hidden_dim, config.hidden_dim * (2 + 2*config.degree)],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                               calc_wt_init(config.hidden_dim)))
            self.hU = [tf.get_variable("hU"+str(i), [config.hidden_dim * (2 + 2*config.degree), config.hidden_dim * (2 + 2*config.degree)],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim * (2 + 2*config.degree)),
                                                                               calc_wt_init(config.hidden_dim * (2 + 2*config.degree))))
                                     for i in range(config.nb_hidden_layers)]

            self.W = tf.get_variable("W", [config.hidden_dim, config.emb_dim],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.emb_dim),
                                                                               calc_wt_init(config.emb_dim)))
            self.hW = [tf.get_variable("hW" + str(i), [config.hidden_dim, config.hidden_dim],
                                       initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                                 calc_wt_init(config.hidden_dim)))
                       for i in range(config.nb_hidden_layers)]

            self.b = tf.get_variable("b", [config.hidden_dim * (2 + 2*config.degree)],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                               calc_wt_init(config.hidden_dim)))
            self.hb = [tf.get_variable("hb" + str(i), [config.hidden_dim * (2 + 2*config.degree)],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                               calc_wt_init(config.hidden_dim)))
                       for i in range(config.nb_hidden_layers)]

            self.b_out = tf.get_variable("b_out", [config.emb_dim],
                                     initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                               calc_wt_init(
                                                                                   config.hidden_dim)))
            self.hb_out = [tf.get_variable("b_out" + str(i), [config.hidden_dim],
                                         initializer=tf.random_uniform_initializer(-calc_wt_init(config.hidden_dim),
                                                                                   calc_wt_init(
                                                                                       config.hidden_dim)))
                           for i in range(config.nb_hidden_layers)]
            # , regularizer=tf.contrib.layers.l2_regularizer(0.0))
            self.const0f = tf.constant([0], dtype=tf.float32)
            self.start_height = tf.placeholder(tf.int32, shape=[])


        self.training_variables = [self.U, self.W, self.b, self.b_out] + self.hW + self.hU + self.hb + self.hb_out + self.tree_lstm.training_variables
        #self.optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        self.optimizer = tf.train.AdagradOptimizer(self.config.lr)
        self.loss = self.get_loss()
        self.accuracy = self.get_max_prob_accuracy()
        self.opt = self.optimizer.minimize(self.loss, var_list=self.training_variables)
        self.gv = self.optimizer.compute_gradients(self.loss, var_list=[self.tree_lstm.b])
        self.saver = tf.train.Saver(self.training_variables)

    def save(self, sess, save_path):
        self.saver.save(sess, save_path)

    def restore(self, sess, save_path):
        self.saver.restore(sess, save_path)

    def get_max_prob_accuracy(self):
        self.encoder_hiddens = self.tree_lstm.get_output_unscattered()
        range = tf.range(self.start_height, self.tree_lstm.tree_height)
        # range = tf.Print(range, [range])
        def foldfn(accu, height):
            pred, _, indices = self.get_outputs(height)
            # target = tf.Print(target, [target], "target", None, 100)
            # target = tf.Print(target, [pred], "pred", None, 100)

            normalized_pred = tf.nn.l2_normalize(pred, 1)
            normalized_embed = tf.nn.l2_normalize(self.tree_lstm.embedding,1)
            cosinus = tf.matmul(normalized_pred, normalized_embed, False, True)
            pred_indices = tf.to_int32(tf.argmax(cosinus, axis=1))
            accu += tf.reduce_mean(tf.to_float(tf.equal(indices, pred_indices)))
            return accu

        accu = tf.divide(tf.foldl(foldfn, range, initializer=self.const0f), tf.to_float(self.tree_lstm.tree_height - self.start_height))
        return accu

    def get_loss(self):
        self.encoder_hiddens = self.tree_lstm.get_output_unscattered()
        range = tf.range(self.start_height, self.tree_lstm.tree_height)
        #range = tf.Print(range, [range])
        def foldfn(loss, height):
            pred, target, _ = self.get_outputs(height)
            #target = tf.Print(target, [target], "target", None, 100)
            #target = tf.Print(target, [pred], "pred", None, 100)
            loss += tf.reduce_sum(tf.square(target-pred))
            return loss

        #target, pred = self.get_outputs(2)
        #loss = tf.reduce_sum(tf.square(target - pred))
        loss = tf.foldl(foldfn, range, initializer=self.const0f)
        return tf.divide(loss, tf.to_float(self.tree_lstm.batch_size))
    def get_encoder_output(self, height):
        nodes_h = self.tree_lstm.get_output_unscattered()
        roots_h = nodes_h.read(nodes_h.size()-1)
        return roots_h

    def get_feed_dict(self, batch_sample, testing = False, dropout = 1.0):
        feed_dict = {self.start_height: len(batch_sample.out_indices)-2 if testing else 0 if self.config.train_sub_trees else len(batch_sample.out_indices)-2}
        feed_dict.update(self.tree_lstm.get_feed_dict(batch_sample, dropout))
        return feed_dict

    def get_outputs(self, height):
        with tf.variable_scope("Decoder", reuse=True):
            W = tf.get_variable("W", [self.config.hidden_dim, self.config.emb_dim])
            U = tf.get_variable("U", [self.config.hidden_dim , self.config.hidden_dim * (2 + 2*self.config.degree)])
            b = tf.get_variable("b", [self.config.hidden_dim * (2 + 2*self.config.degree)])
            b_out = tf.get_variable("b_out", [self.config.emb_dim])

            pred_outs = tf.TensorArray(tf.float32, size=height+1, clear_after_read=False)
            target_outs = tf.TensorArray(tf.float32, size=height+1, clear_after_read=False)
            target_indices_outs = tf.TensorArray(tf.int32, size=height + 1, clear_after_read=False)
            nodes_h = tf.TensorArray(tf.float32, size = height+2, clear_after_read=False)
            nodes_c = tf.TensorArray(tf.float32, size = height+2, clear_after_read=False)

            nodes_h = nodes_h.write(height+1, self.encoder_hiddens.read(height))
            #nodes_h = nodes_h.write(self.tree_lstm.tree_height, tf.ones([self.tree_lstm.batch_size, self.config.hidden_dim]))
            #nodes_c = nodes_c.write(self.tree_lstm.tree_height, tf.zeros([self.tree_lstm.batch_size, self.config.hidden_dim]))

            o_begin = tf.constant([0, 2 * self.config.hidden_dim], dtype=tf.int32)
            u_begin = tf.constant([0, (2 + self.config.degree) * self.config.hidden_dim], dtype=tf.int32)
            child_size_0 = tf.constant([self.config.degree * self.config.hidden_dim], dtype = tf.int32)

            idx_var = height
            def _recurrence(nodes_h, nodes_c, pred_outs, target_outs, target_indices_outs, idx_var):

                idx_var_dim1 = tf.expand_dims(idx_var, 0)

                #compute observables stuff
                h = nodes_h.read(idx_var+1)
                observables_indice_begin, observables_indice_end = tf.split(
                    tf.slice(self.tree_lstm.observables_indices, idx_var_dim1, [2]), 2)
                observables_size = observables_indice_end - observables_indice_begin
                observable = tf.slice(self.tree_lstm.observables, observables_indice_begin, observables_size)
                target_indices_outs = target_indices_outs.write(idx_var, observable)
                observable = tf.squeeze(observable)
                input_embed = tf.reshape(tf.nn.embedding_lookup(self.tree_lstm.embedding, observable), [-1, self.config.emb_dim])
                input_gather = tf.slice(self.tree_lstm.input_scatter, observables_indice_begin, observables_size)
                target_out = input_embed
                pred_out = tf.gather(h, input_gather)
                for hW, hb_out in izip(self.hW, self.hb_out):
                    pred_out = tf.nn.dropout(tf.nn.relu(tf.matmul(pred_out, hW) + hb_out), self.tree_lstm.dropout)
                pred_out = tf.matmul(pred_out, W) + b_out
                pred_outs = pred_outs.write(idx_var, pred_out)
                target_outs = target_outs.write(idx_var, target_out)
                #compute hidden stuff
                def compute_next_layer():
                    out_ = b

                    h = nodes_h.read(idx_var+1)
                    next_idx_var_dim1 = tf.expand_dims(idx_var - 1, 0)
                    scatter_indice_begin, scatter_indice_end = tf.split(
                        tf.slice(self.tree_lstm.scatter_in_indices, next_idx_var_dim1, [2]), 2)
                    scatter_indice_size = scatter_indice_end - scatter_indice_begin
                    gather_in = tf.slice(self.tree_lstm.scatter_in, scatter_indice_begin, scatter_indice_size)

                    h = tf.gather(h, gather_in)

                    out_ += tf.matmul(h, U)

                    for hU, hb in izip(self.hU, self.hb):
                        out_ = tf.matmul(tf.nn.dropout(tf.nn.relu(out_), self.tree_lstm.dropout), hU) + hb

                    v = tf.split(out_, 2 + 2*self.config.degree, axis=1)

                    def compute_cf():
                        c = nodes_c.read(idx_var + 1)
                        c = tf.gather(c, gather_in)
                        cf = tf.multiply(tf.sigmoid(v[1]), c)
                        cf = tf.tile(cf, [1, self.config.degree])
                        return cf

                    cf = tf.cond(tf.less(idx_var, height),
                                 lambda: compute_cf(),
                                 lambda: self.const0f
                                 )
                    # c stuff
                    child_size = tf.concat([scatter_indice_size, child_size_0], axis = 0)
                    u = tf.slice(out_, u_begin, child_size)
                    i = tf.sigmoid(v[0])
                    it = tf.tile(i, [1, self.config.degree])
                    ck = tf.multiply(it, tf.tanh(u)) + cf

                    #h stuff
                    o = tf.slice(out_, o_begin, child_size)
                    hk = tf.nn.dropout(tf.multiply(tf.sigmoid(o), tf.tanh(ck)), self.tree_lstm.dropout)

                    # gather ck and hk
                    hk = tf.reshape(hk, [-1, self.config.hidden_dim])
                    ck = tf.reshape(ck, [-1, self.config.hidden_dim])


                    next_level_indice_begin, next_level_indice_end = tf.split(
                        tf.slice(self.tree_lstm.out_indices, next_idx_var_dim1, [2]), 2)
                    next_level_indice_size = next_level_indice_end - next_level_indice_begin
                    child_gather = tf.slice(self.tree_lstm.child_scatter_indices, next_level_indice_begin, next_level_indice_size)

                    hk = tf.gather(hk, child_gather)
                    ck = tf.gather(ck, child_gather)
                    return [hk, ck]

                output = tf.cond(tf.less(0,idx_var),
                                lambda: compute_next_layer(),
                                lambda: [self.const0f, self.const0f]
                                )
                nodes_h = nodes_h.write(idx_var, output[0])
                nodes_c = nodes_c.write(idx_var, output[1])

                idx_var = tf.add(idx_var, -1)
                return nodes_h, nodes_c, pred_outs, target_outs, target_indices_outs, idx_var
            loop_cond = lambda x, y, z, a, b, id: tf.less(-1, id)

            loop_vars = [nodes_h, nodes_c, pred_outs, target_outs, target_indices_outs, idx_var]

            nodes_h, nodes_c, pred_outs, target_outs, target_indices_outs, idx_var = tf.while_loop(loop_cond, _recurrence, loop_vars,
                                                              parallel_iterations=1)

            nodes_h.close()
            nodes_c.close()
            target = target_outs.concat()
            pred = pred_outs.concat()
            target_indices = target_indices_outs.concat()
            target_outs.close()
            pred_outs.close()
            target_indices_outs.close()

            return pred,target,target_indices

    def train(self, batch_tree, session):
        feed_dict = self.get_feed_dict(batch_tree, False, self.config.dropout)
        e,  _ = session.run([self.loss, self.opt], feed_dict=feed_dict)
        #import random
        #if random.random() < 0.1:
        #    print "grad", gv
        # v = session.run([self.output], feed_dict=feed_dict)
        return e

    def train_epoch(self, data, session):
        total_error = 0.0
        for batch in data:
            total_error += self.train(batch, session)
        print 'average error :', total_error/len(data)

    def test(self, data, session):
        total_error = 0.0
        for batch in data:
            feed_dict = self.get_feed_dict(batch, True, self.config.dropout)
            total_error += session.run([self.loss], feed_dict=feed_dict)[0]
        return total_error/len(data)

    def test_accuracy(self, data, session):
        total_acc = 0.0
        for batch in data:
            feed_dict = self.get_feed_dict(batch, True, self.config.dropout)
            total_acc += session.run([self.accuracy], feed_dict=feed_dict)[0]
        return total_acc / len(data)

def test_model():
    class Config(object):
        num_emb = 10
        emb_dim = 5
        hidden_dim = 10
        output_dim = None
        degree = 2
        num_epochs = 3
        early_stopping = 2
        dropout = 0.5
        lr = 0.1
        emb_lr = 0.1
        reg = 0.0001
        fine_grained = False
        trainable_embeddings = False
        num_labels = 2
        embeddings = None
        nb_hidden_layers = 0
        train_sub_trees = False

    tree = BatchTree.empty_tree()

    tree.root.add_sample(7, 1)


    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(2, 1, 0)
    tree.root.expand_or_add_child(2, 1, 1)
    #
    #
    # tree.root.add_sample(7, 1)
    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(3, 1, 0)
    tree.root.expand_or_add_child(-1, 1, 1)
     #tree.root.children[0].expand_or_add_child(3, 0, 0)
     #tree.root.children[0].expand_or_add_child(3, 0, 1)
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

    model = NarytreeLSTMAutoEncoder(Config())
    sess = tf.InteractiveSession()

    summarywriter = tf.summary.FileWriter('/tmp/tensortest', graph=sess.graph)
    tf.global_variables_initializer().run()

    tf.Graph.finalize(sess.graph)
    sample = [(batch_sample, labels)]

    for i in range(100):
        start_time = time.time()
        model.train_epoch([batch_sample], sess)
        print "Training time per epoch is {0}".format(
            time.time() - start_time)
        e = model.test_accuracy([batch_sample], sess)
        print "test error", e

    print "saving"
    model.save(sess, "./test/test_save.save")
    print "restoring"
    model.restore(sess, "./test/test_save.save")
    for i in range(100):
        start_time = time.time()
        model.train_epoch([batch_sample], sess)
        print "Training time per epoch is {0}".format(
            time.time() - start_time)
        e = model.test_accuracy([batch_sample], sess)
        print "test error", e

    return 0


if __name__ == '__main__':
    test_model()
