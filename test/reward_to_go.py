from unittest import TestCase
from tf_tree_lstm import compute_tree_reward_to_go
import numpy as np
import tensorflow as tf

class RewardToGo(TestCase):

    def setUp(self):
        self.reward_by_nodes = np.array(range(11), dtype=np.float64)
        self.tree_str = np.array([[0, 1], [2, 3], [6, 7], [4, 5], [8, 9]])

    def test_no_gamma(self):
        actual = RewardToGo.reward_to_go(self.reward_by_nodes, self.tree_str, gamma=1.0)
        expected = [24.0, 25.0, 27.0, 28.0, 23.0, 24.0, 24.0, 25.0, 18.0, 19.0, 10.0]
        self.assert_result(actual, expected)

        tf_actual = self.tf_tree_reward_to_go()

        self.assert_result(tf_actual, expected)

    def tf_tree_reward_to_go(self, gamma=1):
        sy_node_reward = tf.placeholder(shape=[None], dtype=tf.float32)
        sy_tree_str = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        rtg_reward = compute_tree_reward_to_go(sy_node_reward, sy_tree_str, gamma=gamma)
        with tf.Session() as sess:
            output_final_ = sess.run(fetches=rtg_reward, feed_dict={
                sy_node_reward: self.reward_by_nodes,
                sy_tree_str: self.tree_str
            })
        return output_final_

    def test_gamma(self):

        actual_less = RewardToGo.reward_to_go(self.reward_by_nodes, self.tree_str, gamma=0.5)
        half_gamma_expected = [6.25, 7.25, 8.75, 9.75, 11.0, 12.0, 12.5, 13.5, 13.0, 14.0, 10.0]
        self.assert_result(actual_less, half_gamma_expected)

        tf_expected = self.tf_tree_reward_to_go(0.5)
        self.assert_result(tf_expected, half_gamma_expected)

    def assert_result(self, actual, expected):
        for (e, a) in zip(expected, actual):
            self.assertAlmostEqual(e, a, msg="Tree reward to go is incorrect")

    @staticmethod
    def reward_to_go(rewards, tree_str, gamma=1):
        rewards_to_go = [r for r in rewards]
        offset = len(rewards) - len(tree_str)

        def solve(i_idxs):
            next_roots = []
            for i in i_idxs:
                if i - offset >= 0:
                    children = tree_str[i - offset]
                    for c in children:
                        rewards_to_go[c] += gamma * rewards_to_go[i]
                        next_roots.append(c)

            if len(next_roots) > 0:
                solve(next_roots)

        root_idx = [len(rewards_to_go) - 1]
        solve(root_idx)
        return rewards_to_go





