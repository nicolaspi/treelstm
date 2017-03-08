import Queue
import numpy as np
import collections

class BatchTreeSample(object):
    def __init__(self, tree):
        o, m, f, p, s, c, sc, l, r = tree.build_batch_tree_sample()
        self.prefixes = p
        self.suffixes = s
        self.observables = o
        self.masks = m
        self.flows = f
        self.children_offsets = c
        self.scatter_indices = sc
        self.labels = l
        self.root_labels = r

class BatchTree(object):

    def set_root(self, root_node):
        self.root = root_node

    class Node(object):
        def __init__(self, tree_parent, samples_values, labels_values, scatter_indices, parent_node = None):
            self.parent = parent_node
            self.samples_values = samples_values
            self.labels_values = labels_values
            self.scatter_indices = scatter_indices
            self.children = []
            self.tree_parent = tree_parent
            self.flow_prefix = 0 if parent_node is None else len(parent_node.samples_values)-1

        def add_child(self, child):
            self.children.append(child)

        def expand_or_add_child(self, sample_values, label_value, child_index):
            if len(self.children) <= child_index : # create a child
                assert(child_index == len(self.children)) # we should grow the tree in a constant way
                self.children.append(BatchTree.Node(self.tree_parent, [-1 if sample_values is None else sample_values], [-1 if label_value is None else label_value], [len(self.samples_values) - 1] , self))
            else:
                self.children[child_index].add_sample(-1 if sample_values is None else sample_values, -1 if label_value is None else label_value, len(self.samples_values) - 1)
            return self.children[child_index]

        def add_sample(self, sample_values, label_values, scatter_indice = None):
            self.samples_values.append(sample_values)
            self.labels_values.append(-1 if label_values is None else label_values)
            if scatter_indice is None:
                self.scatter_indices.append(0 if not self.scatter_indices else self.scatter_indices[-1] + 1)
            else:
                self.scatter_indices.append(scatter_indice)

    def build_batch_tree_sample(self):
        q = collections.deque([self.root])
        flows = collections.deque([])
        prefixes = collections.deque([])
        sufixes = collections.deque([])
        child_offsets = collections.deque([])
        scatter_indices = collections.deque([])
        masks = collections.deque([])
        observables = collections.deque([])
        labels = collections.deque([])
        max_flow = len(self.root.samples_values)
        while q:
            node = q.popleft()
            zero_filling = np.zeros((1,max_flow - len(node.samples_values)))
            mask = np.concatenate([(1.0 * (np.array((node.samples_values)) >= 0)).reshape((1,len(node.samples_values))),zero_filling],axis=1)
            masks.appendleft(mask)
            flows.appendleft(len(node.samples_values))
            prefixes.appendleft(node.flow_prefix)
            sufixes.appendleft(0 if node.parent is None else (len(self.root.samples_values)- (len(node.samples_values) + node.flow_prefix))
                                )
            observable = np.concatenate([np.array(node.samples_values).reshape((1,len(node.samples_values))),zero_filling], axis=1)
            observables.appendleft(observable)
            scatter_indices.appendleft(np.concatenate([np.array(node.scatter_indices).reshape((1,len(node.scatter_indices))),zero_filling], axis=1))
            labels.appendleft(node.labels_values)
            if node.children:
                child_offsets.appendleft([len(q)])
                q.extend(node.children)
            else:
                child_offsets.appendleft([-1])
        observables = np.concatenate(observables,axis=0).astype(dtype=np.int32)
        labels = np.concatenate(labels, axis=0).astype(dtype=np.int32)
        scatter_indices = np.concatenate(scatter_indices, axis=0).astype(dtype=np.int32)
        masks = np.concatenate(masks, axis=0).astype(dtype=np.int32)
        flows = np.array(flows)
        prefixes = np.array(prefixes)
        sufixes = np.array(sufixes)
        child_offsets = np.concatenate(child_offsets, axis=0)
        return observables, masks, flows, prefixes, sufixes, child_offsets, scatter_indices, labels, self.root.labels_values

    @staticmethod
    def empty_tree():
        tree = BatchTree()
        root = BatchTree.Node(tree, [], [], [])
        tree.set_root(root)
        return tree

    def count_nodes(self):
        def count_childs(node):
            c=0
            for child in node.children:
                c += count_childs(child)
            return c+1
        return count_childs(self.root)

    def get_depht(self):
        def depth(node, d):
            max = 0
            if len(node.children) > 0:
                for child in node.children:
                    r =  depth(child, d + 1)
                    if r > max:
                        max = r
            else:
                return d
            return max
        return depth(self.root, 1)

    def check_consistency(self):
        def consistency(node, flow):
            if flow < len(node.samples_values):
                return False
            for child in node.children:
                if consistency(child, len(node.samples_values)) == False :
                    return False
            return True
        return consistency(self.root, len(self.root.samples_values))


def tree_to_matrice_test():
    tree = BatchTree.empty_tree()
    # tree.root.add_sample(1)
    # tree.root.add_sample(1)
    # tree.root.add_sample(1)
    # tree.root.expand_or_add_child(1, 0)
    # tree.root.expand_or_add_child(6, 0)
    # tree.root.expand_or_add_child(6, 0)
    # tree.root.expand_or_add_child(6, 0)
    # tree.root.expand_or_add_child(2, 1)
    # tree.root.expand_or_add_child(8, 1)
    # tree.root.expand_or_add_child(8, 1)
    # tree.root.expand_or_add_child(8, 1)
    # #tree.root.children[0].expand_or_add_child(9, 0)
    # #tree.root.children[0].expand_or_add_child(9, 1)
    # tree.root.children[1].expand_or_add_child(9, 0)
    # tree.root.children[1].expand_or_add_child(9, 1)
    # tree.root.add_sample(1)
    # tree.root.add_sample(1)
    # tree.root.add_sample(1)
    # tree.root.add_sample(1)

    tree.root.add_sample(-1, None)
    tree.root.expand_or_add_child(-1, None, 0)
    tree.root.expand_or_add_child(1, None, 1)
    tree.root.children[0].expand_or_add_child(1, None, 0)
    tree.root.children[0].expand_or_add_child(1, None, 1)

    tree.root.add_sample(-1, None)
    tree.root.expand_or_add_child(2, None, 0)
    tree.root.expand_or_add_child(2, None, 1)

    tree.root.add_sample(-1, None)
    tree.root.expand_or_add_child(-1, None, 0)
    tree.root.expand_or_add_child(3, None, 1)
    tree.root.children[0].expand_or_add_child(3, None, 0)
    tree.root.children[0].expand_or_add_child(3, None, 1)


    o, m, f, p, s, c, sc, l, r =  tree.build_batch_tree_sample()
    print o
    print m
    print f
    print p
    print s
    print c
    print sc
    print l
    print r

if __name__=='__main__':
    tree_to_matrice_test()
