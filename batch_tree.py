import numpy as np
import collections

class BatchTreeSample(object):
    def __init__(self, tree):
        observables, flows, input_scatter, scatter_out, scatter_in, scatter_in_indices, labels, observables_indices, out_indices, child_scatter_indices, nodes_count, nodes_count_per_indice = tree.build_batch_tree_sample()
        self.observables = observables
        self.flows = flows
        self.input_scatter = input_scatter
        self.scatter_out = scatter_out
        self.scatter_in = scatter_in
        self.scatter_in_indices = scatter_in_indices
        self.labels = labels
        self.observables_indices = observables_indices
        self.out_indices = out_indices
        self.root_labels = labels[out_indices[-2]:out_indices[-1]]
        self.child_scatter_indices = child_scatter_indices
        self.nodes_count = nodes_count
        self.nodes_count_per_indice = nodes_count_per_indice

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
        q = collections.deque([(self.root,0)])
        batch_levels = dict()
        max_flow = len(self.root.samples_values)
        max_level = 0
        while q:
            node, level = q.popleft()
            max_level = level
            mask = np.array(node.samples_values) >= 0#.reshape((1,len(node.samples_values)))
            observable = np.array(node.samples_values)[np.array(node.samples_values) >= 0]#.reshape((1, len(node.samples_values)))
            scatter_out= np.array(node.scatter_indices)#.reshape((1, len(node.scatter_indices)))
            labels= np.array(node.labels_values)#.reshape((1, len(node.labels_values)))

            if(level in batch_levels):
                level_dict = batch_levels[level]
                level_dict["mask"].append(mask)
                level_dict["observables"].append(observable)
                level_dict["scatter_out"].append(scatter_out + max_flow*len(level_dict["scatter_out"]))
                level_dict["flow"] += len(node.samples_values)
                level_dict["labels"].append(labels)
            else:
                level_dict = dict()
                level_dict["mask"] = collections.deque([mask])
                level_dict["observables"] = collections.deque([observable])
                level_dict["flow"] = len(node.samples_values)
                level_dict["scatter_out"] = collections.deque([scatter_out])
                level_dict["labels"] = collections.deque([labels])
                level_dict["childs_transpose_scatter"] = collections.deque([])
                level_dict["scatter_in"] = collections.deque([])
                level_dict["childs_transpose_scatter_offset"] = 0
                level_dict["scatter_in_offset"] = 0
                batch_levels[level] = level_dict

            if node.children:
                q.extend(zip(node.children, [level+1]*len(node.children)))
                c = node.children[0]
                childs_info = np.arange(len(c.samples_values)*len(node.children)).reshape(len(c.samples_values),len(node.children)).transpose().reshape(-1) + batch_levels[level]["childs_transpose_scatter_offset"]
                batch_levels[level]["scatter_in"].append(np.array(c.scatter_indices) + batch_levels[level]["scatter_in_offset"])
                batch_levels[level]["childs_transpose_scatter"].append(childs_info)
                batch_levels[level]["childs_transpose_scatter_offset"] = childs_info[-1] + 1

            batch_levels[level]["scatter_in_offset"] += len(node.samples_values)

        max_level += 1
        input_scatter = np.array([]).astype(dtype=np.int32)
        observables = np.array([]).astype(dtype=np.int32)
        scatter_out = np.array([]).astype(dtype=np.int32)
        scatter_in = np.array([]).astype(dtype=np.int32)
        childs_transpose_scatter = np.array([]).astype(dtype=np.int32)
        labels = np.array([]).astype(dtype=np.int32)
        nodes_count = np.zeros(max_level).astype(dtype=np.int32)
        observables_indices = np.zeros(max_level+1).astype(dtype=np.int32)
        out_indices = np.zeros(max_level + 1).astype(dtype=np.int32)
        flows = np.zeros(max_level).astype(dtype=np.int32)
        scatter_in_indices = np.zeros(max_level).astype(dtype=np.int32)
        levels = range(max_level)
        levels.reverse()
        for l,i in zip(range(max_level),levels):
            level_dict = batch_levels[i]
            mask_level = np.concatenate(level_dict["mask"], axis=0)
            input_scatter_level = np.arange(len(mask_level))[mask_level]
            input_scatter = np.concatenate([input_scatter, input_scatter_level], axis=0)
            observables = np.concatenate([observables, np.concatenate(level_dict["observables"], axis=0).astype(dtype=np.int32)], axis=0)
            observables_indices[l+1] = observables_indices[l] + len(input_scatter_level)
            out_indices[l + 1] = out_indices[l] + level_dict["flow"]
            flows[l] = level_dict["flow"]
            nodes_count[l] = len(level_dict["observables"])
            scatter_out = np.concatenate([scatter_out, np.concatenate(level_dict["scatter_out"], axis=0).astype(dtype=np.int32)], axis=0)
            if l > 0:
                childs_transpose_scatter = np.concatenate([childs_transpose_scatter, np.concatenate(level_dict["childs_transpose_scatter"], axis=0).astype(dtype=np.int32)], axis=0)
                scatter_in_level = np.concatenate(level_dict["scatter_in"], axis=0).astype(dtype=np.int32)
                scatter_in_indices[l] = scatter_in_indices[l-1] + len(scatter_in_level)
                scatter_in = np.concatenate(
                    [scatter_in, scatter_in_level], axis=0)
            labels = np.concatenate([labels, np.concatenate(level_dict["labels"], axis=0).astype(dtype=np.int32)], axis=0)
        samples_indices = scatter_out % max_flow
        _, c = np.unique(samples_indices, return_counts=True)
        nodes_count_per_indice = c[samples_indices]
        return observables, flows, input_scatter, scatter_out, scatter_in, scatter_in_indices, labels, observables_indices, out_indices, childs_transpose_scatter, nodes_count, nodes_count_per_indice

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

    # tree.root.add_sample(-1, None)
    # tree.root.expand_or_add_child(-1, None, 0)
    # tree.root.expand_or_add_child(1, None, 1)
    # tree.root.children[0].expand_or_add_child(1, None, 0)
    # tree.root.children[0].expand_or_add_child(1, None, 1)
    #
    # tree.root.add_sample(-1, None)
    # tree.root.expand_or_add_child(2, None, 0)
    # tree.root.expand_or_add_child(2, None, 1)
    #
    # tree.root.add_sample(-1, None)
    # tree.root.expand_or_add_child(-1, None, 0)
    # tree.root.expand_or_add_child(3, None, 1)
    # tree.root.children[0].expand_or_add_child(3, None, 0)
    # tree.root.children[0].expand_or_add_child(3, None, 1)
    # tree.root.children[1].expand_or_add_child(3, None, 0)
    # tree.root.children[1].expand_or_add_child(3, None, 1)

    #tree.root.add_sample(7, 1)

    tree.root.add_sample(-1, 1)
    tree.root.expand_or_add_child(1, 1, 0)
    tree.root.expand_or_add_child(-1, 1, 1)
    tree.root.children[1].expand_or_add_child(3, 0, 0)
    tree.root.children[1].expand_or_add_child(3, 0, 1)
    #tree.root.children[1].expand_or_add_child(3, 0, 0)
    #tree.root.children[1].expand_or_add_child(3, 0, 1)

    observables, flows, input_scatter, scatter_out, scatter_in, scatter_in_indices, labels, observables_indices, out_indices, childs_transpose_scatter, nodes_count, nodes_count_per_indice = tree.build_batch_tree_sample()

    print observables, "observables"
    print observables_indices, "observables_indices"
    print flows, "flows"
    print input_scatter, "input_scatter"
    print scatter_out, "scatter_out"
    print scatter_in, "scatter_in"
    print scatter_in_indices, "scatter_in_indices"
    print labels , "labels"
    print out_indices, "out_indices"
    print childs_transpose_scatter , "childs_transpose_scatter"
    print nodes_count, "nodes_count"
    print nodes_count_per_indice, "nodes_count_per_indice"

if __name__=='__main__':
    tree_to_matrice_test()
