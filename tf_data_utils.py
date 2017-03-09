

from tf_treenode import tNode,processTree
import numpy as np
import os
import random
import struct
from batch_tree import BatchTree, BatchTreeSample

class Vocab(object):

    def __init__(self,path):
        self.words = set()
        self.word2idx={}
        self.idx2word={}
        self.embed_matrix = None
        self.load(path)


    def load(self,path):

        with open(path,'r') as f:
            for line in f:
                w=line.strip()
                assert w not in self.words
                self.words.add(w)
                self.word2idx[w] = len(self.words) -1 # 0 based index
                self.idx2word[self.word2idx[w]]=w

    def __len__(self):
        return len(self.words)

    def encode(self,word):
        if word not in self.words:
            return None
        return self.word2idx[word]

    def decode(self,idx):
        assert idx < len(self.words) and idx >=0
        return self.idx2word[idx]
    def size(self):
        return len(self.words)

    def compute_gloves_embedding(self, glove_dir):
        vector_format = 'f' * 300
        size = struct.calcsize(vector_format)
        glove_voc = Vocab(os.path.join(glove_dir, 'glove_vocab.txt'))
        self.embed_matrix = np.random.rand(self.size(), 300) * 0.1 - 0.05
        with open(os.path.join(glove_dir, 'glove_embeddings.b'), "rb") as fi:
            id = 0
            while True:
                vector = fi.read(size)
                if len(vector) == size:
                    v = struct.unpack(vector_format, vector)
                    idx = self.encode(glove_voc.decode(id))
                    if idx is not None:
                        self.embed_matrix[idx,:] = v
                else:
                    break
                id += 1
                if(id%100000 == 0):
                    print("read " + str(id) + " embeds and counting...")

def load_sentiment_treebank(data_dir, glove_dir, fine_grained):
    voc=Vocab(os.path.join(data_dir,'vocab-cased.txt'))
    if glove_dir is not None:
        voc.compute_gloves_embedding(glove_dir)


    split_paths={}
    for split in ['train','test','dev']:
        split_paths[split]=os.path.join(data_dir,split)

    fnlist=[tNode.encodetokens,tNode.relabel]
    arglist=[voc.encode,fine_grained]
    #fnlist,arglist=[tNode.relabel],[fine_grained]

    data={}
    for split,path in split_paths.iteritems():
        sentencepath=os.path.join(path,'sents.txt')
        treepath=os.path.join(path,'parents.txt')
        labelpath=os.path.join(path,'labels.txt')
        trees=parse_trees(sentencepath,treepath,labelpath)
        if not fine_grained:
            trees=[tree for tree in trees if tree.label != 0]
        trees = [(processTree(tree,fnlist,arglist),tree.label) for tree in trees]
        data[split]=trees

    return data,voc

def load_subtitles(data_dir, only_supervised_data = False):
    voc=Vocab(os.path.join(data_dir,'vocab.txt'))

    fnlist=[tNode.encodetokens]
    arglist=[voc.encode]
    #fnlist,arglist=[tNode.relabel],[fine_grained]

    sentencepath=os.path.join(data_dir,'subtitles.txt')
    treepath=os.path.join(data_dir,'subtitles.cparents')
    labelpath=os.path.join(data_dir,'labels.txt')
    trees=parse_trees(sentencepath,treepath, labelpath)
    with open(labelpath) as fl:
        label_line = fl.readline()
        labels = [-1 if (l[1]=="?" or l[1]=="#") else (1 if l[1]=="1" else 0) for l in label_line.split()]
        trees = [(processTree(tree,fnlist + tNode.compute_labels, arglist + [labels]),tree.label) for tree in trees]
        if (only_supervised_data):
            trees = [tree for tree in trees if tree.label >= 0]
        data = trees
    return data,voc


def parse_trees(sentencepath, treepath, labelpath):
    trees=[]
    with open(treepath,'r') as ft, open(
        sentencepath,'r') as f:
        fl = None
        if labelpath is not None:
            fl = open(labelpath)

        while True:
            parentidxs = ft.readline()
            labels = None
            if fl is not None:
                labels = fl.readline()
                labels = [int(l) if l != '#' and l != '?' else None for l in labels.strip().split()]

            sentence=f.readline()
            if not parentidxs or not labels or not sentence:
                break
            parentidxs=[int(p) for p in parentidxs.strip().split() ]

            tree=parse_tree(sentence,parentidxs,labels)
            trees.append(tree)
    if fl is not None:
        fl.close()

    return trees

def parse_tree(sentence, parents, labels):
    nodes = {}
    parents = [p - 1 for p in parents]  #change to zero based
    sentence=[w for w in sentence.strip().split()]
    for i in xrange(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tNode(idx)  
                if prev is not None:
                    assert prev.idx != node.idx
                    node.add_child(prev)

                if labels is not None:
                    node.label = labels[idx]
                else:
                    node.label = None
                nodes[idx] = node

                if idx < len(sentence):
                    node.word = sentence[idx]

                parent = parents[idx]
                if parent in nodes:
                    assert len(nodes[parent].children) < 2
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break

                prev = node
                idx = parent

    return root

def BFStree(root):
    from collections import deque
    node=root
    leaves=[]
    inodes=[]
    queue=deque([node])
    func=lambda node:node.children==[]

    while queue:
        node=queue.popleft()
        if func(node):
            leaves.append(node)
        else:
            inodes.append(node)
        if node.children:
            queue.extend(node.children)

    return leaves,inodes

def extract_tree_data(tree,max_degree=2,only_leaves_have_vals=True,with_labels=False):
    #processTree(tree)
    #fnlist=[tree.encodetokens,tree.relabel]
    #arglist=[voc.encode,fine_grained]
    #processTree(tree,fnlist,arglist)
    leaves,inodes=BFStree(tree)
    labels=[]
    leaf_emb=[]
    tree_str=[]
    i=0
    for leaf in reversed(leaves):
        leaf.idx = i
        i+=1
        labels.append(leaf.label)
        leaf_emb.append(leaf.word)
    for node in reversed(inodes):
        node.idx=i
        c=[child.idx for child in node.children]
        tree_str.append(c)
        labels.append(node.label)
        if not only_leaves_have_vals:
            leaf_emb.append(-1)
        i+=1
    if with_labels:
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(leaf_emb,dtype='int32'),
                np.array(tree_str,dtype='int32'),
                np.array(labels,dtype=float),
                np.array(labels_exist,dtype=float))
    else:
        print leaf_emb,'asas'
        return (np.array(leaf_emb,dtype='int32'),
               np.array(tree_str,dtype='int32'))

def build_batch_trees(trees, mini_batch_size):
    def expand_batch_with_sample(batch_node, sample_node):
        if batch_node.parent is None: # root
            batch_node.add_sample(-1 if sample_node.word is None else sample_node.word, tree.label)
        for child in zip(range(len(sample_node.children)),sample_node.children):
            batch_node.expand_or_add_child(child[1].word, child[1].label, child[0])
        for children in zip(batch_node.children, sample_node.children):
            expand_batch_with_sample(children[0], children[1])
    batches = []
    while(len(trees)>0):
        batch = trees[-mini_batch_size:]
        del trees[-mini_batch_size:]
        batch_tree = BatchTree.empty_tree()
        for tree in batch:
            expand_batch_with_sample(batch_tree.root, tree)
        batches.append(BatchTreeSample(batch_tree))
    return batches


def build_labelized_batch_trees(data, mini_batch_size):
    trees = [s[0] for s in data]
    labels = [s[1] for s in data]
    labels_batches = []
    while (len(labels) > 0):
        batch = labels[-mini_batch_size:]
        del labels[-mini_batch_size:]
        labels_batches.append(np.array(batch))
    tree_batches = build_batch_trees(trees, mini_batch_size)
    return zip(tree_batches, labels_batches)


def extract_batch_tree_data(batchdata,fillnum=120):

    dim1,dim2=len(batchdata),fillnum
    #leaf_emb_arr,treestr_arr,labels_arr=[],[],[]
    leaf_emb_arr = np.empty([dim1,dim2],dtype='int32')
    leaf_emb_arr.fill(-1)
    treestr_arr = np.empty([dim1,dim2,2],dtype='int32')
    treestr_arr.fill(-1)
    labels_arr = np.empty([dim1,dim2],dtype=float)
    labels_arr.fill(-1)
    for i,(tree,_) in enumerate(batchdata):
        input_,treestr,labels,_=extract_tree_data(tree,
                                          max_degree=2,
                               only_leaves_have_vals=False,
                                          with_labels = True)
        leaf_emb_arr[i,0:len(input_)]=input_
        treestr_arr[i,0:len(treestr),0:2]=treestr
        labels_arr[i,0:len(labels)]=labels

    return leaf_emb_arr,treestr_arr,labels_arr

def extract_seq_data(data,numsamples=0,fillnum=100):
    seqdata=[]
    seqlabels=[]
    for tree,_ in data:
        seq,seqlbls=extract_seq_from_tree(tree,numsamples)
        seqdata.extend(seq)
        seqlabels.extend(seqlbls)

    seqlngths=[len(s) for s in seqdata]
    maxl=max(seqlngths)
    assert fillnum >=maxl
    if 1:
        seqarr=np.empty([len(seqdata),fillnum],dtype='int32')
        seqarr.fill(-1)
        for i,s in enumerate(seqdata):
            seqarr[i,0:len(s)]=np.array(s,dtype='int32')
        seqdata=seqarr
    return seqdata,seqlabels,seqlngths,maxl

def extract_seq_from_tree(tree,numsamples=0):

    if tree.span is None:
        tree.postOrder(tree,tree.get_spans)

    seq,lbl=[],[]
    s,l=tree.span,tree.label
    seq.append(s)
    lbl.append(l)

    if not numsamples:
        return seq,lbl


    num_nodes = tree.idx
    if numsamples==-1:
        numsamples=num_nodes
    #numsamples=min(numsamples,num_nodes)
    #sampled_idxs = random.sample(range(num_nodes),numsamples)
    #sampled_idxs=range(num_nodes)
    #print sampled_idxs,num_nodes

    subtrees={}
    #subtrees[tree.idx]=
    #func=lambda tr,su:su.update([(tr.idx,tr)])
    def func_(self,su):
        su.update([(self.idx,self)])

    tree.postOrder(tree,func_,subtrees)

    for j in xrange(numsamples):#sampled_idxs:
        i=random.randint(0,num_nodes)
        root = subtrees[i]
        s,l=root.span,root.label
        seq.append(s)
        lbl.append(l)

    return seq,lbl

def get_max_len_data(datadic):
    maxlen=0
    for data in datadic.values():
        for tree,_ in data:
            tree.postOrder(tree,tree.get_numleaves)
            assert tree.num_leaves > 1
            if tree.num_leaves > maxlen:
                maxlen=tree.num_leaves

    return maxlen

def get_max_node_size(datadic):
    maxsize=0
    for data in datadic.values():
        for tree,_ in data:
            tree.postOrder(tree,tree.get_size)
            assert tree.size > 1
            if tree.size > maxsize:
                maxsize=tree.size

    return maxsize

def test_fn():
    data_dir='./stanford_lstm/data/sst'
    fine_grained=0
    data,_=load_sentiment_treebank(data_dir,fine_grained)
    for d in data.itervalues():
        print len(d)

    d=data['dev']
    a,b,c,_=extract_seq_data(d[0:1],5)
    print a,b,c

    print get_max_len_data(data)
    return data
if __name__=='__main__':
    test_fn()
