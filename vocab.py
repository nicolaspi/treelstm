class Vocab(object):

    def __init__(self,path):
        self.words = set()
        self.word2idx={}
        self.idx2word={}

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