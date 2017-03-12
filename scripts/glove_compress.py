import struct
import os

glove_300_filepath = 'data/glove/glove.840B.300d.txt'
glove_300_bin_filepath = 'data/glove/glove_embeddings.b'
glove_300_vocab_filepath = 'data/glove/glove_vocab.txt'
def write_glove_embeddings(glove_300_filepath):
    if os.path.exists(glove_300_bin_filepath):
        print('Found Glove binarized vectors - skip')
        return
    with open(glove_300_filepath) as data_file, open(glove_300_vocab_filepath, 'w') as fv, open(glove_300_bin_filepath, 'wb') as fo:
        print("Binarizing GLOVE embeddings...")
        count = 0
        vocab = {}
        vocab_list = []
        for line in data_file:
            count += 1
            toks = line.split(" ")
            w = toks[0]
            if w in vocab or len(w) <= 32:
                assert len(toks)==301
                wid = len(vocab)
                vocab[w] = wid
                vocab_list.append(w)
                v = map(lambda x : fo.write(struct.pack('f', float(x))), toks[1:])
            if count%10000==0:
                print("Words processed : " + str(count))
        map(lambda x: fv.write(x + "\n"), vocab_list)

if __name__ == '__main__':
    write_glove_embeddings(glove_300_filepath)