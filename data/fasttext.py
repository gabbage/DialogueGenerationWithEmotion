import io
import numpy as np
import pickle

class FasttextLoader:
    def __init__(self, ftext_filename, word2id, initrange):
        self.vocab = dict()
        fin = io.open(ftext_filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        _, self.emb_dim = map(int, fin.readline().split())
        self.vocab_size = len(word2id)
        np.random.seed(42)
        self.emb_weights = np.random.uniform(-initrange, initrange, (self.vocab_size, self.emb_dim))

        for line in fin:
            tokens = line.rstrip().split(' ')
            word = tokens[0].lower()
            if word in word2id:
                word_vec = np.array(list(map(float, tokens[1:])))
                self.vocab[word] = word2id[word]
                self.emb_weights[word2id[word]] = word_vec
        self.emb_weights[word2id['<pad>']] = np.zeros([self.emb_dim])

    def get_embedding_weights(self):
        return self.emb_weights

    def get_vocab(self):
        return self.vocab

if __name__ == "__main__":
    f = open('OpenSubData/word_dict.pkl', 'br')
    (word2id, id2word) = pickle.load(f)
    ftextLoader = FasttextLoader('data/crawl-300d-2M.vec', word2id, 0.1)

    zero_count = 0
    check = np.ones(ftextLoader.emb_dim)
    emb_w = ftextLoader.get_embedding_weights()
    for x in emb_w:
        if check.dot(x) == 0.0:
            zero_count += 1

    print("Zero Elements in weights: {} / {}".format(zero_count, len(emb_w)))
