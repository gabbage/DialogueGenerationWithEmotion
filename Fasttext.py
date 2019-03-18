import io
import numpy as np
import pickle

class FasttextLoader:
    def __init__(self, ftext_filename, word2id):
        self.vocab = dict()
        fin = io.open(ftext_filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        _, self.emb_dim = map(int, fin.readline().split())
        self.vocab_size = len(word2id)
        self.emb_weights = np.zeros((self.vocab_size, self.emb_dim))

        for line in fin:
            tokens = line.rstrip().split(' ')
            word = tokens[0].lower()
            if word in word2id:
                word_vec = np.array(list(map(float, tokens[1:])))
                self.vocab[word] = word2id[word]
                self.emb_weights[word2id[word]] = word_vec

    def get_embedding_weights(self):
        return self.emb_weights

    def get_vocab(self):
        return self.vocab

if __name__ == "__main__":
    f = open('OpenSubData/word_dict.pkl', 'br')
    (word2id, id2word) = pickle.load(f)
    print(word2id['hello'], type(word2id['hello']))
    print(type(word2id))
    ftextLoader = FasttextLoader('data/crawl-300d-2M.vec', word2id)
    print(ftextLoader.get_embedding_weights())
