import torch 
import torch.nn as nn
from data.fasttext import FasttextLoader

# Bidirectional recurrent neural network (many-to-one)
class CBET_BiRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes, pad, word2id):
        super(CBET_BiRNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ftextLoader = FasttextLoader('data/crawl-300d-2M.vec', word2id, 0.1)

        self.emb_dim = ftextLoader.emb_dim
        self.vocab_size = ftextLoader.vocab_size
        self.embedding = nn.Embedding(
                self.vocab_size, self.emb_dim, pad)
        self.embedding.weight = nn.Parameter(
                torch.FloatTensor(ftextLoader.get_embedding_weights()))

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

        # Init Weights
        torch.nn.init.normal_(self.fc.weight, mean=0, std=1)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        x_emb = self.embedding(x)

        # Forward propagate LSTM
        out, _ = self.lstm(x_emb, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2) 
        # Inputs: input, (h_0, c_0)
        # From the DOC: input of shape (seq_len, batch, input_size): tensor
        # containing the features of the input sequence. 

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
