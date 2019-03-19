import pandas as pd
import numpy as np
# from nltk.tokenize import TweetTokenizer
# the TweetTokenizer doesn't split '#Hashtag' into two tokens '#' 'Hashtag', which is needed to apply word2id
from nltk import word_tokenize 
from sklearn.model_selection import train_test_split
import pickle
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
import torchvision
import torchvision.transforms as transforms
from Fasttext import FasttextLoader

def load_data():
    #TODO: some sentences have two emotions => one datapoint for each emotion
    cbet = pd.read_csv('data/CBET.csv')
    f = open('OpenSubData/word_dict.pkl', 'br')
    (word2id, id2word) = pickle.load(f)

    text_tokens = []
    emotion_tokens = []
    inter = np.vectorize(int)
    class_label_sel = torch.tensor(list(range(9)))

    # for row_i in range(len(cbet.index)):
    for row_i in range(1000):
        try:
            row = cbet.iloc[row_i]

            emotion_vec = inter(row[2:].to_numpy())
            # since some tweets have more than one emotion, 
            # loop over the identity to disentangle the emotions
            for selector in np.identity(len(emotion_vec)):
                if selector.dot(emotion_vec) >= 1.0:
                    class_lbl = torch.dot(class_label_sel, torch.from_numpy(selector).type(torch.long))
                    emotion_tokens.append(class_lbl)

                    word_ids = []
                    w_tokens = word_tokenize(row['text'])
                    for word in w_tokens:
                        if word.lower() in word2id:
                            word_ids.append(word2id[word.lower()])
                        else:
                            word_ids.append(word2id['UNknown'])

                    # padding, if needbe. 42 being the max amount of tokens
                    # for i in range(len(w_tokens), 42):
                        # word_ids.append(word2id['<pad>'])

                    word_vec = torch.tensor(word_ids, dtype=torch.long)
                    text_tokens.append(word_vec)
        except IndexError:
            print(row_i, "caused problems",cbet.size)
            break
    # = 
    return train_test_split(text_tokens, emotion_tokens, test_size=0.1, random_state=42)

class CBET_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(X)

    def __getitem__(self, idx):
        return X[i]


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
(tt_train, tt_test, et_train, et_test) = load_data()

train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)

# word dicts
f = open('OpenSubData/word_dict.pkl', 'br')
(word2id, id2word) = pickle.load(f)

# Hyper-parameters
# sequence_length = 28
# input_size = 28
hidden_size = 300
num_layers = 2
num_classes = 9
batch_size = 100
num_epochs = 2
learning_rate = 0.0007

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes, pad, word2id):
        super(BiRNN, self).__init__()
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
        h0 = torch.zeros(self.num_layers*2, 1, self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, 1, self.hidden_size).to(device)
        x_emb = self.embedding(x)

        # Forward propagate LSTM
        out, _ = self.lstm(x_emb, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2) 
        # Inputs: input, (h_0, c_0)
        # From the DOC: input of shape (seq_len, batch, input_size): tensor
        # containing the features of the input sequence. 

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = BiRNN(hidden_size, num_layers, num_classes, word2id['<pad>'], word2id).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# total_step = len(train_loader)
for epoch in range(num_epochs):
    for i in range(len(tt_train)):
        text = tt_train[i].unsqueeze(0).to(device)
        labels = et_train[i].unsqueeze(0).to(device)

        # Forward pass
        outputs = model(text)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            with open('loss_graph_data.txt', 'a') as f:
                f.write("{} {} {}\n".format(((epoch+1)*i), (num_epochs*len(tt_train)), loss.item()))
                f.close()
            print ('Epoch [{}/{}], Step {}/{}, Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(tt_train), loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(len(tt_test)):
        text = tt_test[i].unsqueeze(0).to(device)
        labels = et_test[i].unsqueeze(0).to(device)
        outputs = model(text)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'cbet_model.ckpt')
