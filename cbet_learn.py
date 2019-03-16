import pandas as pd
import numpy as np
# from nltk.tokenize import TweetTokenizer
# the TweetTokenizer doesn't split '#Hashtag' into two tokens '#' 'Hashtag', which is needed to apply word2id
from nltk import word_tokenize 
from sklearn.model_selection import train_test_split
import pickle
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def load_data():
    cbet = pd.read_csv('data/CBET.csv')
    f = open('OpenSubData/word_dict.pkl', 'br')
    (word2id, id2word) = pickle.load(f)

    text_tokens = []
    emotion_tokens = []
    floater = np.vectorize(float)
    max_words = 0

    # for row_i in range(len(cbet.index)):
    for row_i in range(1000):
        try:
            row = cbet.iloc[row_i]

            emotion_vec = floater(row[2:].to_numpy())
            emotion_tokens.append(torch.from_numpy(emotion_vec).type(torch.float))

            word_ids = []
            w_tokens = word_tokenize(row['text'])
            max_words = max(max_words, len(w_tokens))
            for word in w_tokens:
                if word.lower() in word2id:
                    word_ids.append(word2id[word.lower()])
                else:
                    word_ids.append(word2id['UNknown'])
            word_vec = torch.tensor(word_ids, dtype=torch.double)
            text_tokens.append(word_vec)
        except IndexError:
            print(row_i, "caused problems",cbet.size)
            break
    # = 
    return (train_test_split(text_tokens, emotion_tokens, test_size=0.1, random_state=42), max_words)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
((tt_train, tt_test, et_train, et_test), input_size) = load_data()

# Hyper-parameters
sequence_length = 28
# input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 9
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, 1, self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, 1, self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2) 
        # Inputs: input, (h_0, c_0)
        # From the DOC: input of shape (seq_len, batch, input_size): tensor
        # containing the features of the input sequence. 

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# t = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           # batch_size=batch_size, 
                                           # shuffle=True)
model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
for i in range(10):
    print(model((1, tt_train[i], len(tt_train[i]))))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
    # for i, (images, labels) in enumerate(train_loader):
        # images = images.reshape(-1, sequence_length, input_size).to(device)
        # labels = labels.to(device)
        
        # # Forward pass
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        
        # # Backward and optimize
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # if (i+1) % 100 == 0:
            # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   # .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# # # Test the model
# # with torch.no_grad():
    # # correct = 0
    # # total = 0
    # # for images, labels in test_loader:
        # # images = images.reshape(-1, sequence_length, input_size).to(device)
        # # labels = labels.to(device)
        # # outputs = model(images)
        # # _, predicted = torch.max(outputs.data, 1)
        # # total += labels.size(0)
        # # correct += (predicted == labels).sum().item()

    # # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# # # Save the model checkpoint
# # torch.save(model.state_dict(), 'model.ckpt')
