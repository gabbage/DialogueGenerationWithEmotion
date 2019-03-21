import pandas as pd
import numpy as np
# from nltk.tokenize import TweetTokenizer
# the TweetTokenizer doesn't split '#Hashtag' into two tokens '#' 'Hashtag', which is needed to apply word2id
from nltk import word_tokenize 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from data.fasttext import FasttextLoader
from model.cbet_model import CBET_BiRNN

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--no_train", help="skip the training part, only evaluate existing model",
                    action="store_true")
parser.add_argument("-t", "--test", help="test the code on only 1000 datapoints (for dev purpose)",
                    action="store_true")
args = parser.parse_args()

####################
# Hyper-parameters #
####################
hidden_size = 300
num_layers = 2
num_classes = 9
batch_size = 1
num_epochs = 2
learning_rate = 0.0005
####################

def load_data():
    #TODO: some sentences have two emotions => one datapoint for each emotion
    cbet = pd.read_csv('data/CBET.csv')
    f = open('OpenSubData/word_dict.pkl', 'br')
    (word2id, id2word) = pickle.load(f)

    text_tokens = []
    emotion_tokens = []
    inter = np.vectorize(int)
    class_label_sel = np.array(list(range(9)))
    maxlen = 0

    data_range = 1000 if args.test else len(cbet.index)

    for row_i in range(data_range):
        try:
            row = cbet.iloc[row_i]

            emotion_vec = inter(row[2:].to_numpy())
            # since some tweets have more than one emotion, 
            # loop over the identity to disentangle the emotions
            for selector in np.identity(len(emotion_vec)):
                if selector.dot(emotion_vec) >= 1.0:
                    class_lbl = class_label_sel.dot(selector)
                    emotion_tokens.append(int(class_lbl.item(0)))

                    word_ids = []
                    w_tokens = word_tokenize(row['text'])
                    for word in w_tokens:
                        if word.lower() in word2id:
                            word_ids.append(word2id[word.lower()])
                        else:
                            word_ids.append(word2id['UNknown'])

                    maxlen = max(maxlen, len(word_ids))
                    # # padding, if needbe. 118 being the max amount of tokens
                    # for i in range(len(w_tokens), 119):
                        # word_ids.append(word2id['<pad>'])

                    word_vec = np.array(word_ids)
                    text_tokens.append(word_vec)
        except IndexError:
            print(row_i, "caused problems",cbet.size)
            break
    print("Maximum Length of tweets: ", maxlen)
    return train_test_split(text_tokens, emotion_tokens, test_size=0.1, random_state=42)

class CBET_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.LongTensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
(tt_train, tt_test, et_train, et_test) = load_data()

trainloader = torch.utils.data.DataLoader(CBET_Dataset(tt_train, et_train), batch_size=batch_size, num_workers=8)
testloader = torch.utils.data.DataLoader(CBET_Dataset(tt_test, et_test), batch_size=batch_size, num_workers=8)

# word dicts
f = open('OpenSubData/word_dict.pkl', 'br')
(word2id, id2word) = pickle.load(f)

model = CBET_BiRNN(hidden_size, num_layers, num_classes, word2id['<pad>'], word2id).to(device)

if args.no_train:
    model.load_state_dict(torch.load('cbet_model_good.ckpt'))
    model.eval()
else: 
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(trainloader)
    for epoch in range(num_epochs):
        for i, (text, labels) in enumerate(trainloader, 0):
            # for i in range(len(tt_train)):
            text = text.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(text)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                with open('loss_graph_data.txt', 'a') as f:
                    f.write("{} {} {}\n".format(((epoch+1)*i), num_epochs*total_step, loss.item()))
                    f.close()
                print ('Epoch [{}/{}], Step {}/{}, Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    all_actual = np.array([])
    all_predicted = np.array([])
    for text, labels in testloader:
        text = text.to(device)
        labels = labels.to(device)
        # all_actual = all_actual + Variable(labels).to_list()
        outputs = model(text)
        _, predicted = torch.max(outputs.data, 1)
        all_actual = np.append(all_actual, Variable(labels).numpy())
        all_predicted = np.append(all_predicted, Variable(predicted).numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    cm = confusion_matrix(all_actual, all_predicted)
    class_names = ["anger","fear","joy","love","sadness","surprise","thankfulness","disgust","guilt"]
    cr = classification_report(all_actual, all_predicted, target_names=class_names)
    print(cr)
    cm_out_f = open('cm_result.pkl', 'bw')
    pickle.dump(cm, cm_out_f)

    print('Test Accuracy of the model: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'cbet_model.ckpt')
