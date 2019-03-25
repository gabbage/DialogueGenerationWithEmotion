import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from model.cbet_model import CBET_BiRNN

from argparse import ArgumentParser
import os.path


class DatasetOSBtargets(Dataset):
    def __init__(self, targets, _pad_len, word2int, max_size=None):
        self.target = targets
        self.pad_len = _pad_len
        self.start_int = word2int['<s>']
        self.eos_int = word2int['</s>']
        self.pad_int = word2int['<pad>']
        self.word2id = word2int
        if max_size is not None:
            self.target = self.target[:max_size]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # for trg add <s> ahead and </s> end
        trg = [int(x) for x in self.target[idx].split()]
        if len(trg) > self.pad_len - 2:
            trg = trg[:self.pad_len - 2]
        trg = trg + [self.eos_int] + [self.pad_int] * (self.pad_len - len(trg) - 2)
        return torch.LongTensor(trg)

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


parser = ArgumentParser(description="tag the OpenSubData with emotion")
parser.add_argument("-f", dest="filename", required=True,
                    help="input file to be tagged as .csv", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
args = parser.parse_args()
filename, file_extension = os.path.splitext(args.filename.name)

####################
# Hyper-parameters #
####################
hidden_size = 300
num_layers = 2
num_classes = 9
batch_size = 1
num_epochs = 2
learning_rate = 0.0005
pad_len = 30
NUM_EMOS = 9
####################

# word dicts
f = open('OpenSubData/word_dict.pkl', 'br')
(word2id, id2word) = pickle.load(f)

model = CBET_BiRNN(hidden_size, num_layers, num_classes, word2id['<pad>'], word2id)

model.load_state_dict(torch.load('cbet_model_good.ckpt'))
model.eval()

df = pd.read_csv(args.filename, index_col=0)
ds = DatasetOSBtargets(df['target'], pad_len, word2id)
dl = DataLoader(ds, batch_size)
df['tag'] = NUM_EMOS
print(len(df.index))

confidence_values = []
confidence_threshold = 23.45

small_df_size = 100

for i, trg in enumerate(dl, 0):
    if i > small_df_size:
        break
    try:
        outputs = model(trg)
        _, predicted = torch.max(outputs.data, 1)
        out_np = outputs.detach().numpy()
        pred_val = predicted.detach().numpy()[0]
        conf_val = out_np[0][pred_val]
        if conf_val >= confidence_threshold:
            df.set_value(i,'tag', pred_val)
        # confidence_values.append(conf_val)
        if i % 1000 == 0:
            print("processed: {} / {} ({}%)".format(i, len(ds), 100.0*float(i)/len(ds)))
    except KeyboardInterrupt:
        print("Amount of Data seen: ", i)
        print("Confidence Percentile: ", np.percentile(np.array(confidence_values), 34))
        sys.exit()

    # df.set_value(i,'tag', predicted.numpy()[0])
df[:small_df_size].to_csv(filename+"_tagged_small"+file_extension)
# print(df)
