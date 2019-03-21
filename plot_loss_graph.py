import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


df = pd.read_csv('loss_graph_data.txt', sep='\s+', header=None, names=['round', 'num_rounds', 'value'])

fig, ax = plt.subplots()
ysmoothed = gaussian_filter1d(df['value'][-1000:], sigma=2)
ax.plot( df.index[-1000:], df['value'][-1000:], color='lightgrey', linewidth=1)
ax.plot( df.index[-1000:], ysmoothed, color='red', linewidth=1)
# print(df['value'].mean(axis=0))

last_row = df.iloc[len(df.index)-1]
ax.set(xlabel='#steps', ylabel='loss', title="Seen Datapoints: {}%".format(np.round(last_row['round']/last_row['num_rounds']*100, 2)))
ax.grid()
# plt.yscale('log')

fig.savefig('loss_graph.pdf')
