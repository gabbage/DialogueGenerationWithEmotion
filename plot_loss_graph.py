import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


df = pd.read_csv('loss_graph_data.txt', sep='\s+', header=None, names=['round', 'num_rounds', 'value'])

fig, ax = plt.subplots()
ysmoothed = gaussian_filter1d(df['value'], sigma=2)
ax.plot( df.index, df['value'], color='lightgrey', linewidth=1)
ax.plot( df.index, ysmoothed, color='red', linewidth=1)

last_row = df.iloc[len(df.index)-1]
ax.set(xlabel='#datapoints / 1000', ylabel='loss', title="Seen Datapoints: {} / {} ({}%)".format(int(last_row['round']), int(last_row['num_rounds']), np.round(last_row['round']/last_row['num_rounds']*100, 2)))
ax.grid()

fig.savefig('loss_graph.pdf')
