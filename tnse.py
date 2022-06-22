from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

origin =   np.array([    
                  [4,	4]
                    [0,	8]
                    [2,	4]
                    [4,	9]
                    [3,	10]
                    [4,	0]
                    [5,	1]
                    [0,	0]
                    [4,	0]
                    [0,	7]
                    [0,	2]
                    [3,	2]
                    [3,	2]
                    [4,	2]
                    [4,	8]
                    [5,	9]
                    [1,	3]
                    [0,	5]
                    [3,	10]
                    [0,	6]
                    [3,	1]
                    [0,	8]
                    [1,	1]
                    [4,	10]
                    [0,	4]
                    [4,	2]
                    [2,	7]
                    [0,	1]
                    [3,	4]
                    [0,	2]
                    [0,	2]
                    [2,	6]
                    [1,	4]
                    [4,	6]
                    [2,	5]                   
                       ])

dest = np.array([
              [3,	3],
                [0,	11],
                [0,	3],
                [3,	13],
                [1,	15],
                [3,	0],
                [5,	0],
                [0,	0],
                [3,	0],
                [0,	9],
                [0,	0],
                [1,	0],
                [1,	0],
                [3,	0],
                [3,	11],
                [5,	13],
                [0,	1],
                [0,	5],
                [1,	15],
                [0,	7],
                [1,	0],
                [0,	11],
                [0,	0],
                [3,	15],
                [0,	3],
                [3,	0],
                [0,	9],
                [0,	0],
                [1,	3],
                [0,	0],
                [0,	0],
                [0,	7],
                [0,	3],
                [3,	7],
                [0,	5]                 
                      ]) 

df = pd.DataFrame(data=origin, columns=["column1", "column2"])                      


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(origin)

df = pd.DataFrame()
#df["y"] = y
df["comp-1"] = tsne_results[:,0]
df["comp-2"] = tsne_results[:,1]

nbr_clusters = 3
sns.scatterplot(x="comp-1", y="comp-2",
                palette=sns.color_palette("hls", nbr_clusters),
                data=df).set(title="T-SNE projection")

