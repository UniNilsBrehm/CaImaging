import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
import pandas as pd
import seaborn as sns
from scipy.cluster.vq import whiten
from scipy.cluster.hierarchy import fcluster, linkage
from IPython import embed


# Generate initial data
data = np.vstack((
    (rand(30, 2) + 1),
    (rand(30, 2) + 2.5),
    (rand(30, 2) + 4)
))

# standardize (normalize) the features
data = whiten(data)

# Compute the distance matrix
matrix = linkage(
    data,
    method='ward',
    metric='euclidean'
)

# Assign cluster labels
labels = fcluster(
    matrix, 3,
    criterion='maxclust'
)

# Create DataFrame
df = pd.DataFrame(data, columns=['x', 'y'])
df['labels'] = labels

# Plot Clusters
sns.scatterplot(
    x='x',
    y='y',
    hue='labels',
    data=df
)

plt.title('Hierachical Clustering with SciPy')
plt.show()
embed()
exit()