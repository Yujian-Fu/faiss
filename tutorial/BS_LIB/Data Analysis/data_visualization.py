'''
This is for using the T-SNE lib to visualize the dataset
'''


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

y = np.random.normal(0, 0.5, (100, 10))
print(y.shape)
tsne = TSNE(n_components=2)
tsne.fit_transform(y)
print(tsne.embedding_)

fig = plt.figure()
plt.title("T-SNE Scatter Figure")
plt.scatter(tsne.embedding_[:, 0],tsne.embedding_[:, 1],c = 'r',marker = 'o')

plt.legend('x1')
plt.show()

