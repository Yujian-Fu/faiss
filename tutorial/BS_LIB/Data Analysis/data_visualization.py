'''
This is for using the T-SNE lib to visualize the dataset
'''


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import utils

dataset_path = "/home/y/yujianfu/ivf-hnsw/data/analysis/"
dataset_list = ["DEEP", "SIFT","GIST", "Random_100", "Random_400", "Random_700", "Random_1000", "Gaussian_100", "Gaussian_400", "Gaussian_700", "Gaussian_1000"]

for dataset_name in dataset_list:
    dataset_file = dataset_path + dataset_name + "_query.fvecs"
    dataset = utils.fvecs_read(dataset_file)

    tsne = TSNE(n_components=2)
    tsne.fit_transform(dataset)

    fig = plt.figure()
    title = dataset_name + " T-SNE Scatter"
    plt.title(title)
    plt.scatter(tsne.embedding_[:, 0],tsne.embedding_[:, 1],c = 'r',marker = 'o')
    figure_path = dataset_path + dataset_name + "_T-SNE.png"
    plt.savefig(figure_path)

