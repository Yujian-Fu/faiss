'''
This is for using the T-SNE lib to visualize the dataset
'''


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import utils
from mpl_toolkits.mplot3d import Axes3D
import faiss

#dataset_path = "/home/y/yujianfu/ivf-hnsw/data/analysis/"
#dataset_list = ["DEEP", "SIFT","GIST", "Random_100", "Random_400", "Random_700", "Random_1000", "Gaussian_100", "Gaussian_400", "Gaussian_700", "Gaussian_1000"]

record_path = "/home/yujian/Desktop/extra/Similarity_Search/similarity_search_datasets/models_VQ/"
dataset_path = "/home/yujian/Desktop/extra/Similarity_Search/similarity_search_datasets/data/"
centroid_size = range(100, 1200, 100)
dataset_list = ["SIFT10K"]
all_colors = [
'black',
#'blanchedalmond',
'brown',
'red',
'saddlebrown',
'linen',
'darkorange',
'y',
'g',
'darkslategray',
'deepskyblue',
#'cornsilk',
'b',
#'cyan',
'blueviolet',
'violet',
'm',
'crimson',
'gold',
'darkred',
'darkgrey',
'teal',
'dodgerblue']


for dataset_name in dataset_list:
    base_dataset = dataset_path+ dataset_name+"/"+dataset_name+"_base.fvecs"
    base_vectors = utils.fvecs_read(base_dataset)
    sample_size = 1000
    sample_vectors = base_vectors[np.random.randint(0, base_vectors.shape[0], sample_size), :]
    
    for size in centroid_size:
        centroid_file = record_path + dataset_name+ "/centroids_"+ str(size)+".fvecs"
        centroid = utils.fvecs_read(centroid_file)
        index = faiss.IndexFlatL2(centroid.shape[1])
        index.add(centroid)
        D, I = index.search(sample_vectors, 1)

        tsne = TSNE(n_components=2)
        tsne.fit_transform(np.concatenate((centroid, sample_vectors)))

        fig= plt.figure()
        #ax = Axes3D(fig)
        #ax.scatter(tsne.embedding_[:size, 0],tsne.embedding_[:size, 1], tsne.embedding_[:size, 2], c = all_colors[0], label = "centroids")
        
        #ax.scatter(tsne.embedding_[size:, 0],tsne.embedding_[size:, 1], tsne.embedding_[size:, 2], c = all_colors[1])
        '''
        for i in range(size):
            for j in range (sample_size):
                if (I[j][0] == i):
                    ax.scatter(tsne.embedding_[j+size, 0],tsne.embedding_[j+size, 1], tsne.embedding_[j+size, 2], c = all_colors[i+1])
        '''
        title =  dataset_name + " " + str(int(size)) + " T-SNE Scatter"
        plt.title(title)
        #plt.xlim(np.max(tsne.embedding_[size:, 0]) * 1.1, np.max(tsne.embedding_[size:, 0]) * 1.1)
        #plt.ylim(np.max(tsne.embedding_[size:, 1]) * 1.1, np.max(tsne.embedding_[size:, 1]) * 1.1)
        #plt.zlim(np.max(tsne.embedding_[size:, 2]) * 1.1, np.max(tsne.embedding_[size:, 2]) * 1.1)
        
        plt.scatter(tsne.embedding_[size:, 0],tsne.embedding_[size:, 1],c = 'r')
        plt.scatter(tsne.embedding_[:size, 0],tsne.embedding_[:size, 1],c = 'b', label = "centroids")
        
        plt.legend()
        plt.show()
        #figure_path = dataset_path + dataset_name + "_T-SNE.png"
        #plt.savefig(figure_path)

