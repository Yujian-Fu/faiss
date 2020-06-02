import numpy as np
import faiss
import time
import random
import struct

nc1 = 10000
nc2 = 500
niter = 30
train_size = 10000000
verbose = True
use_GPU = False
num_batches = 10000
learn_set = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs"
base_set = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs"

first_level_path = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/nc1_" + str(nc1) + "_nt_" + str(train_size) + ".fvecs"
second_level_path = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/nc2_" + str(nc2) + "_nt_" + str(train_size) + ".fvecs"

LearnSet = np.memmap(learn_set, dtype='uint8', mode='r')
dimension = LearnSet[:4].view('int32')[0]
print("The dimension of dataset is ", dimension)

print("Training the first level centroids")
file = open(first_level_path, "wb")
start = time.time()
LearnSet = LearnSet.reshape(-1, dimension + 4)[:, 4 :]
SubLearnSet = LearnSet[random.sample(range(LearnSet.shape[0]), train_size), :]
SubLearnSet = np.ascontiguousarray(SubLearnSet.astype('float32'))
kmeans = faiss.Kmeans(dimension, nc1, niter = niter, verbose = verbose, gpu = use_GPU)
kmeans.train(SubLearnSet)
end = time.time()
print("Finished training the first level centroids in ", end-start, "s")
print("Saving the first level centroids")
for i in range(nc1):
    file.write(struct.pack('i', dimension))
    for j in range(dimension):
        file.write(struct.pack('f', kmeans.centroids[i][j]))
file.close()

print("Assigning the LearnSet vectors to first level centroids")
index_lists = [[] for i in range(nc1)]
D, I = kmeans.index.search(LearnSet, 1)
for i in range(I.shape[0]):
    index_lists[I[i][0]].append(i)

print("Training second level centroids")
file = open(second_level_path, "wb")
for i in range(nc1):
    start = time.time()
    kmeans = faiss.Kmeans(dimension, nc2, niter = niter, verbose = verbose, gpu = use_GPU)
    train_set = LearnSet[index_lists[i], :]
    kmeans.train(train_set)
    for j in range(nc2):
        file.write(struct.pack('i', dimension))
        for k in range(dimension):
            file.write(struct.pack('f', kmeans.centroids[j][k]))
    end = time.time()
    if (i % 100 == 0):
        print(i, " / ", nc1, " in ", end - start, " s ")
file.close()

'''
BaseSet = np.memmap(base_set, dtype='uint8', mode='r')
dimension = BaseSet[:4].view('int32')[0]
BaseSet = BaseSet.reshape(-1, d + 4)[:, 4:]
base_size = BaseSet.shape[0]
batch_size = int(base_size / num_batches)
for i in range(num_batches):
'''
    




    



