import numpy as np
import faiss
import time
import random
import struct

learn_set = "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_learn.fvecs"
base_set = "/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_base.fvecs"

#a = np.memmap(learn_set, dtype='uint8', mode='r')
#d = int(a[:4].view('int32')[0])
#b = a.reshape(-1, d + 4)[:, 4:]

a = np.fromfile(learn_set, dtype='int32')
d = a[0]
b =  a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

print("Dataset dimension is ", d, " train set size is: ", b.shape[0])
niter = 20
verbose = True

start = time.time()
train_size = 100000
b = b[random.sample(range(b.shape[0]), train_size), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 10000
print("start training")
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " Time for reading: ", end-start, " s \n\n")
f = open("/home/y/yujianfu/ivf-hnsw/data/SIFT1M/centroids_sift1M.fvecs", "wb")
for i in range(ncentroids):
    f.write(struct.pack('i', d))
    for j in range(d):
        f.write(struct.pack('f', kmeans.centroids[i][j]))
f.close()


#a = np.memmap(base_set, dtype='uint8', mode='r')
#d = a[:4].view('int32')[0]
#b = a.reshape(-1, d + 4)[:, 4:]

a = np.fromfile(base_set, dtype='int32')
d = a[0]
b =  a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

base_size = b.shape[0]
print("Base set size is ", base_size)
num_batches = 10000
batch_size = int(base_size / num_batches)
f = open("/home/y/yujianfu/ivf-hnsw/data/SIFT1M/precomputed_idxs_sift1M.ivecs", "wb")
print("Saving ids")
for i in range(num_batches):
    if (i % 100 == 0):
        print("Finishied [ ", i , " / ", str(num_batches), " ]")
    subset = b[range(batch_size*i, batch_size * (i+1)), :]
    subset = np.ascontiguousarray(subset.astype('float32'))
    D, I = kmeans.index.search(subset, 1)
    f.write(struct.pack('i', batch_size))
    for j in range(I.shape[0]):
        f.write(struct.pack('i', I[j][0]))
f.close()


