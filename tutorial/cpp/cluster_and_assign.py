import numpy as np
import faiss
import time
import random
import struct

learn_set = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs"
base_set = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs"

a = np.memmap(learn_set, dtype='uint8', mode='r')
d = int(a[:4].view('int32')[0])
print("Dataset dimension is ", d)
niter = 20
verbose = True

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 1000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 15
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")
f = open("/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_nc_"+str(ncentroids)+".fvecs", "wb")
for i in range(ncentroids):
    f.write(struct.pack('i', d));
    for j in range(d):
        f.write(struct.pack('f', kmeans.centroids[i][j]))
f.close()


a = np.memmap(base_set, dtype='uint8', mode='r')
d = a[:4].view('int32')[0]
b = a.reshape(-1, d + 4)[:, 4:]
base_size = b.shape[0]
print("Base set size is ", base_size)
num_batches = 100000
batch_size = int(base_size / num_batches)
f = open("/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_ids_"+str(batch_size)+".fvecs", "wb")
print("Saving ids")
for i in range(10):
    subset = b[range(batch_size*i, batch_size * (i+1)), :]
    subset = np.ascontiguousarray(subset.astype('float32'))
    D, I = kmeans.index.search(subset, 1)
    print(I)
    f.write(struct.pack('i', batch_size))
    for j in range(I.shape[0]):
        f.write(struct.pack('f', I[j][0]))
f.close()


