#Testing the speed of kmeans in Faiss

import numpy as np
import faiss
import time
import random

a = np.memmap('/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs', dtype='uint8', mode='r')
d = a[:4].view('int32')[0]
niter = 20
verbose = True

print("This is the GPU version of Kmeans")
'''
start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 10000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 100
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 100000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 100
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 1000000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 100
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 10000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 1000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 100000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 1000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 1000000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 1000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 100000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 10000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 1000000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 10000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")


start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 10000000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 10000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")
'''

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 20000000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 1000000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

'''
start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 100000000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 10000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")


start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 100000000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 100000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")

start = time.time()
b = a.reshape(-1, d + 4)[:, 4:]
b = b[random.sample(range(b.shape[0]), 100000000), :]
b = np.ascontiguousarray(b.astype('float32'))
ncentroids = 1000000
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(b)
end = time.time()
print("Dataset size: ", b.shape[0], " Ncentroids: ", ncentroids, " : ", end-start, "\n\n")
'''




