import numpy as np 
import math
import scipy
import skdim
import two_NN
import entropy_estimators as ee
import struct

def distance_L2sqr(v1, v2):
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = sum(dist)
    return dist

def distance_L2(v1, v2):
    return math.sqrt(distance_L2sqr(v1, v2))
    
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def bvecs_read(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def fvecs_write(fname, dataset):
    dimension = dataset.shape[1]
    file = open(fname, "wb")
    for i in range(dataset.shape[0]):
        file.write(struct.pack('i', dimension))
        for j in range(dimension):
            file.write(struct.pack('f', dataset[i][j]))
    file.close()

#use https://github.com/j-bac/scikit-dimension
def compute_LID(dataset):
    return two_NN.twonn_dimension(dataset)

#use https://github.com/gregversteeg/NPEET
def compute_entropy(dataset, k = 100):
    return ee.entropy(dataset, k)