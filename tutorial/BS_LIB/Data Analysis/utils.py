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

def ivecs_write(fname, dataset):
    dimenison = dataset.shape[1]
    file = open(fname, "wb")
    for i in range(dataset.shape[0]):
        file.write(struct.pack('i', dimenison))
        for j in range(dimenison):
            file.write(struct.pack('i', dataset[i][j]))
    file.close()
    

#use https://github.com/j-bac/scikit-dimension
def compute_LID(dataset):
    return two_NN.twonn_dimension(dataset)

#use https://github.com/gregversteeg/NPEET
def compute_entropy(dataset):
    return ee.entropy(dataset)

def get_dataset_path_real():
    dataset_path = "/home/y/yujianfu/ivf-hnsw/data/"
    path_list = []
    real_dataset_list = ["SIFT", "GIST", "DEEP"]
    size_list = ["10K"]
    for dataset in real_dataset_list:
        for size in size_list:
            sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + size +"_base.fvecs"
            path_list.append(sample_dataset_file)

    return path_list


def get_dataset_path_gaussian():
    dataset_path = "/home/y/yujianfu/ivf-hnsw/data/"
    path_list = []
    size_list = ["10K"]
    dataset = "Gaussian"
    for dimension in range(100, 1100, 300):
        for size in size_list:
            sample_size = int(size.split("K")[0]) 
            sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + str(sample_size) + "K_" + str(dimension) + "_base.fvecs"
            path_list.append(sample_dataset_file)
    
    return path_list 

def get_dataset_path_random():
    dataset_path = "/home/y/yujianfu/ivf-hnsw/data/"
    path_list = []
    size_list = ["10K"]
    dataset = "Random"
    for dimension in range(100, 1100, 300):
        for size in size_list:
            sample_size = int(size.split("K")[0]) 
            sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + str(sample_size) + "K_" + str(dimension) + "_base.fvecs"
            path_list.append(sample_dataset_file)
    
    return path_list