# Datasets to be evaluated: 
# SIFT, GIST, DEEP, Gaussian, Random
# Size:
# 10K, 100K, 1M, 10M

import numpy as np 
import random 
import utils 

real_dataset_list = ["SIFT", "GIST", "DEEP"]
size_list = ["10K", "100K"]

dataset_path = "/home/y/yujianfu/ivf-hnsw/data/"

for dataset in real_dataset_list:
    for size in size_list:
        dataset_file = dataset_path + dataset + "1M/" + dataset + "1M_base.fvecs" 
        sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + size + ".fvecs"
        print("read dataset from ", dataset_file)
        real_dataset = utils.fvecs_read(dataset_file)
        sample_size = int(size.split("K")[0])
        print(real_dataset.shape)
        print(sample_size)
        index = random.sample(range(real_dataset.shape[0]), sample_size * 1000)
        sample_dataset = real_dataset[index, :]
        utils.fvecs_write(sample_dataset_file, sample_dataset)


size_list = ["10K", "100K", "1000K", "10000K"]

dataset = "Gaussian"
mu = 10
sigma = 0.1
for size in size_list:
    for dimension in range(100, 1000, 100):
        sample_size = float(size.split("K")[0])
        sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + sample_size + "_" + dimension + ".fvecs"
        dataset_file = dataset_path + dataset + size + dimension
        sample_dataset = np.zeros((sample_size, dimension))
        for i in range(sample_size):
            sample_dataset[i, :] = np.random.normal(mu, sigma, dimension)
        utils.fvecs_write(sample_dataset_file, sample_dataset)


dataset = "Random"
for size in size_list:
    for dimension in range(100, 1000, 100):
        sample_size = float(size.split("K")[0])
        sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + sample_size + "_" + dimension + ".fvecs"
        dataset_file = dataset_path + dataset + size + dimension
        sample_dataset = np.zeros((sample_size, dimension))
        for i in range(sample_size):
            sample_dataset[i, :] = np.random.randint(0, 100, (sample_size, dimension))
        utils.fvecs_write(sample_dataset_file, sample_dataset)


