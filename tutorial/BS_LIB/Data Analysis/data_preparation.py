# Datasets to be evaluated: 
# SIFT, GIST, DEEP, Gaussian, Random
# Size:
# 10K, 100K, 1M, 10M

import numpy as np 
import random 
import utils 


dataset_path = "/home/y/yujianfu/ivf-hnsw/data/"
'''
real_dataset_list = ["SIFT", "GIST", "DEEP"]
size_list = ["10K", "100K"]


for dataset in real_dataset_list:
    for size in size_list:
        dataset_file = dataset_path + dataset + "1M/" + dataset + "1M_base.fvecs" 
        sample_dataset_file = dataset_path + "analysis/" + dataset + size +"_base" + ".fvecs"
        print("read dataset from ", dataset_file)
        real_dataset = utils.fvecs_read(dataset_file)
        sample_size = int(size.split("K")[0])
        print(real_dataset.shape)
        print(sample_size)
        index = random.sample(range(real_dataset.shape[0]), sample_size * 1000)
        sample_dataset = real_dataset[index, :]
        utils.fvecs_write(sample_dataset_file, sample_dataset)
'''

size_list = ["10K", "100K", "1000K", "10000K"]

'''
dataset = "Gaussian"
mu = 10
sigma = 0.1
for dimension in range(100, 1000, 300):
    query_dataset_file = dataset_path + "analysis/" + dataset + "_"  + str(dimension) + "_query.fvecs"
    sample_dataset = np.zeros((1000, dimension))
    for i in range(1000):
        sample_dataset[i, :] = np.random.normal(mu, sigma, dimension)
    utils.fvecs_write(query_dataset_file, sample_dataset)

    for size in size_list:
        sample_size = int(size.split("K")[0]) 
        sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + str(sample_size) + "K_" + str(dimension) + "_base.fvecs"
        print("Generating dataset to ", sample_dataset_file)
        sample_dataset = np.zeros((sample_size * 1000, dimension))
        for i in range(sample_size * 1000):
            sample_dataset[i, :] = np.random.normal(mu, sigma, dimension)
        utils.fvecs_write(sample_dataset_file, sample_dataset)
'''

dataset = "Random"

for dimension in range(100, 1000, 300):
    query_dataset_file = dataset_path + "analysis/" + dataset + "_"  + str(dimension) + "_query.fvecs"
    sample_dataset = np.zeros((1000, dimension))
    for i in range(1000):
        sample_dataset[i, :] = np.random.randint(0, 100, (1, dimension))
    utils.fvecs_write(query_dataset_file, sample_dataset)

    for size in size_list:
        sample_size = int(size.split("K")[0])
        sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + str(sample_size) + "K_" + str(dimension) + "_base.fvecs"
        print("Generating dataset to ", sample_dataset_file)
        sample_dataset = np.zeros((sample_size* 1000, dimension))
        for i in range(sample_size* 1000):
            sample_dataset[i, :] = np.random.randint(0, 100, (1, dimension))
        utils.fvecs_write(sample_dataset_file, sample_dataset)


