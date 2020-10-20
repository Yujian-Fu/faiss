# Datasets to be evaluated: 
# SIFT, GIST, DEEP, Gaussian, Random
# Size:
# 10K, 100K, 1M, 10M

import numpy as np 
import random 
import utils 
import faiss

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
        index = random.sample(range(real_dataset.shape[0]), sample_size * 1000)
        sample_dataset = real_dataset[index, :]
        utils.fvecs_write(sample_dataset_file, sample_dataset)
'''

query_size = 1000

size_list = ["10K", "100K", "1M", "10M"]
dataset = "Gaussian"
mu = 10
sigma = 0.1
for dimension in range(500, 600, 300):

    for size in size_list:
        if "K" in size:
            sample_size = int(size.split("K")[0]) * 1000
        elif "M" in size:
            sample_size = int(size.split("M")[0]) * 1000000

        sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + size + "_" + str(dimension) + "_base.fvecs"
        print("Generating dataset to ", sample_dataset_file)
        sample_dataset = np.random.normal(mu, sigma, (sample_size, dimension))
        utils.fvecs_write(sample_dataset_file, sample_dataset)

        sample_query_file = dataset_path + "analysis/" + dataset + "_" + size + "_" + str(dimension) + "_query.fvecs"
        print("Generating query dataset to ", sample_query_file)
        query_dataset = np.random.normal(mu, sigma, (query_size, dimension))
        utils.fvecs_write(sample_query_file, query_dataset)

        sample_ID_file = dataset_path + "analysis/" + dataset + "_" + size + "_" + str(dimension) + "_gt.ivecs"
        index = faiss.IndexFlatL2(dimension)
        index.add(sample_size, sample_dataset)
        dis, ID = index.search(query_dataset, 100)
        utils.ivecs_write(sample_ID_file, ID)



dataset = "Random"
for dimension in range(500, 600, 300):

    for size in size_list:
        if "K" in size:
            sample_size = int(size.split("K")[0]) * 1000
        elif "M" in size:
            sample_size = int(size.split("M")[0]) * 1000000

        sample_dataset_file = dataset_path + "analysis/" + dataset + "_" + str(sample_size) + "K_" + str(dimension) + "_base.fvecs"
        print("Generating dataset to ", sample_dataset_file)
        sample_dataset = np.random.randint(0, 100, (sample_size, dimension))
        utils.fvecs_write(sample_dataset_file, sample_dataset)


        sample_query_file = dataset_path + "analysis/" + dataset + "_" + str(sample_size) + "K_" + str(dimension) + "_query.fvecs"
        query_dataset = np.random.randint(0, 100, (query_size, dimension))
        print("Generating query dataset to ", sample_query_file)
        utils.fvecs_write(sample_query_file, query_dataset)

        sample_ID_file = dataset_path + "analysis/" + dataset + "_" + str(sample_size) + "K_" + str(dimension) + "_gt.ivecs"
        index = faiss.IndexFlatL2(dimension)
        index.add(sample_size, sample_dataset)
        dis, ID = index.search(query_dataset, 100)
        utils.ivecs_write(sample_ID_file, ID)
