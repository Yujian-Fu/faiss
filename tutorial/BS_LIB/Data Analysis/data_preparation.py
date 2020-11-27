# Datasets to be evaluated: 
# SIFT, GIST, DEEP, Gaussian, Random
# Size:
# 10K, 100K, 1M, 10M

import numpy as np 
import random 
import utils 
import faiss
import os

#dataset_path = "/home/y/yujianfu/ivf-hnsw/data/"

dataset_path = "/home/yujian/Desktop/extra/Similarity_Search/similarity_search_datasets/data/"
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



size_list = ["100K"]

'''
dataset = "Gaussian"
mu = 10
sigma = 0.1

for dimension in range(128, 257, 128):

    for size in size_list:
        if "K" in size:
            sample_size = int(size.split("K")[0]) * 1000
        elif "M" in size:
            sample_size = int(size.split("M")[0]) * 1000000

        sample_dataset_file = dataset_path + "analysis/" + dataset + size + "_" + str(dimension) + "_base.fvecs"
        print("Generating dataset to ", sample_dataset_file)
        sample_dataset = np.random.normal(mu, sigma, (sample_size, dimension)).astype('float32')
        utils.fvecs_write(sample_dataset_file, sample_dataset)

        sample_query_file = dataset_path + "analysis/" + dataset + size + "_" + str(dimension) + "_query.fvecs"
        print("Generating query dataset to ", sample_query_file)
        query_dataset = np.random.normal(mu, sigma, (query_size, dimension)).astype('float32')
        utils.fvecs_write(sample_query_file, query_dataset)
        
        sample_learn_file = dataset_path + "analysis/" + dataset + size + "_" + str(dimension) + "_learn.fvecs"
        print("Generating train dataset to ", sample_learn_file)
        learn_dataset = np.random.normal(mu, sigma, (int(sample_size / 10), dimension)).astype('float32')
        utils.fvecs_write(sample_learn_file, learn_dataset)

        sample_ID_file = dataset_path + "analysis/" + dataset + size + "_" + str(dimension) + "_groundtruth.ivecs"
        index = faiss.IndexFlatL2(dimension)
        index.add(sample_dataset)
        dis, ID = index.search(query_dataset, 100)
        utils.ivecs_write(sample_ID_file, ID)
'''

dataset = "Random"
random_start = 200
random_end = 300
for dimension in range(128, 129, 128):

    for size in size_list:
        if "K" in size:
            sample_size = int(size.split("K")[0]) * 1000
        elif "M" in size:
            sample_size = int(size.split("M")[0]) * 1000000
        dataset_folder = dataset_path + dataset + size + "_" + str(dimension)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)


        sample_dataset_file = dataset_folder +"/" + dataset + size + "_" + str(dimension) + "_base.fvecs"
        print("Generating dataset to ", sample_dataset_file)
        sample_dataset = np.random.randint(random_start, random_end, (sample_size, dimension)).astype('float32')
        utils.fvecs_write(sample_dataset_file, sample_dataset)

        sample_query_file = dataset_folder +"/" + dataset + size + "_" + str(dimension) + "_query.fvecs"
        query_dataset = np.random.randint(random_start, random_end, (query_size, dimension)).astype('float32')
        print("Generating query dataset to ", sample_query_file)
        utils.fvecs_write(sample_query_file, query_dataset)
        
        sample_learn_file = dataset_folder +"/" + dataset + size + "_" + str(dimension) + "_learn.fvecs"
        print("Generating train dataset to ", sample_learn_file)
        learn_dataset = np.random.randint(random_start, random_end, (int(sample_size), dimension)).astype('float32')
        utils.fvecs_write(sample_learn_file, learn_dataset)

        sample_ID_file = dataset_folder +"/" + dataset + size + "_" + str(dimension) + "_groundtruth.ivecs"
        index = faiss.IndexFlatL2(dimension)
        index.add(sample_dataset)
        dis, ID = index.search(query_dataset, 100)
        utils.ivecs_write(sample_ID_file, ID)

