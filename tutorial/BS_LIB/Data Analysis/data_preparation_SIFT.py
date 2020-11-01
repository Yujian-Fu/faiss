import numpy as np 
import random 
import utils 
import faiss
import os

dataset_path = "/home/y/yujianfu/ivf-hnsw/data/"

size_list = ["100K"]

billion_dataset_base = dataset_path + "SIFT1B/" + "bigann_base.bvecs" 
billion_dataset_learn = dataset_path + "SIFT1B/" + "bigann_learn.bvecs" 
billion_dataset_query = dataset_path + "SIFT1B/" + "bigann_query.bvecs"
ngt = 100
nq = 1000
dimension = 128


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

for size in size_list:
    folder_path = dataset_path + "SIFT" + size
    create_dir_not_exist(dataset_path + "SIFT" + size)
    
    print("read dataset from ", billion_dataset_base)
    real_base_dataset = utils.bvecs_read(billion_dataset_base)
    if "K" in size:
        sample_size = int(size.split("K")[0]) * 1000
    if "M" in size:
        sample_size = int(size.split("M")[0]) * 1000000
    print("The file size is: ", sample_size)
    
    index = random.sample(range(real_base_dataset.shape[0]), sample_size)
    base_dataset = real_base_dataset[index, :]
    base_dataset_file = folder_path + "/SIFT" + size +"_base" + ".fvecs"
    utils.fvecs_write(base_dataset_file, base_dataset)
    print("Write base file")

    real_learn_dataset = utils.bvecs_read(billion_dataset_learn)
    index = random.sample(range(real_learn_dataset.shape[0]), int(sample_size / 10))
    learn_dataset = real_learn_dataset[index, :]
    learn_dataset_file = folder_path + "/SIFT" + size +"_learn" + ".fvecs"
    utils.fvecs_write(learn_dataset, learn_dataset_file)
    print("Write learn file")

    real_query_dataset = utils.bvecs_read(billion_dataset_query)
    index = random.sample(range(real_query_dataset.shape[0]), nq)
    query_dataset = real_query_dataset[index, :]
    query_dataset_file = folder_path + "/SIFT" + size + "_query" + ".fvecs"
    utils.fvecs_write(query_dataset, query_dataset_file)
    print("Write query file")

    assert(base_dataset.shape[1] == dimension)
    index = faiss.IndexFlatL2(dimension)
    index.add(base_dataset)
    D, I = index.search(query_dataset, ngt)
    groundtruth_file = folder_path + "/SIFT" + size + "_groundtruth" + ".fvecs"
    utils.fvecs_write(I, groundtruth_file)
    print("Write GT file")


