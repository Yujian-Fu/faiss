#include <cstdio>
#include <iostream>
#include <string>


const bool use_reranking = false;
const bool use_HNSW_VQ = false;
const bool use_norm_quantization = false;
const bool use_dynamic_reranking = false;
const bool use_OPQ = false;
const bool use_parallel_indexing = false;
const bool use_train_selector = false;
const bool use_HNSW_group = false;

const bool is_recording = true;
const bool saving_index = true;

const std::string dataset = "SIFT1B";
const std::string path_folder = "/home/y/yujianfu/ivf-hnsw/";

//For train PQ
const size_t M_PQ = 8;
const size_t M_norm_PQ = 1;
const size_t nbits = 8; //Or 16
const size_t dimension = 128;

const size_t train_size = 100000000; //This is the size of train set
const size_t OPQ_train_size = 100000000;
const size_t selector_train_size = 10000000;
const size_t selector_group_size = 2000;

const size_t PQ_train_size = 100000000;
const size_t nb = 1000000000;
const size_t nbatches = 100; //100
const uint32_t batch_size =  nb / nbatches;


const size_t ngt = 1000;
const size_t nq = 10000;
const size_t num_search_paras = 10;
const size_t num_recall = 2;

const size_t result_k[num_recall] = {1, 10};
const size_t max_vectors[num_search_paras] = {10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};
const size_t reranking_space[num_recall] = {10, 150};
const std::string search_mode = "non parallel";

