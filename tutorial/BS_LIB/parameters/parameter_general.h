#include <string>

//const std::string base_path = "/home/y/yujianfu/ivf-hnsw/";
const std::string base_path = "/home/yujian/Desktop/extra/Similarity Search/similarity_search_datasets/";

const std::string dataset = "Random10K_500";
const size_t dimension = 500;


size_t nb = 10000;
const size_t train_size = 1000; //This is the size of train set

const size_t nq = 100;

//For train PQ
const size_t M_PQ = 25;
const size_t M_norm_PQ = 1;
const size_t nbits = 8; //Or 16

const size_t PQ_train_size = train_size;
const size_t OPQ_train_size = 10000;
const size_t selector_train_size = 100000;
const size_t selector_group_size = 2000;

size_t nbatches = 100; //100
uint32_t batch_size = nb / nbatches;
//For searching
const size_t ngt = 100;
const size_t num_recall = 2;

const size_t result_k[num_recall] = {1, 10};
const size_t reranking_space[num_recall] = {10};
const std::string search_mode = "non parallel";

const bool is_recording = true;
const bool use_reranking = false;
const bool use_HNSW_VQ = false;
const bool use_norm_quantization = false;
const bool use_dynamic_reranking = false;
const bool use_OPQ = false;
const bool use_parallel_indexing = true;
const bool use_train_selector = false;

