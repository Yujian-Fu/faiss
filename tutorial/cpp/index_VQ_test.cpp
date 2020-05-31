#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_set>
#include <sys/resource.h>

#include "index_VQ.h"

using namespace bslib;
typedef faiss::Index::idx_t idx_t;

int main(){
    const char * path_gt = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/gnd/idx_1000M.ivecs";
    const char * path_query = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    const char * path_base = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs";
    const char * path_learn = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    
    const char * path_centroids = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/centroids_sift1b.fvecs";
    const char * path_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/PQ16.pq";
    const char * path_norm_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/PQ16_NORM.pq";
    
    const char * path_idxs = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/idxs.ivecs";


    const char * path_index;

    size_t ngt = 1000;
    size_t nq = 10000;
    size_t bytes_per_codes = 16;
    size_t nbits_per_idx = 8;
    bool use_quantized_distance = true;
    size_t nc = 993127;
    size_t max_group_size = 100000;
    size_t nt = 10000000;
    //size_t nsubt = 65536;
    size_t nsubt = 10000;
    size_t nb = 1000000000;
    size_t k = 1;

    //const uint32_t batch_size = 1000000;
    //const size_t nbatches = nb / batch_size;
    struct rusage r_usage;

    const uint32_t batch_size = 1000;
    const size_t nbatches = 2;

    if (use_quantized_distance)
        path_index = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/PQ16_quantized.index";
    else
        path_index = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/PQ16.index";
    
    size_t dimension = 128;


    std::cout << "Loading groundtruth from " << path_gt << std::endl;
    
    //Load Groundtruth
    std::vector<uint32_t> groundtruth(nq * ngt);
    {
        std::ifstream gt_input(path_gt, std::ios::binary);
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, true);
    }

    //Load Query
    std::cout << "Loading queries from " << path_query << std::endl;
    std::vector<float> query(nq * dimension);
    {
        std::ifstream query_input(path_query, std::ios::binary);
        readXvecFvec<uint8_t>(query_input, query.data(), dimension, nq, true);
    }

    getrusage(RUSAGE_SELF,&r_usage);
    // Print the maximum resident set size used (in kilobytes).
    std::cout << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss;

    //Initialize the index
    
    BS_LIB * index = new BS_LIB(dimension, nc, bytes_per_codes, nbits_per_idx, use_quantized_distance, max_group_size);
    index->build_quantizer(path_centroids);


    //Train the PQ
    if (exists(path_pq) && (!use_quantized_distance || exists(path_norm_pq))){
        std::cout << "Loading Reidual PQ codebook from " << path_pq << std::endl;
        if (index->pq) delete index->pq;
        index->pq = faiss::read_ProductQuantizer(path_pq);

        if (use_quantized_distance){
            std::cout << "Loading Norm PQ codebook from " << path_norm_pq << std::endl;
            if (index->norm_pq) delete index->norm_pq;
            index->norm_pq = faiss::read_ProductQuantizer(path_norm_pq);
        }
    }
    else{
        //Load learn Set
        bool train_pq = exists(path_pq) ? false : true;
        bool train_norm_pq = (exists(path_norm_pq) || (!use_quantized_distance)) ? false : true;
        
        std::vector<float> LearnSet(nt * dimension);
        {
            std::cout << "Loading Learn Set" << std::endl;
            std::ifstream learn_input(path_learn, std::ios::binary);
            readXvecFvec<uint8_t>(learn_input, LearnSet.data(), dimension, nt, true);
        }

        std::cout << "Randomly select subset for training" << std::endl;
        std::vector<float> LearnSubset(nsubt * dimension);
        random_subset(LearnSet.data(), LearnSubset.data(), dimension, nt, nsubt);
        PrintVector<float>(LearnSubset.data(), dimension);

        std::cout << "Training PQ codebooks " << std::endl;
        index->train_pq(nsubt, LearnSubset.data(), train_pq, train_norm_pq);

        if (train_pq){
            std::cout << "Saving Redisual PQ codebook to " << path_pq << std::endl;
            faiss::write_ProductQuantizer(index->pq, path_pq);
        }
        if (train_norm_pq){
            std::cout << "Saving Norm PQ codebook to " << path_norm_pq << std::endl;
            faiss::write_ProductQuantizer(index->norm_pq, path_norm_pq);
        }
    }

    getrusage(RUSAGE_SELF,&r_usage);
    // Print the maximum resident set size used (in kilobytes).
    // printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);
    std::cout << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss;

    //Assign all base vectors
    if (!exists(path_idxs)){
        std::cout << "Assigning all base vectors " << std::endl;
        StopW stopw = StopW();

        std::ifstream input (path_base, std::ios::binary);
        std::ofstream output (path_idxs, std::ios::binary);

        std::vector <float> batch(batch_size * dimension);
        std::vector <idx_t> idxs(batch_size);

        for (size_t i = 0; i < nbatches; i++){


            readXvecFvec<uint8_t>(input, batch.data(), dimension, batch_size, true);
            index->assign(batch_size, batch.data(), idxs.data());

            output.write((char *) & batch_size, sizeof(uint32_t));
            output.write((char *) idxs.data(), batch_size * sizeof(idx_t));
            std::cout << " [ " << stopw.getElapsedTimeMicro() / 1000000 << "s ] in " << i << " / " << nbatches << std::endl;
            stopw.reset();
        }
    }

    //Construct the index
    if (exists(path_index)){
        std::cout << "Loading index from " << path_index << std::endl;
        index->read(path_index);
    }
    else{
        std::cout << "Constructing the index " << std::endl;
        std::ifstream base_input(path_base, std::ios::binary);
        std::ifstream idx_input(path_idxs, std::ios::binary);


        std::vector<float> batch(batch_size * dimension);
        std::vector<idx_t> quantization_ids(batch_size);
        std::vector<idx_t> origin_ids(batch_size);

        for (size_t b = 0; b < nbatches; b++){
            readXvec<idx_t>(idx_input, quantization_ids.data(), batch_size, 1);
            readXvecFvec<uint8_t>(base_input, batch.data(), dimension, batch_size);

            for (size_t i = 0; i < batch_size; i++){
                origin_ids[i] = batch_size * b + i;
            }

            index->add_batch(batch_size, batch.data(), origin_ids.data(), quantization_ids.data());
        }

        std::cout << "Computing the centroid norms " << std::endl;
        index->compute_centroid_norms();

        std::cout << "Saving index to " << path_index << std::endl;
        index->write(path_index);
    }

    getrusage(RUSAGE_SELF,&r_usage);
    // Print the maximum resident set size used (in kilobytes).
    // printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);
    std::cout << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss;
    /*
    //Search
    std::cout << "Start Searching " << std::endl;
    float distances[nq * k];
    faiss::Index::idx_t labels[nq * k];
    size_t correct = 0;
    StopW stopw = StopW();
    index->search(nq, k, query.data(), distances, labels);
    const float time_us_per_query = stopw.getElapsedTimeMicro() / nq;

    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt;


        for (size_t j = 0; j < k; j++)
            gt.insert(groundtruth[ngt * i + j]);

        assert (gt.size() == k);
        for (size_t j = 0; j < k; j++){
            if (gt.count(labels[i * nq + j]))
                correct++;
        }
    }

    std::cout << "Recall@" << k << " : " << correct / (nq * k) << std::endl;
    std::cout << "Time per query: " << time_us_per_query << " us" << std::endl; 
    */
}
