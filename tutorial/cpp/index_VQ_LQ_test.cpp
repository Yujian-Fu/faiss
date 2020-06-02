#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_set>
#include <sys/resource.h>

#include "index_VQ_LQ.h"

using namespace bslib_VQ_LQ;
typedef faiss::Index::idx_t idx_t;

int main(){
    const char * path_gt = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/gnd/idx_1000M.ivecs";
    const char * path_query = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    const char * path_base = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs";
    const char * path_learn = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    const char * path_idxs = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/precomputed_idxs_sift1b.ivecs";

    const char * path_centroids = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/centroids_sift1b.fvecs";
    const char * path_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/PQ16.pq";
    const char * path_norm_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ_LQ/SIFT1B/PQ16_NORM.pq";
    
    const char * path_index;

    size_t ngt = 1000;
    size_t nq = 10000;
    size_t bytes_per_codes = 16;
    size_t nbits_per_idx = 8;
    bool use_quantized_distance = true;
    size_t nc = 993127;
    size_t max_group_size = 100000;
    size_t nt = 10000000;
    size_t nsubt = 65536;
    size_t nb = 1000000000;
    size_t k = 1;
    size_t groups_per_iter = 250000;
    size_t dimension = 128;

    const uint32_t batch_size = 1000000;
    const size_t nbatches = nb / batch_size;
    struct rusage r_usage;

    if (use_quantized_distance)
        path_index = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/PQ16_quantized.index";
    else
        path_index = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1B/PQ16.index";
    

    getrusage(RUSAGE_SELF,&r_usage);
    // Print the maximum resident set size used (in kilobytes).
    std::cout << std::endl << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;


    BS_LIB_VQ_LQ * index = new BS_LIB_VQ_LQ(dimension, nc, bytes_per_codes, nbits_per_idx, nsubt, use_quantized_distance);
    index->build_quantizer(path_centroids);

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
            readXvecFvec<uint8_t>(learn_input, LearnSet.data(), dimension, nt, true, true);
        }
        std::cout << "Randomly select subset for training" << std::endl;
        std::vector<float> LearnSubset(nsubt * dimension);
        random_subset(LearnSet.data(), LearnSubset.data(), dimension, nt, nsubt);
        PrintVector<float>(LearnSubset.data(), dimension);

        std::cout << "Training PQ codebooks " << std::endl;
        index->train_pq(nsubt, LearnSubset.data(), train_pq, train_norm_pq);

        if (train_pq){
            std::cout << "Saving Residual PQ codebook to " << path_pq << std::endl;
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
    std::cout << std::endl << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;

    StopW stopw = StopW();
    if (!exists(path_idxs)){
        std::cout << "Assigning all base vectors " << std::endl;

        std::ifstream input (path_base, std::ios::binary);
        std::ofstream output (path_idxs, std::ios::binary);

        std::vector <float> batch(batch_size * dimension);
        std::vector <idx_t> idxs(batch_size);

        for (size_t i = 0; i < nbatches; i++){
            readXvecFvec<uint8_t>(input, batch.data(), dimension, batch_size);
            index->assign(batch_size, batch.data(), idxs.data());

            output.write((char *) & batch_size, sizeof(uint32_t));
            output.write((char *) idxs.data(), batch_size * sizeof(uint32_t));
            std::cout << " [ " << stopw.getElapsedTimeMicro() / 1000000 << "s ] in " << i << " / " << nbatches << std::endl;
        }
    }

    //Build the index
    if(exists(path_index)){
        std::cout << "Loading index from " << path_index << std::endl;
        index->read(path_index);
    }
    else{
        std::cout << "Adding groups to index " << std::endl;
        StopW stopw = StopW();

        std::vector<uint8_t> batch(batch_size * dimension);
        std::vector<idx_t> idx_batch(batch_size);
        std::vector<uint32_t> idx_batch_temp(batch_size);

        for (size_t ngroups_added = 0; ngroups_added < nc; ngroups_added += groups_per_iter){
            std::cout << " [ " << stopw.getElapsedTimeMicro() / 1000000 << " s]"
                      << ngroups_added << " / " << nc << std::endl;
            std::vector<std::vector<uint8_t>> data(groups_per_iter);
            std::vector<std::vector<idx_t>> ids(groups_per_iter);

            std::ifstream base_input(path_base, std::ios::binary);
            std::ifstream idx_input(path_idxs, std::ios::binary);

            for (size_t b = 0; b < nbatches; b++){
                readXvec<uint8_t> (base_input, batch.data(), dimension, batch_size);
                readXvec<uint32_t> (idx_input, idx_batch_temp.data(), batch_size, 1);

                for (size_t i = 0; i < batch_size; i++){
                    idx_batch[i] = idx_batch_temp[i];
                }

                for (size_t i = 0; i < batch_size; i++){
                    if(idx_batch[i] < ngroups_added || 
                       idx_batch[i] >= ngroups_added + groups_per_iter)
                        continue;
                    
                    idx_t idx = idx_batch[i] % groups_per_iter;
                    for (size_t j = 0; j < dimension; j++){
                        data[idx].push_back(batch[i * dimension + j]);
                    }
                    ids[idx].push_back(b * batch_size + i);
                }
            }

            if (nc - ngroups_added <= groups_per_iter)
                groups_per_iter = nc - ngroups_added;
            
            size_t j = 0;
#pragma omp parallel for
            for (size_t i = 0; i < groups_per_iter; i++){
                #pragma omp critical
                {
                    if (j % 1000 == 0){
                        std::cout << " [ " << stopw.getElapsedTimeMicro() / 1000000 << " s] "
                                  << (100. * (ngroups_added + j)) / nc << "%" << std::endl;
                    }
                    j++;
                }
                const size_t group_size = ids[i].size();
                std::vector<float> group_data(group_size * dimension);
                for (size_t k = 0; k < group_size * dimension ; k++)
                    group_data[k] = 1.0 * data[i][k];
                
                index->add_group(ngroups_added + i, group_size, group_data.data(), ids[i].data());
            }
        }
        std::cout << "Compputing centroid norms" << std::endl;
        index->compute_centroid_norms();
        std::cout << "computing centroid dists" << std::endl;
        index->compute_inter_centroid_dists();

        std::cout << "Saving index to " << path_index << std::endl;
        index->write(path_index);

    }

    std::cout << "Loading groundtruth from " << path_gt << std::endl;
    
    //Load Groundtruth
    std::vector<uint32_t> groundtruth(nq * ngt);
    {
        std::ifstream gt_input(path_gt, std::ios::binary);
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, false, true);
    }

    //Load Query
    std::cout << "Loading queries from " << path_query << std::endl;
    std::vector<float> query(nq * dimension);
    {
        std::ifstream query_input(path_query, std::ios::binary);
        readXvecFvec<uint8_t>(query_input, query.data(), dimension, nq, true, true);
    }

























}