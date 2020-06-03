#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_set>
#include <sys/resource.h>

#include "index_VQ_VQ.h"

using namespace bslib_VQ_VQ;
typedef faiss::Index::idx_t idx_t;

int main(){
    const char * path_gt = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/gnd/idx_1000M.ivecs";
    const char * path_query = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    const char * path_base = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs";
    const char * path_learn = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    
    const char * path_centroids = "";
    const char * path_subcentroids = "";
    const char * path_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/PQ16.pq";
    const char * path_norm_pq = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/PQ16_NORM.pq";
    
    const char * path_idxs = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/precomputed_idxs_level1.ivecs";
    const char * path_sub_idxs = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/precomputed_idxs_level2.ivecs";

    const char * path_index = "/home/y/yujianfu/ivf-hnsw/models_VQ_VQ/SIFT1B/PQ16_quantized.index";

    //Part 1:
    size_t bytes_per_codes = 16;
    size_t nbits_per_idx = 8;
    size_t nt = 10000000;
    size_t nsubt = 65536;
    size_t dimension = 128;

    //Part 2:
    size_t nc1;
    size_t nc2;

    //Part 3:
    size_t nb = 1000000000;
    const uint32_t batch_size = 1000000;
    const size_t nbatches = nb / batch_size;

    //Part 4:
    size_t ngt = 1000;
    size_t nq = 10000;
    size_t k = 1;
    size_t max_vectors = 10000;
    size_t nprobe1 = 20;
    size_t nprobe2 = 100;

    std::cout << "Now starting the indexing process " << std::endl;
    struct rusage r_usage;
    getrusage(RUSAGE_SELF,&r_usage);
    std::cout << std::endl << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;

    //Initialize the index
    std::cout << "Initializing the index " << std::endl;
    BS_LIB_VQ_VQ * index = new BS_LIB_VQ_VQ(dimension, nc1, nc2, bytes_per_codes, nbits_per_idx);
    index->build_quantizer(path_centroids, path_subcentroids);

    //Train the PQ for compression
    if (exists(path_pq)){
        std::cout << "Loading residual PQ codebook from " << path_pq << std::endl;
        if (index->pq) delete index->pq;
        index->pq = faiss::read_ProductQuantizer(path_pq);
    }
    else{
        //Load learn set
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
        index->train_pq(nsubt, LearnSubset.data());

        std::cout << "Saving residual PQ codebook to " << path_pq << std::endl;
        faiss::write_ProductQuantizer(index->pq, path_pq);
    }

    getrusage(RUSAGE_SELF,&r_usage);
    // Print the maximum resident set size used (in kilobytes).
    // printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);
    std::cout << std::endl << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;

    StopW stopw = StopW();
    //Assign all base vectors
    if (! (exists(path_idxs) && exists(path_sub_idxs))){
        std::cout << "Assigning all base vectors " << std::endl;
        std::ifstream input (path_base, std::ios::binary);
        std::ofstream output1 (path_idxs, std::ios::binary);
        std::ofstream output2 (path_sub_idxs, std::ios::binary);

        std::vector <float> batch(batch_size * dimension);
        std::vector<uint32_t> save_idxs(batch_size);
        std::vector<uint32_t> save_sub_idxs(batch_size);
        std::vector <idx_t> idxs(batch_size);
        std::vector<idx_t> sub_idxs(batch_size);

        for (size_t i = 0; i < nbatches; i++){
            readXvecFvec<uint8_t>(input, batch.data(), dimension, batch_size);
            index->assign(batch_size, batch.data(), idxs.data(), sub_idxs.data());

            for (size_t j = 0; j < batch_size; j++){
                save_idxs[j] = idxs[j];
                save_sub_idxs[j] = sub_idxs[j];
            }
            output1.write((char *) & batch_size, sizeof(uint32_t));
            output1.write((char *) save_idxs.data(), batch_size * sizeof(uint32_t));
            output2.write((char *) & batch_size, sizeof(uint32_t));
            output2.write((char *) save_sub_idxs.data(), batch_size * sizeof(uint32_t));
        }
    }

    if (exists(path_index)){
        std::cout << "Loading index from " << path_index << std::endl;
        index->read(path_index);
    }
    else{
        std::cout << "Constructing the index " << std::endl;
        std::ifstream base_input(path_base, std::ios::binary);
        std::ifstream idx_input(path_idxs, std::ios::binary);
        std::ifstream sub_idx_input(path_sub_idxs, std::ios::binary);
        
        std::vector<float> batch(batch_size * dimension);
        std::vector<idx_t> idxs(batch_size);
        std::vector<idx_t> sub_idxs(batch_size);
        std::vector<uint32_t> read_idxs(batch_size);
        std::vector<uint32_t> read_sub_idxs(batch_size);
        std::vector<idx_t> origin_ids(batch_size);

        for (size_t b = 0; b < nbatches; b++){
            readXvec<uint32_t>(idx_input, read_idxs.data(), dimension, batch_size, true);
            readXvec<uint32_t> (sub_idx_input, read_sub_idxs.data(), dimension, batch_size, true);
            readXvecFvec<uint8_t> (base_input, batch.data(), dimension, batch_size);

            for (size_t i = 0; i < batch_size; i++){
                idxs[i] = read_idxs[i];
                sub_idxs[i] = read_sub_idxs[i];
            }

            for (size_t i = 0; i < batch_size; i++){
                origin_ids[i] = batch_size * b + i;
            }

            index->add_batch(batch_size, batch.data(), origin_ids.data(), idxs.data(), sub_idxs.data());
            std::cout << " [ " << stopw.getElapsedTimeMicro() / 1000000 << "s ] in " << b << " / " << nbatches << std::endl;
        }

        std::cout << "computing the centroid norms " << std::endl;
        index->compute_centroid_norms();

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
    getrusage(RUSAGE_SELF,&r_usage);
    // Print the maximum resident set size used (in kilobytes).
    // printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);
    std::cout << std::endl << "Memory usage: " << r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;

    /*
    //Search
    index->nprobe1 = nprobe1;
    index->nprobe2 = nprobe2;
    index->max_vectors = 10000;

    std::cout << "Starting Searching " << std::endl;
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    size_t correct = 0;
    StopW stopw = StopW();
    index->search(nq, k, query.data(), distances.data(), labels.data());
    const float time_us_per_query = stopw.getElapsedTimeMicro() / nq;

    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt;

        for (size_t j = 0; j < k; j++)
            gt.insert(groundtruth[ngt * i + j]);
        
        assert (gt.size() == k);
        for (size_t j = 0; j < k; j++){
            if (gt.count(labels[i * nq + j]))
                correct ++;
        }
    }

    std::cout << "Recall@ " << k << " : " << correct / (nq * k) << std::endl;
    std::cout << "Time per query " << time_us_per_query << " us" << std::endl;

    */

}