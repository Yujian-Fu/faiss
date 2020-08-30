#include "../PQ_quantizer.h"
#include "../parameters/parameters_millions_PQ.h"
#include "../utils/utils.h"

using namespace bslib;

int main(){
    size_t dimension = 128;

    nb = 100000;
    std::vector<float> base_set(dimension * nb);
    std::vector<float> train_set(dimension * train_size);
    std::vector<idx_t> train_ids(train_size, 0);
    const std::string path_learn =     "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/sift_learn.fvecs";
    const std::string path_base =      "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/sift_base.fvecs";
    const std::string path_record_PQ =  "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/recording/VQ_PQ.txt";
    std::ofstream PQ_record;
    PQ_record.open(path_record_PQ, std::ios::binary);

    std::ifstream learn_file(path_learn, std::ios::binary);
    std::ifstream base_file(path_base, std::ios::binary);

    readXvec<float>(learn_file, train_set.data(), dimension, train_size);
    readXvec<float> (base_file, base_set.data(), dimension, nb);

    for (size_t M = 2; M <= 4; M += 2){
        for (size_t nbits = 4; nbits <= 8; nbits += 2){

    PQ_quantizer pq (dimension, 1, M, nbits);

    pq.build_centroids(train_set.data(), train_size, train_ids.data());
    PQ_record << "PQ structure: M: " << M << " nbits: " << nbits << std::endl;
    std::cout << "PQ structure: M: " << M << " nbits: " << nbits << std::endl;

    
    std::cout << "The nc of PQ layer: " << pq.nc << std::endl;
    time_recorder Trecorder = time_recorder();
    size_t search_space = 2000;
    for (size_t nb = 100; nb <= search_space; nb += 100){
        for (size_t pq_keep_space = 10; pq_keep_space <= 100; pq_keep_space+=10){
            Trecorder.reset();
            for (size_t i = 0; i < nb; i++){
                std::vector<idx_t> query_ids(pq_keep_space, 0);
                std::vector<float> query_dists(pq_keep_space, 0);
                pq.search_in_group(1, base_set.data() + i * dimension, train_ids.data(), query_dists.data(), query_ids.data(), pq_keep_space);
                //std::cout << "Finished " << i << std::endl;
            }
            float time_consumption = Trecorder.get_time_usage();
            PQ_record << "Time for " << nb << " PQ queries with search para: " << pq_keep_space << " " << time_consumption << std::endl;
            std::cout << "Time for " << nb << " PQ queries with search para: " << pq_keep_space << " " << time_consumption << std::endl;
        }
        PQ_record << std::endl;
        std::cout << "\n\n";

    }

    /*
    size_t sum_centroids = pq.nc;
    std::vector<float> all_centroids(sum_centroids * dimension);
    for (size_t i = 0; i < sum_centroids; i++){
        std::vector<float> each_centroid(dimension);
        pq.compute_final_centroid(i, all_centroids.data() + i * dimension);
    }
    faiss::IndexFlatL2 index(dimension);
    index.add(sum_centroids, all_centroids.data());
    std::vector<float> index_dist(train_size * keep_space);
    std::vector<idx_t> index_labels(train_size * keep_space);
    Trecorder.reset();
    index.search(train_size, train_set.data(), keep_space, index_dist.data(), index_labels.data());
    Trecorder.print_time_usage("The time for search in L2 layer");

    size_t correct = 0;
    for (size_t i = 0; i < train_size * keep_space; i++){
        if (index_labels[i] == train_next_ids[i]){
            correct ++;
        }
        else{
            std::cout << i << " " << index_labels[i] << " " << train_next_ids[i] << " " << index_dist[i] << " " << train_dists[i] << " ";
        }
    }
    std::cout << std::endl;
    std::cout << "The correct num is: " << correct << std::endl;
    */

    }
    }
}
