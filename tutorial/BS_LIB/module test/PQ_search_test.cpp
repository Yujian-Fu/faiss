#include "../PQ_quantizer.h"
#include "../parameters/parameters_millions_PQ.h"
#include "../utils/utils.h"

using namespace bslib;

int main(){
    size_t dimension = 128;
    size_t keep_space = 1;
    std::vector<float> base_set(dimension * nb);
    std::vector<float> train_set(dimension * train_size);
    std::vector<idx_t> train_ids(train_size, 0);

    std::ifstream learn_file(path_learn, std::ios::binary);
    std::ifstream base_file(path_base, std::ios::binary);

    readXvec<float>(learn_file, train_set.data(), dimension, train_size);
    readXvec<float> (base_file, base_set.data(), dimension, nb);
    PQ_quantizer pq (dimension, 1, 2, 4);

    pq.build_centroids(train_set.data(), train_size, train_ids.data());
    std::vector<idx_t> train_next_ids(train_size * keep_space, 0);
    std::vector<float> train_dists(train_size * keep_space, 0);
    
    std::cout << "The nc of PQ layer: " << pq.nc << std::endl;
    time_recorder Trecorder = time_recorder();
    Trecorder.reset();
    pq.search_in_group(train_size, train_set.data(), train_ids.data(), train_dists.data(), train_next_ids.data(), keep_space);
    Trecorder.print_time_usage("The time for search in PQ layer: ");

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

    std::vector<size_t> id_set(pq.nc);
    for (size_t i = 0; i < train_size * keep_space; i++){
        id_set[index_labels[i]] ++;
    }
    for (size_t i = 0; i < pq.nc; i++){
        std::cout << id_set[i] << " ";
    }
    std::cout << std::endl;
}
