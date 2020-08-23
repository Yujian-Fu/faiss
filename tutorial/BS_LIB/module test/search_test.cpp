#include "../PQ_quantizer.h"
#include "../parameters/parameters_millions_PQ.h"
#include "../utils/utils.h"


using namespace bslib;

int main(){
    size_t dimension = 128;
    std::vector<float> base_set(dimension * nb);
    std::vector<float> train_set(dimension * train_size);
    std::vector<idx_t> train_ids(train_size, 0);

    std::ifstream learn_file(path_learn, std::ios::binary);
    std::ifstream base_file(path_base, std::ios::binary);

    readXvec<float>(learn_file, train_set.data(), dimension, train_size);
    readXvec<float> (base_file, base_set.data(), dimension, nb);
    PQ_quantizer pq (dimension, 1, 2, 8);

    pq.build_centroids(train_set.data(), train_size, train_ids.data());
    std::vector<idx_t> train_next_ids(train_size, 0);
    std::vector<float> train_dists(train_size, 0);

    pq.search_in_group(train_size, train_set.data(), train_ids.data(), train_dists.data(), train_next_ids.data(), 10);

    std::cout << "The search result: " << std::endl;
    for (size_t i = 0; i < 100; i++){
        std::cout << train_next_ids[i] << " ";
    }
}
