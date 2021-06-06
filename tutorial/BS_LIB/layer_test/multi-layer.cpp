#include <fstream>

#include "../bslib_index.h"

//Change the base_data_type for billion and million scale datasets
#include "../parameters/parameters_billion/parameters_VQ_VQ.h"

using namespace bslib;

int main(){

/*Prepare the work space*/
    PrepareFolder((char *) folder_model.c_str());
    PrepareFolder((char *) (std::string(folder_model)+"/" + dataset).c_str());
    std::cout << "Preparing work space: " << folder_model << std::endl;

    //For recording 
    std::ofstream record_file;
    std::ofstream qps_record_file;
    if (is_recording){
        record_file.open(path_record, std::ios::app);
        qps_record_file.open(path_speed_record, std::ios::binary);

        qps_record_file << "The num of centroid is: " << std::to_string(ncentroids[0]) << std::endl;
        time_t now = time(0);
        char* dt = ctime(&now);
        record_file << std::endl << "The time now is " << dt << std::endl;
        record_file << "The memory usage format is ixrss, isrss, idrss, maxrss" << std::endl;
        record_file << "Now starting the indexing process " << std::endl;
    }

    Bslib_Index index = Bslib_Index(dimension, layers, index_type, use_reranking, saving_index, use_norm_quantization, is_recording,
    use_HNSW_VQ, use_HNSW_group, use_OPQ, use_train_selector, train_size, M_PQ, nbits);

    index.build_index(M_PQ, path_learn, path_groups, path_labels, path_quantizers, VQ_layers,
    PQ_layers, path_OPQ, ncentroids, M_HNSW, efConstruction, efSearch, M_PQ_layer, nbits_PQ_layer, num_train, OPQ_train_size, selector_train_size, selector_group_size, record_file);

    exit(0);
    index.assign_vectors(path_ids, path_base, batch_size, nbatches, record_file);

    index.train_pq_quantizer(path_pq, path_pq_norm, M_norm_PQ, path_learn, PQ_train_size, record_file);

    index.load_index(path_index, path_ids, path_base, path_base_norm, path_centroid_norm, batch_size, nbatches, nb, record_file);

    index.index_statistic();

    index.query_test(num_search_paras, num_recall, nq, ngt, max_vectors, result_k, keep_space, reranking_space, record_file, 
                    qps_record_file, search_mode, path_base, path_gt, path_query);

}