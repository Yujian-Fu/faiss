#include <fstream>
#include "../bslib_index.h"
/* Important */
//Change the base_data_type and learn_data_type accordingly 
// in bslib_index.h for billion and million scale datasets
/*          */
#include "../parameters/parameters_million/parameters_result.h"
//Baeline: July 10 5:13 SIFT1B_grouping_OPQ 
//Build HNSW: 5:28
//Train PQ: 

/*
Todo:
Components: 

Decreasing the number of centroids/ shrink strategy

HNSW_VQ
OPQ
HNSW_group
Line Quantization

Test:
Search Parameters
1-3 (4)layers
Parameter Tuning
Longer Code Length

Baseline:
Faiss (IMI)
IVFHNSW
(LOPQ)

Check the name of the index files
*/

using namespace bslib;

int main(){

/*Prepare the work space*/
    PrepareFolder((char *) (path_folder + model).c_str());
    PrepareFolder((char *) (path_folder + model+"/" + dataset).c_str());
    std::cout << "Preparing work space: " << path_folder + model << std::endl;
    std::cout << "The centroid configuration is: " << ncentroid_conf << std::endl;
    //For recording 
    std::ofstream record_file;
    std::ofstream qps_record_file;
    
    if (is_recording){
        char hostname[100] = {0};
        if (gethostname(hostname, sizeof(hostname)) < 0){
            perror("Error in get host name");
            exit(0);
        }

        record_file.open(path_record, std::ios::app);
        qps_record_file.open(path_speed_record, std::ios::binary);
        
        qps_record_file << "The num of centroid is: " << std::to_string(ncentroids[0]) << std::endl;
        time_t now = time(0);
        char* dt = ctime(&now);
        
        record_file << std::endl << "The time now is " << dt;
        record_file << "The server node is: " << hostname << std::endl;
        record_file << "The memory usage format is ixrss, isrss, idrss, maxrss" << std::endl;
        record_file << "Now starting the indexing process " << std::endl;
        record_file << "/*            Parameter Setting          */ " << std::endl;
        record_file << "Training vectors: ";
        for (size_t i = 0; i < layers; i++) {record_file << num_train[i] << " ";} record_file << std::endl;
        record_file << "PQ Training Vectors: " << PQ_train_size << std::endl;
        record_file << "Number of batches: " << nbatches << std::endl;
    }

    Bslib_Index index = Bslib_Index(dimension, layers, index_type, use_reranking, saving_index, use_norm_quantization, is_recording,
    use_HNSW_VQ, use_HNSW_group, use_all_HNSW, use_OPQ, use_train_selector, train_size, M_PQ, nbits, group_HNSW_thres);

    index.build_index(path_learn, path_groups, path_labels, path_quantizers, VQ_layers,
    PQ_layers, LQ_layers, ncentroids, M_HNSW, efConstruction, M_PQ_layer, nbits_PQ_layer, 
    num_train, selector_train_size, selector_group_size, LQ_type, record_file);

    index.assign_vectors(path_ids, path_base, path_alphas_raw, batch_size, nbatches, record_file);

    index.train_pq_quantizer(path_pq, path_pq_norm, M_PQ, path_learn, path_OPQ, PQ_train_size, record_file);

    index.load_index(path_index, path_ids, path_base, path_base_norm, path_centroid_norm, path_group_HNSW, path_alphas_raw,
                     path_alphas, group_HNSW_M, group_HNSW_efConstruction, batch_size, nbatches, nb, record_file);

    index.index_statistic(path_base, path_ids, path_alphas_raw, nb, nbatches);

    index.query_test(num_search_paras, num_recall, nq, ngt, max_vectors, result_k, keep_space, reranking_space, record_file, 
                    qps_record_file, search_mode, path_base, path_gt, path_query);
}