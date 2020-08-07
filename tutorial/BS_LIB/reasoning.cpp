#include "utils/utils.h"
#include <unordered_set>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>

//Parameters
    /*// SIFT dataset 
    std::string dataset = "SIFT1M";
    std::string model = "models_VQ";
    size_t dimension = 128;
    size_t train_set_size = 100000;
    size_t base_set_size = 1000000;
    size_t query_set_size = 1000;
    size_t ngt = 100;
    */

    std::string dataset = "GIST1M";
    std::string model = "models_VQ";
    size_t dimension = 960;
    size_t train_set_size = 500000;
    size_t base_set_size = 1000000;
    size_t query_set_size = 1000;
    size_t ngt = 100;
    bool use_sub_train_set = true;

    
    const std::string path_learn = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_learn.fvecs";
    const std::string path_base = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
    const std::string path_gt = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
    const std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";
    std::string path_record = "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/reasoning.txt";

    std::vector<float> train_set(dimension * train_set_size);
    std::vector<float> base_set(dimension * base_set_size);
    std::vector<float> query_set(dimension * query_set_size);
    std::vector<uint32_t> gt_set(ngt * base_set_size);
    
    std::ifstream train_input(path_learn, std::ios::binary);
    std::ifstream base_input(path_base, std::ios::binary);
    std::ifstream gt_input(path_gt, std::ios::binary);
    std::ifstream query_input(path_query, std::ios::binary);
    std::ofstream record_output;
    

typedef faiss::Index::idx_t idx_t;
using namespace bslib;
int main(){
    PrepareFolder((char *) ("/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset).c_str());
    record_output.open(path_record, std::ios::out);
    readXvec<float>(train_input, train_set.data(), dimension, train_set_size, false, false);
    readXvec<float>(base_input, base_set.data(), dimension, base_set_size, false, false);
    readXvec<uint32_t>(gt_input, gt_set.data(), ngt, query_set_size, false, false);
    readXvec<float> (query_input, query_set.data(), dimension, query_set_size, false, false);
    
    if (use_sub_train_set){
        size_t query_subset_size = query_set_size / 5;
        std::vector<float> query_subset(dimension * query_subset_size);
        RandomSubset<float>(query_set.data(), query_subset.data(), dimension, query_set_size, query_subset_size);
        query_set.resize(query_subset.size());
        memcpy(query_set.data(), query_subset.data(), query_subset.size() * sizeof(float));
    }
    
    time_recorder trecorder;
    for (size_t centroid_num = 200; centroid_num < 250; centroid_num += 50){

        PrintMessage("Training vectors");
        trecorder.reset();
        faiss::Clustering clus (dimension, centroid_num);
        clus.verbose = true;
        faiss::IndexFlatL2 index (dimension);
        clus.train (train_set_size, train_set.data(), index);
        
        trecorder.record_time_usage(record_output, "Finish clustering: ");
        trecorder.print_time_usage("Finish clustering with time usage: ");
        trecorder.reset();
        record_output << "Construction parameter: dataset: " << dataset << " train_size: " << train_set_size << " n_centroids: " << centroid_num << " iteration: " << clus.niter << std::endl; 
        std::vector<idx_t> base_assigned_ids(base_set_size); std::vector<float> base_assigned_dis(base_set_size);
        std::vector<idx_t> query_assigned_ids(query_set_size); std::vector<float> query_assigned_dis(query_set_size);
        std::vector<idx_t> train_assigned_ids(train_set_size); std::vector<float> train_assigned_dis(train_set_size);

        index.search(base_set_size, base_set.data(), 1, base_assigned_dis.data(), base_assigned_ids.data());
        index.search(query_set_size, query_set.data(), 1, query_assigned_dis.data(), query_assigned_ids.data());
        index.search(train_set_size, train_set.data(), 1, train_assigned_dis.data(), train_assigned_ids.data());
        trecorder.record_time_usage(record_output, "Assigned base vectors and query vectors ");
        trecorder.print_time_usage("Assigned base vectors and query vectors ");

        std::vector<std::vector<idx_t>> train_assigned_set(centroid_num);
        for (size_t i = 0; i < train_set_size; i++){train_assigned_set[train_assigned_ids[i]].push_back(i);}
        float avg_distance = 0;
        for (size_t i = 0; i < train_set_size; i++){ avg_distance += train_assigned_dis[i];} avg_distance /= train_set_size;
        record_output << "Avg train distance: " << avg_distance << std::endl;
        for (size_t i = 0; i < centroid_num; i++){record_output << train_assigned_set[i].size() << " ";} record_output << std::endl;
        for (size_t i = 0; i < query_set_size; i++){std::cout << query_assigned_ids[i] << " ";} std::cout << std::endl;

        //Quality analysis
        std::vector<std::vector<size_t>> assigned_set(centroid_num);
        for (size_t i = 0; i < base_set_size; i++){assigned_set[base_assigned_ids[i]].push_back(i);}
        const size_t recall_test[3] = {1, 10, 100};
        std::vector<std::vector<size_t>> query_search_result(query_set_size);
        std::vector<size_t> query_max_centroids(query_set_size * 3);

        PrintMessage("Analysing clustering quality");
        trecorder.reset();
#pragma omp parallel for
        for (size_t i = 0; i < query_set_size; i++){
            std::vector<std::vector<size_t>> result_distribution_test(3, std::vector<size_t>(centroid_num, 0));
            std::vector<std::vector<size_t>> result_visited_test(3, std::vector<size_t>(centroid_num, 0));

            std::vector<idx_t> centroids_ids(centroid_num); std::vector<float> centroids_dis(centroid_num);

            index.search(1, index.xb.data()+query_assigned_ids[i]*dimension, centroid_num, centroids_dis.data(), centroids_ids.data());
            
            for (size_t j = 0; j < 3; j++){
                size_t recall_num = recall_test[j];
                size_t max_centroids = 0;
                std::unordered_set<idx_t> gt_test_set;
                for (size_t k = 0; k < recall_num; k++){gt_test_set.insert(gt_set[i * ngt + k]);} // std::cout<< gt_set[i * ngt + k] << " ";}std::cout << std::endl;

                for (size_t k = 0; k < centroid_num; k++){
                    size_t centroid_id = centroids_ids[k];

                    result_distribution_test[j][k] = k == 0 ? 0 : result_distribution_test[j][k - 1];
                    result_visited_test[j][k] = k == 0 ? assigned_set[centroid_id].size() : result_visited_test[j][k - 1] + assigned_set[centroid_id].size();
                    
                    for (size_t l = 0; l < assigned_set[centroid_id].size(); l++){
                        if (gt_test_set.count(assigned_set[centroid_id][l]) != 0){
                            result_distribution_test[j][k] += 1;
                        }
                    }
                    if (result_distribution_test[j][k] == recall_num){
                        max_centroids = k + 1;
                        break;
                    }
                }
                query_max_centroids[i * 3 + j] = max_centroids;
                for (size_t k = 0; k < max_centroids; k++){
                    query_search_result[i].push_back(result_distribution_test[j][k]);
                    
                }
                for (size_t k = 0; k < max_centroids; k++){
                    query_search_result[i].push_back(result_visited_test[j][k]);
                    //std::cout << result_distribution_test[j][k] << " " << result_visited_test[j][k] << " ";
                }
            }
        }

        for (size_t i = 0; i < query_set_size; i++){
            size_t visiting_centroids = 0;
            record_output << "Q: " << i << std::endl;
            for (size_t j = 0; j < 3; j++){
                
                size_t max_centroids = query_max_centroids[i * 3 + j];
                record_output << "R@" << recall_test[j] << " MC: " << max_centroids << std::endl;
                for (size_t k = 0; k < max_centroids; k++){
                    record_output << query_search_result[i][visiting_centroids + k] << " "; 
                }
                record_output << std::endl;
                visiting_centroids += max_centroids;

                for (size_t k = 0; k < max_centroids; k++){
                    record_output << query_search_result[i][visiting_centroids + k] << " "; 
                }
                record_output << std::endl;
                visiting_centroids += max_centroids;
            }
        }
        trecorder.record_time_usage(record_output, "Finished result analysis: ");
    }
}