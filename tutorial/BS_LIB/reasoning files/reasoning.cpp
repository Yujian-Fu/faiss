#include "../utils/utils.h"
#include <unordered_set>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
#include <fstream>
#include "../quantizer.h"

//Parameters

/*
const std::string dataset = "SIFT1M";
const std::string model = "models_VQ";
const size_t dimension = 128;
size_t train_set_size = 100000;
const size_t base_set_size = 1000000;
const size_t query_set_size = 1000;
const size_t ngt = 100;
const bool use_sub_train_set = false;
const size_t recall_test_size = 3;
*/


/*
const std::string dataset = "GIST1M";
const std::string model = "models_VQ";
const size_t dimension = 960;
size_t train_set_size = 500000;
const size_t base_set_size = 1000000;
const size_t query_set_size = 1000;
const size_t ngt = 100;
const bool use_sub_train_set = true;
const size_t recall_test_size = 3;
*/

/*
const std::string dataset = "DEEP1M";
const std::string model = "models_VQ";
const size_t dimension = 256;
size_t train_set_size =  100000;
const size_t base_set_size = 1000000;
const size_t query_set_size = 1000;
const size_t ngt = 100;
const bool use_sub_train_set = false;
const size_t recall_test_size = 3;
*/
typedef faiss::Index::idx_t idx_t;
using namespace bslib;
int main(){
    
    /*
    std::string dataset_name = "SIFT";
    std::string path_list[3] = {"/home/y/yujianfu/ivf-hnsw/data/analysis/SIFT_10K_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/SIFT_100K_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/SIFT_1000K_base.fvecs"};
    std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/analysis/SIFT_query.fvecs";
    size_t size_list[3] = {10000, 100000, 1000000};
    const std::string path_record = "/home/y/yujianfu/ivf-hnsw/data/analysis/SIFT_reasoning.txt";
    size_t dimension = 128;
    */
    

    /*
    std::string dataset_name = "GIST";
    std::string path_list[3] = {"/home/y/yujianfu/ivf-hnsw/data/analysis/GIST_10K_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/GIST_100K_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/GIST_1000K_base.fvecs"};
    std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/analysis/GIST_query.fvecs";
    size_t size_list[3] = {10000, 100000, 1000000};
    const std::string path_record = "/home/y/yujianfu/ivf-hnsw/data/analysis/GIST_reasoning.txt";
    size_t dimension = 960;
    */

    /*
    std::string dataset_name = "DEEP";
    std::string path_list[3] = {"/home/y/yujianfu/ivf-hnsw/data/analysis/DEEP_10K_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/DEEP_100K_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/DEEP_1000K_base.fvecs"};
    std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/analysis/DEEP_query.fvecs";
    size_t size_list[3] = {10000, 100000, 1000000};
    const std::string path_record = "/home/y/yujianfu/ivf-hnsw/data/analysis/DEEP_reasoning.txt";
    size_t dimension = 256;
    */

    size_t dim_para = 100;
    std::string dataset_name = "Random_" + std::to_string(dim_para);
    std::string path_list[3] = {"/home/y/yujianfu/ivf-hnsw/data/analysis/Random_10K_"+std::to_string(dim_para)+"_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/Random_10K_"+std::to_string(dim_para)+"_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/Random_10K_"+std::to_string(dim_para)+"_base.fvecs"};
    std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/analysis/Random_"+std::to_string(dim_para)+"_query.fvecs";
    size_t size_list[3] = {10000, 100000, 1000000};
    const std::string path_record = "/home/y/yujianfu/ivf-hnsw/data/analysis/Random_"+std::to_string(dim_para)+"_reasoning.txt";
    size_t dimension = dim_para;
    
    

   /*
    size_t dim_para = 1000;
    std::string dataset_name = "Gaussian_" + std::to_string(dim_para);
    std::string path_list[3] = {"/home/y/yujianfu/ivf-hnsw/data/analysis/Gaussian_10K_"+std::to_string(dim_para)+"_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/Gaussian_10K_"+std::to_string(dim_para)+"_base.fvecs", "/home/y/yujianfu/ivf-hnsw/data/analysis/Gaussian_10K_"+std::to_string(dim_para)+"_base.fvecs"};
    std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/analysis/Gaussian_"+std::to_string(dim_para)+"_query.fvecs";
    size_t size_list[3] = {10000, 100000, 1000000};
    const std::string path_record = "/home/y/yujianfu/ivf-hnsw/data/analysis/Gaussian_"+std::to_string(dim_para)+"_reasoning.txt";
    size_t dimension = dim_para;
    */
    
    const size_t recall_test_size = 3;
    const size_t ngt = 100;
    std::ofstream record_output;
    record_output.open(path_record, std::ios::out);

    for (size_t dataset_index = 0; dataset_index < 3; dataset_index++){
        const std::string path_base = path_list[dataset_index];
        std::cout << "Processing dataset " << path_base << std::endl;
        size_t base_set_size = size_list[dataset_index];
        size_t train_set_size = base_set_size / 10;
        size_t query_set_size = 1000;

        std::vector<float> train_set(dimension * train_set_size);
        std::vector<float> base_set(dimension * base_set_size);
        std::vector<float> query_set(dimension * query_set_size);


    
    //const std::string path_learn = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_learn.fvecs";
    //const std::string path_base = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
    //const std::string path_gt = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
    //const std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";



        //std::ifstream train_input(path_learn, std::ios::binary);
        std::ifstream base_input(path_base, std::ios::binary);
        //std::ifstream gt_input(path_gt, std::ios::binary);
        std::ifstream query_input(path_query, std::ios::binary);
        
    
        //PrepareFolder((char *) ("/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset).c_str());
        
        //readXvec<float>(train_input, train_set.data(), dimension, train_set_size, false, false);
        readXvec<float>(base_input, base_set.data(), dimension, base_set_size, false, false);
        //readXvec<uint32_t>(gt_input, gt_set.data(), ngt, query_set_size, false, false);
        readXvec<float> (query_input, query_set.data(), dimension, query_set_size, false, false);
        memcpy(train_set.data(), base_set.data(), train_set_size * dimension * sizeof(float));


    time_recorder trecorder;
    for (size_t centroid_num = base_set_size / 2000; centroid_num < base_set_size / 200; centroid_num += base_set_size / 2000){

        PrintMessage("Training vectors");
        trecorder.reset();
        faiss::Clustering clus (dimension, centroid_num);
        clus.verbose = true;
        faiss::IndexFlatL2 index (dimension);
        clus.train (train_set_size, train_set.data(), index);

        trecorder.record_time_usage(record_output, "Finish clustering: ");
        trecorder.print_time_usage("Finish clustering with time usage: ");
        trecorder.reset();
        record_output << "Construction parameter: dataset: " << dataset_name << " " << base_set_size << " train_size: " << train_set_size << " n_centroids: " << centroid_num << " iteration: " << clus.niter << std::endl; 
        std::vector<idx_t> base_assigned_ids(base_set_size); std::vector<float> base_assigned_dis(base_set_size);
        std::vector<idx_t> train_assigned_ids(train_set_size); std::vector<float> train_assigned_dis(train_set_size);

        index.search(base_set_size, base_set.data(), 1, base_assigned_dis.data(), base_assigned_ids.data());
        index.search(train_set_size, train_set.data(), 1, train_assigned_dis.data(), train_assigned_ids.data());
        trecorder.record_time_usage(record_output, "Assigned base vectors and train vectors ");
        trecorder.print_time_usage("Assigned base vectors and train vectors ");

        std::vector<std::vector<idx_t>> train_assigned_set(centroid_num);
        for (size_t i = 0; i < train_set_size; i++){train_assigned_set[train_assigned_ids[i]].push_back(i);}
        float avg_distance = 0, std_distance = 0;
        for (size_t i = 0; i < train_set_size; i++){ avg_distance += train_assigned_dis[i];} avg_distance /= train_set_size;
        for (size_t i = 0; i < train_set_size; i++){ std_distance += (train_assigned_dis[i]-avg_distance) * (train_assigned_dis[i]-avg_distance);} std_distance /= (train_set_size-1);
        record_output << "Avg train distance: " << avg_distance << " std train distance: " << std_distance << std::endl;
        for (size_t i = 0; i < centroid_num; i++){record_output << train_assigned_set[i].size() << " ";} record_output << std::endl;

        //Quality analysis
        std::vector<std::vector<size_t>> assigned_set(centroid_num);
        for (size_t i = 0; i < base_set_size; i++){assigned_set[base_assigned_ids[i]].push_back(i);}
        const size_t recall_test[recall_test_size] = {1, 10, 100};
        std::vector<std::vector<size_t>> query_search_result(query_set_size);
        std::vector<size_t> query_max_centroids(query_set_size * recall_test_size);

        faiss::IndexFlatL2 quantizer_index = faiss::IndexFlatL2(dimension);
        quantizer_index.add(base_set_size, base_set.data());
        std::vector<idx_t> gt_set(ngt * query_set_size);
        std::vector<float> gt_distance_set(ngt * query_set_size);
        quantizer_index.search(query_set_size, query_set.data(), ngt, gt_distance_set.data(), gt_set.data());

        PrintMessage("Analysing clustering quality");
        trecorder.reset();
#pragma omp parallel for
        for (size_t i = 0; i < query_set_size; i++){
            const float * query = query_set.data() + i * dimension;
            std::vector<std::vector<size_t>> result_distribution_test(recall_test_size, std::vector<size_t>(centroid_num, 0));
            std::vector<std::vector<size_t>> result_visited_test(recall_test_size, std::vector<size_t>(centroid_num, 0));

            std::vector<idx_t> centroids_ids(centroid_num); std::vector<float> centroids_dis(centroid_num);

            index.search(1, query, centroid_num, centroids_dis.data(), centroids_ids.data());
            
            for (size_t j = 0; j < recall_test_size; j++){
                size_t recall_num = recall_test[j];
                size_t max_centroids = 0;
                std::unordered_set<idx_t> gt_test_set;
                for (size_t k = 0; k < recall_num; k++){gt_test_set.insert(gt_set[i * ngt + k]);} // std::cout<< gt_set[i * ngt + k] << " ";}std::cout << std::endl;

                for (size_t k = 0; k < centroid_num; k++){
                    idx_t centroid_id = centroids_ids[k];

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
                query_max_centroids[i * recall_test_size + j] = max_centroids;
                for (size_t k = 0; k < max_centroids; k++){
                    query_search_result[i].push_back(result_distribution_test[j][k]);
                    
                }
                for (size_t k = 0; k < max_centroids; k++){
                    query_search_result[i].push_back(result_visited_test[j][k]);
                    //std::cout << result_distribution_test[j][k] << " " << result_visited_test[j][k] << " ";
                }
            }
        }

        std::vector<size_t> avg_max_centroids(recall_test_size, 0);
        for (size_t i = 0; i < query_set_size; i++){
            size_t visiting_centroids = 0;
            record_output << "Q: " << i << std::endl;
            for (size_t j = 0; j < recall_test_size; j++){
                
                size_t max_centroids = query_max_centroids[i * recall_test_size + j];
                record_output << "R@" << recall_test[j] << " MC: " << max_centroids << std::endl;
                avg_max_centroids[j] += max_centroids;
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

        record_output << "The average max centroids: " << float(avg_max_centroids[0]) / query_set_size << " " << float(avg_max_centroids[1]) / query_set_size << " " << float(avg_max_centroids[2]) / query_set_size << std::endl;
        trecorder.record_time_usage(record_output, "Finished result analysis: ");
    }
}
}
