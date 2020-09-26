#include "../utils/utils.h"
#include <unordered_set>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
#include <faiss/utils/utils.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/distances.h>

/*
Draw the visited-vectors - recall && PQ_recall to evaluate the loss from PQ
Draw the figure with different centroid setting

*/

// Parameters
    
    const std::string dataset = "SIFT1M";
    const std::string model = "models_VQ";
    const size_t dimension = 128;
    size_t train_set_size = 100000;
    const size_t base_set_size = 1000000;
    const size_t query_set_size = 1000;
    const size_t ngt = 100;
    const bool use_sub_train_set = false;
    const size_t recall_test_size = 3;
    const size_t nbits = 12;
    
    
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
    const size_t nbits = 12;
    */
    
    
    const std::string path_learn = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_learn.fvecs";
    const std::string path_base = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
    const std::string path_gt = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
    const std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";
    std::string path_record = "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/reasoning_PQ_" + std::to_string(nbits) + ".txt";

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
    
    //Compute the sub train set for PQ training 
    size_t sub_train_set_size = nbits == 8? train_set_size / 10 : train_set_size / 5;
    std::vector<idx_t> sub_assigned_ids(sub_train_set_size);
    std::vector<float> sub_assigned_dists(sub_train_set_size);
    std::vector<float> sub_train_set(dimension * sub_train_set_size);
    RandomSubset(train_set.data(), sub_train_set.data(), dimension, train_set_size, sub_train_set_size);

    time_recorder trecorder;
    for (size_t centroid_num = 1000; centroid_num < 5500; centroid_num += 500){

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

        index.search(sub_train_set_size, sub_train_set.data(), 1, sub_assigned_dists.data(), sub_assigned_ids.data());

        trecorder.reset();
        std::vector<float> base_set_residual(dimension * base_set_size);
#pragma omp parallel for
        for (size_t i = 0; i < base_set_size; i++){
            const idx_t centroid_id = base_assigned_ids[i];
            const float * centroid = index.xb.data() + centroid_id * dimension;
            faiss::fvec_madd(dimension, base_set.data() + i * dimension, -1.0, centroid, base_set_residual.data() + i * dimension);
        }
        trecorder.print_time_usage("Computed base residuals");


        trecorder.reset();
        std::vector<float> sub_train_set_residual(dimension * sub_train_set_size);
#pragma omp parallel for
        for (size_t i = 0; i < sub_train_set_size; i++){
            const idx_t centroid_id = sub_assigned_ids[i];
            const float * centroid = index.xb.data() + centroid_id * dimension;
            faiss::fvec_madd(dimension, sub_train_set.data() + i * dimension, -1.0, centroid, sub_train_set_residual.data() + i * dimension);
        }
        trecorder.print_time_usage("Computed sub train residuals");

        std::vector<std::vector<idx_t>> assigned_set(centroid_num);
        for (size_t i = 0; i < base_set_size; i++){
            assigned_set[base_assigned_ids[i]].push_back(i);
        }

        size_t recall_num = 100;
        

        trecorder.reset();
        std::cout << "Training PQ with " << nbits << " nbits" << std::endl;
        faiss::ProductQuantizer PQ(dimension, 8, nbits);
        record_output << "Training PQ with " << nbits << " nbits" << std::endl;
        PQ.verbose = true;
        PQ.train(sub_train_set_size, sub_train_set_residual.data());
        size_t code_size = PQ.code_size;
        std::vector<uint8_t> base_set_code(code_size * base_set_size);
        PQ.compute_codes(base_set_residual.data(), base_set_code.data(), base_set_size);
        trecorder.record_time_usage(record_output, "Finished training PQ and compute codes: ");

        PrintMessage("Analysing PQ loss to recall");
        std::vector<size_t> query_max_centroids(query_set_size);
        std::vector<std::vector<size_t>> result_distributions(query_set_size);
        std::vector<std::vector<size_t>> result_PQ_distributions(query_set_size);
        std::vector<std::vector<size_t>> result_visited_vectors(query_set_size);

        trecorder.reset();
#pragma omp parallel for
        for (size_t i = 0; i < query_set_size; i++){
            std::vector<size_t> result_distribution;
            std::vector<size_t> result_visited;
            std::vector<size_t> result_PQ_distribution;

            std::vector<idx_t> centroids_ids(centroid_num);
            std::vector<float> centroids_dists(centroid_num);

            const float * query = query_set.data() + i * dimension;
            index.search(1, query, centroid_num, centroids_dists.data(), centroids_ids.data());

            size_t max_centroids = 0;
            std::unordered_set<idx_t> gt_test_set;
            for (size_t j = 0; j < recall_num; j++){gt_test_set.insert(gt_set[ i * ngt + j]);}

            std::vector<idx_t> heap_ids(recall_num);
            std::vector<float> heap_dists(recall_num);
            faiss::maxheap_heapify(recall_num, heap_dists.data(), heap_ids.data());
            
            for (size_t j = 0; j < centroid_num; j++){
                idx_t centroid_id = centroids_ids[j];

                result_distribution.push_back(j == 0? 0 : result_distribution[j-1]);
                result_PQ_distribution.push_back(0);
                result_visited.push_back(j == 0? assigned_set[centroid_id].size() : result_visited[j-1] + assigned_set[centroid_id].size());

                for (size_t k = 0; k < assigned_set[centroid_id].size(); k++){

                    idx_t base_id = assigned_set[centroid_id][k];
                    std::vector<float> decoded_residuals(dimension);
                    PQ.decode(base_set_code.data() + base_id * code_size, decoded_residuals.data());
                    std::vector<float> reconstructed_vector(dimension);
                    faiss::fvec_madd(dimension, decoded_residuals.data(), 1.0, index.xb.data() + centroid_id*dimension, reconstructed_vector.data());
                    std::vector<float> query_base_residual_vector(dimension);
                    faiss::fvec_madd(dimension, reconstructed_vector.data(), -1.0, query, query_base_residual_vector.data());
                    float base_norm = faiss::fvec_norm_L2sqr(query_base_residual_vector.data(), dimension);
                    if (base_norm < heap_dists[0]){
                        faiss::maxheap_pop(recall_num, heap_dists.data(), heap_ids.data());
                        faiss::maxheap_push(recall_num, heap_dists.data(), heap_ids.data(), base_norm, base_id);
                    }
                    if (gt_test_set.count(assigned_set[centroid_id][k]) != 0){
                        result_distribution[j] += 1;
                    }
                }

                for (size_t k = 0; k < recall_num; k++){
                    if (gt_test_set.count(heap_ids[k]) != 0){
                        result_PQ_distribution[j] += 1;
                    }
                }

                if (result_distribution[j] >= recall_num){
                    max_centroids = j + 1;
                    break;
                }
            }
            query_max_centroids[i] = max_centroids;

            assert(result_visited.size() == max_centroids);
            assert(result_distribution.size() == max_centroids);
            assert(result_PQ_distribution.size() == max_centroids);

            result_distributions[i].resize(max_centroids);
            result_PQ_distributions[i].resize(max_centroids);
            result_visited_vectors[i].resize(max_centroids);

            for(size_t j = 0; j < max_centroids; j++){
                result_visited_vectors[i][j] = result_visited[j];
                result_PQ_distributions[i][j] = result_PQ_distribution[j];
                result_distributions[i][j] = result_distribution[j];
            }
        }

        size_t avg_max_centroids = 0;
        for (size_t i = 0; i < query_set_size; i++){
            record_output << "Q: " << i << std::endl;

            size_t max_centroids = query_max_centroids[i];
            avg_max_centroids += max_centroids;
            record_output << "MC: " << max_centroids << std::endl;

            for (size_t j = 0; j < max_centroids; j++){
                record_output << result_distributions[i][j] << " ";
            }
            record_output << std::endl;

            for (size_t j = 0; j < max_centroids; j++){
                record_output << result_PQ_distributions[i][j] << " ";
            }
            record_output << std::endl;

            for (size_t j = 0; j < max_centroids; j++){
                record_output << result_visited_vectors[i][j] << " ";
            }
            record_output << std::endl;
        }

        record_output << "The average max centroids " << float(avg_max_centroids) / query_set_size << std::endl;
        trecorder.record_time_usage(record_output, "Finished result analysis: ");
        trecorder.print_time_usage("Finished result analysis: ");
    }
}
        



