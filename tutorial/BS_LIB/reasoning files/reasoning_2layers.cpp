#include "utils/utils.h"
#include <unordered_set>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
#include <algorithm>
#include <cassert>
//Parameters
    
    /*
    const std::string dataset = "SIFT1M";
    const size_t dimension = 128;
    size_t train_set_size = 100000;
    const size_t base_set_size = 1000000;
    const size_t query_set_size = 1000;
    const size_t ngt = 100;
    const size_t recall_test_size = 3;
    const float MIN_DISTANCE = 1e8;
    */
    
    
    const std::string dataset = "DEEP1M";
    const size_t dimension = 256;
    size_t train_set_size =  100000;
    const size_t base_set_size = 1000000;
    const size_t query_set_size = 1000;
    const size_t ngt = 100;
    const size_t recall_test_size = 3;
    const float MIN_DISTANCE = 100;
    
    

    const std::string model = "models_VQ_VQ";
    const bool use_fast_assign = false;

    const std::string path_learn = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_learn.fvecs";
    const std::string path_base = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
    const std::string path_gt = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
    const std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";
    
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
    
    readXvec<float>(train_input, train_set.data(), dimension, train_set_size, false, false);
    readXvec<float>(base_input, base_set.data(), dimension, base_set_size, false, false);
    readXvec<uint32_t>(gt_input, gt_set.data(), ngt, query_set_size, false, false);
    readXvec<float> (query_input, query_set.data(), dimension, query_set_size, false, false);

    time_recorder trecorder;
    const size_t first_layer_size = 5, second_layer_size = 5;
    const size_t layer1_centroid_num[first_layer_size] = {60, 70, 80, 90, 100};
    const size_t layer2_centroid_num[second_layer_size] = {60, 70, 80, 90, 100};

    std::string path_record = "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/reasoning_" + model + "_" + std::to_string(layer1_centroid_num[0]) + "_" +
                                std::to_string(layer1_centroid_num[first_layer_size - 1]) + "_" + std::to_string(layer2_centroid_num[0]) + "_" + std::to_string(layer2_centroid_num[second_layer_size - 1]) + ".txt";
    record_output.open(path_record, std::ios::out);

    for (size_t temp1 = 0; temp1 < first_layer_size; temp1++){
        for (size_t temp2 = 0; temp2 < second_layer_size; temp2++){
            size_t centroid_num1 = layer1_centroid_num[temp1];
            size_t centroid_num2 = layer2_centroid_num[temp2];

            std::cout << "Training vectors for structure " << centroid_num1 << " " << centroid_num2 << std::endl;
            trecorder.reset();
            faiss::Clustering clus1 (dimension, centroid_num1);
            clus1.verbose = true;
            faiss::IndexFlatL2 index1(dimension);
            clus1.train(train_set_size, train_set.data(), index1);
            record_output << "Construction parameter: dataset: " << dataset << " train_size: " << train_set_size << " n centroids: " << centroid_num1 << " " << centroid_num2 << " iteration: " << clus1.niter << std::endl;
            trecorder.record_time_usage(record_output, "Finish 1st layer clustering: ");

            std::vector<idx_t> train_set_ids(train_set_size);
            std::vector<float> train_set_dists(train_set_size);
            index1.search(train_set_size, train_set.data(), 1, train_set_dists.data(), train_set_ids.data());

            std::vector<std::vector<float>> train_set_second(centroid_num1);
            for (size_t i = 0; i < train_set_size; i++){
                idx_t second_layer_id = train_set_ids[i];
                for (size_t j = 0; j < dimension; j++)
                    train_set_second[second_layer_id].push_back(train_set[i * dimension + j]);
            }
            trecorder.record_time_usage(record_output, "Finish assigning train vectors");

            std::vector<faiss::Clustering *> clus2(centroid_num1);
            std::vector<faiss::IndexFlatL2> index2(centroid_num1);

#pragma omp parallel for
            for (size_t i = 0; i < centroid_num1; i++){
                size_t sub_train_set_size = train_set_second[i].size() / dimension;
                clus2[i] = new faiss::Clustering (dimension, centroid_num2);
                clus2[i]->verbose = false;
                index2[i] = faiss::IndexFlatL2 (dimension);
                clus2[i]->train(sub_train_set_size, train_set_second[i].data(),index2[i]);
            }
            trecorder.record_time_usage(record_output, "Finish 2nd layer clustering");

            trecorder.reset();
            std::vector<idx_t> base_assigned_ids(base_set_size, 0);
            std::vector<float> base_assigned_dists(base_set_size, 1e9);

            if (use_fast_assign){
                std::vector<idx_t> base_assigned_ids_1st(base_set_size);
                std::vector<float> base_assigned_dists_1st(base_set_size);
                index1.search(base_set_size, base_set.data(), 1, base_assigned_dists_1st.data(), base_assigned_ids_1st.data());

                std::vector<idx_t> base_assigned_ids_2nd(base_set_size);
                std::vector<float> base_assgned_dists_2nd(base_set_size);
#pragma omp parallel for
                for (size_t i = 0; i < base_set_size; i++){
                    size_t id_1st_layer = base_assigned_ids_1st[i];
                    index2[id_1st_layer].search(1, base_set.data()+i * dimension, 1, base_assgned_dists_2nd.data()+i, base_assigned_ids_2nd.data()+i);
                }
                for (size_t i = 0; i < base_set_size; i++){
                    base_assigned_ids[i] = base_assigned_ids_1st[i] * centroid_num2 + base_assigned_ids_2nd[i];
                    base_assigned_dists[i] = base_assigned_ids_2nd[i];
                }
            }
            else{
                std::vector<std::vector<idx_t>> base_assigned_ids_indexes(centroid_num1, std::vector<idx_t> (base_set_size));
                std::vector<std::vector<float>> base_assigned_dists_indexes(centroid_num1, std::vector<float> (base_set_size));

#pragma omp parallel for
                for (size_t i = 0; i < centroid_num1; i++){
                    index2[i].search(base_set_size, base_set.data(), 1, base_assigned_dists_indexes[i].data(), base_assigned_ids_indexes[i].data());
                }

#pragma omp parallel for
                for (size_t i = 0; i < base_set_size; i++){
                    float min_distance = MIN_DISTANCE;
                    idx_t min_id = 0;
                    for (size_t j = 0; j < centroid_num1; j++){
                        if (min_distance > base_assigned_dists_indexes[j][i]){
                            min_id = j;
                            min_distance = base_assigned_dists_indexes[j][i];
                        }
                    }
                    base_assigned_ids[i] = base_assigned_ids_indexes[min_id][i] + min_id * centroid_num2;
                    base_assigned_dists[i] = base_assigned_dists_indexes[min_id][i];    
                }   
            }


            std::string message = "Assigned base vectors " + use_fast_assign ? "with fast assign" : "without fast assign";
            trecorder.record_time_usage(record_output, message);
            float avg_base_assigned_dist = 0, std_base_assgned_dist = 0; 
            for (size_t i = 0; i < base_set_size; i++){avg_base_assigned_dist += base_assigned_dists[i];}
            avg_base_assigned_dist /= base_set_size;
            for (size_t i = 0; i < base_set_size; i++){ std_base_assgned_dist += (base_assigned_dists[i]-avg_base_assigned_dist) * (base_assigned_dists[i]-avg_base_assigned_dist);} 
            std_base_assgned_dist /= (base_set_size-1);
            record_output << "Avg base distance: " <<  avg_base_assigned_dist << " std train distance: " << std_base_assgned_dist << std::endl;

            //Quality analysis
            std::vector<std::vector<idx_t>> assigned_set(centroid_num1 * centroid_num2);
            for (size_t i = 0; i < base_set_size; i++){assigned_set[base_assigned_ids[i]].push_back(i);}
            float avg_assigned_vectors = base_set_size / assigned_set.size();
            float std_assigned_vectors = 0;
            for (size_t i = 0; i < assigned_set.size(); i++){
                std_assigned_vectors += (assigned_set[i].size() - avg_assigned_vectors) * (assigned_set[i].size() - avg_assigned_vectors);
            }
            std_assigned_vectors /= assigned_set.size();
            record_output << "STD for num of assigned vectors for " << assigned_set.size() << " clusters: " << std_assigned_vectors << std::endl;
            std::cout << "The assigned set" << std::endl; for (size_t i = 0; i < assigned_set.size(); i++){std::cout << assigned_set[i].size() << " ";} std::cout << std::endl;

            std::vector<std::vector<idx_t>> first_assigned_set(centroid_num1);
            for (size_t i = 0; i < centroid_num1; i++){
                for (size_t j = 0; j < centroid_num2; j++){
                    for (size_t k = 0; k < assigned_set[i * centroid_num2 + j].size(); k++){
                        first_assigned_set[i].push_back(assigned_set[i * centroid_num2 + j][k]);
                    }
                }
            }
            std::cout << "The first assigned set" << std::endl; for (size_t i = 0; i < first_assigned_set.size(); i++){std::cout << first_assigned_set[i].size() << " ";} std::cout << std::endl;

            trecorder.reset();
            PrintMessage("Analysising the construction");
            const size_t recall_num = 100;
            std::vector<size_t> max_query_centroids(query_set_size * 2);
            std::vector<std::vector<size_t>> first_level_distributions(query_set_size);
            std::vector<std::vector<size_t>> first_level_visited_vectors(query_set_size);
            std::vector<std::vector<size_t>> second_level_distributions(query_set_size);
            std::vector<std::vector<size_t>> second_level_visited_vectors(query_set_size);

#pragma omp parallel for
            for (size_t i = 0; i < query_set_size; i++){
                std::vector<size_t> first_level_distribution;
                std::vector<size_t> first_level_visited;
                size_t max_first_centroids = 0;

                const float * query = query_set.data() + i * dimension;
                std::vector<idx_t> centroids_ids(centroid_num1);
                std::vector<float> centroids_dists(centroid_num1);
                index1.search(1, query, centroid_num1, centroids_dists.data(), centroids_ids.data());
                std::unordered_set<idx_t> gt_test_set;
                for (size_t j = 0; j < recall_num; j++){
                    gt_test_set.insert(gt_set[i * ngt + j]);
                }
                for (size_t j = 0; j < centroid_num1; j++){
                    first_level_distribution.push_back(j == 0 ? 0 : first_level_distribution[j-1]);
                    idx_t centroid_id = centroids_ids[j];
                    for (size_t k = 0; k < first_assigned_set[centroid_id].size(); k++){
                        if (gt_test_set.count(first_assigned_set[centroid_id][k]) != 0){
                            first_level_distribution[j] += 1;
                        }
                    }
                    first_level_visited.push_back(j == 0 ? first_assigned_set[centroid_id].size() : first_level_visited[j-1] + first_assigned_set[centroid_id].size());
                    if (first_level_distribution[j] >= recall_num){
                        max_first_centroids = j + 1;
                        break;
                    }
                }

                std::vector<idx_t> all_ids(max_first_centroids * centroid_num2);
                std::vector<float> all_dists(max_first_centroids * centroid_num2);

                for (size_t j = 0; j < max_first_centroids; j++){
                    idx_t centroids_id = centroids_ids[j];
                    std::vector<idx_t> index_ids(centroid_num2);
                    std::vector<float> index_dists(centroid_num2);

                    index2[centroids_id].search(1, query, centroid_num2, index_dists.data(), index_ids.data());
                    for (size_t k = 0; k < centroid_num2; k++){
                        all_dists[j * centroid_num2 + k] = index_dists[k];
                        all_ids[j * centroid_num2 + k] = centroids_id * centroid_num2 + index_ids[k];
                    }
                }

                std::vector<idx_t> sort_dist_index(max_first_centroids * centroid_num2);
                size_t x = 0;
                std::iota(sort_dist_index.begin(), sort_dist_index.end(), x++);
                std::sort(sort_dist_index.begin(),sort_dist_index.end(), [&](int i,int j){return all_dists[i]<all_dists[j];} );
                
                std::vector<size_t> second_level_distribution;
                std::vector<size_t> second_level_visited;
                size_t max_second_centroids = 0;

                for (size_t j = 0; j < max_first_centroids * centroid_num2; j++){
                    size_t sort_id = sort_dist_index[j];
                    idx_t index_id = all_ids[sort_id];
                    second_level_distribution.push_back( j==0 ? 0 : second_level_distribution[j - 1]);
                    
                    for (size_t k = 0; k < assigned_set[index_id].size(); k++){
                        
                        if (gt_test_set.count(assigned_set[index_id][k]) != 0){
                            second_level_distribution[j] += 1;
                        }
                    }
                    second_level_visited.push_back(j == 0 ? assigned_set[index_id].size() : second_level_visited[j-1] + assigned_set[index_id].size());
                    if (second_level_distribution[j] >= recall_num){
                        max_second_centroids = j + 1;
                        break;
                    }
                }

                max_query_centroids[i * 2] = max_first_centroids;
                max_query_centroids[i * 2 + 1] = max_second_centroids;

                assert(first_level_distribution.size() == max_first_centroids);
                assert(first_level_visited.size() == max_first_centroids);
                assert(second_level_distribution.size() == max_second_centroids);
                assert(second_level_visited.size() == max_second_centroids);

                first_level_distributions[i].resize(max_first_centroids);
                first_level_visited_vectors[i].resize(max_first_centroids);
                second_level_distributions[i].resize(max_second_centroids);
                second_level_visited_vectors[i].resize(max_second_centroids);

                for (size_t j = 0; j < max_first_centroids; j++){
                    first_level_distributions[i][j] = first_level_distribution[j];
                    first_level_visited_vectors[i][j] = first_level_visited[j];   
                }

                for (size_t j = 0; j < max_second_centroids; j++){
                    second_level_distributions[i][j] = second_level_distribution[j];
                    second_level_visited_vectors[i][j] = second_level_visited[j];
                }
            }

            size_t avg_first_centroids = 0, avg_second_seconds = 0;

            for (size_t i = 0; i < query_set_size; i++){
                record_output << "Q: " << i << std::endl;
                size_t max_first_centroids = max_query_centroids[i * 2];
                size_t max_second_centroids = max_query_centroids[i * 2 + 1];

                record_output << " MC 1st: " << max_first_centroids << " MC 2nd: " << max_second_centroids << std::endl;
                avg_first_centroids += max_first_centroids;
                avg_second_seconds += max_second_centroids;

                for (size_t j = 0; j < max_first_centroids; j++){
                    record_output << first_level_distributions[i][j] << " ";
                }
                record_output << std::endl;
                for (size_t j = 0; j < max_first_centroids; j++){
                    record_output << first_level_visited_vectors[i][j] << " ";
                }
                record_output << std::endl;
                for (size_t j = 0; j < max_second_centroids; j++){
                    record_output << second_level_distributions[i][j] << " ";
                }
                record_output << std::endl;
                for (size_t j = 0; j < max_second_centroids; j++){
                    record_output << second_level_visited_vectors[i][j] << " ";
                }
                record_output << std::endl;
            }

            record_output << "The avg max first centroids: " << float(avg_first_centroids) / query_set_size << std::endl;
            record_output << "The avg max second centroids: " << float(avg_second_seconds) / query_set_size << std::endl;
            trecorder.record_time_usage(record_output, "Finished result analysis: ");
            trecorder.print_time_usage("Finished result analysis: ");
        }
    }

}