


#include <iostream>
#include "../../utils/utils.h"
#include "../../parameters/parameter_tuning/inverted_index/Inverted_Index.h"
#include "../parameter_multi.h"
#include <unordered_set>
#include <time.h>
#include <sstream>
#include <iomanip>

using namespace bslib;

// ids: sizeof(idx) * nb * M
void assign_residual(const float * residuals, const float * centroid, size_t dimension, idx_t * PQ_ids, size_t nc_PQ, size_t nb){
    size_t dimension_sub = dimension / M;
    for (size_t PQ_index = 0; PQ_index < M; PQ_index++){
        faiss::IndexFlatL2 index_sub(dimension_sub);
        index_sub.add(nc_PQ, centroid + PQ_index * nc_PQ * dimension_sub);
        std::vector<float> distance(nb);
#pragma omp parallel for
        for (size_t i = 0; i < nb; i++){
            index_sub.search(1, residuals + i * dimension + PQ_index * dimension_sub, 1, distance.data()+i, PQ_ids + PQ_index * nb + i);
        }
    }
}

int main(){
    std::string dt = GetNowTime();

    std::vector<float> base_vectors(dimension * nb);
    std::vector<idx_t> base_ids(nb);
    std::vector<float> base_residual(dimension * nb);
    size_t nt = num_train[0];
    std::vector<float> train_vectors(dimension * nt);
    std::vector<idx_t> train_ids(nt);
    std::vector<float> train_residual(dimension * nt);

    std::ifstream base_input(path_base, std::ios::binary);
    std::ifstream train_input(path_learn, std::ios::binary);

    readXvecFvec<float>(base_input, base_vectors.data(), dimension, nb, true);
    readXvecFvec<float>(train_input, train_vectors.data(), dimension, num_train[0], true);
    std::stringstream ss;
    ss << std::setprecision(2) << alpha;

    std::string path_record = "./record/kmeans_PQ_time_" + dt + " _" + ss.str() + "_" + std::to_string(index_iter) + 
                                "_" + std::to_string(PQ_iter) + "_" + std::to_string(total_iter) + ".txt";
    std::cout << "Save path to " << path_record << std::endl;
    std::ofstream record_file(path_record);

    if (dimension % M != 0){
        std::cout << "The dimension " << dimension << " should be an integer multiple of M: " << M << std::endl;
        exit(0); 
    }

    for (size_t nc = nc_low; nc <= nc_up; nc += nc_step){
        record_file << "Kmeans with centroids: " << nc << std::endl;
    time_recorder Trecorder = time_recorder();
    std::vector<float> PQ_centroids(dimension * nc_PQ);
    std::vector<idx_t> PQ_train_ids(nt * M);
    


    std::vector<float> index_centroids(dimension * nc);
    std::cout << "Initializing the index centroids " << std::endl;
    initialize_centroid(train_vectors.data(), index_centroids.data(), nt, dimension, nc);

    std::cout << "Assigning the vectors " << std::endl;
    assign_vector(index_centroids.data(), train_vectors.data(), dimension, train_ids.data(), nc, nt);

    record_file << "Base vector - centroid distance" << std::endl;
    std::cout << "Start the training process " << std::endl;
    for (size_t iter = 0; iter < total_iter * index_iter; iter++){
        kmeans_update_centroids(train_vectors.data(), index_centroids.data(), train_ids.data(), nt, nc, dimension);
        assign_vector(index_centroids.data(), train_vectors.data(), dimension, train_ids.data(), nc, nt);
        std::vector<float> distance_metric(2,0);
        metric_computation(train_vectors.data(), index_centroids.data(), train_ids.data(), PQ_centroids.data(), PQ_train_ids.data(), nt, nc_PQ, dimension, distance_metric.data(), true, false);
        record_file << distance_metric[0] << " ";
    }
    record_file << std::endl;

    std::cout << "Computing the residual " << std::endl;
    compute_index_residual(train_vectors.data(), index_centroids.data(), train_ids.data(), train_residual.data(), nt, dimension);

    std::cout << "Initializing the PQ centroids " << std::endl;
    initialize_PQ_centroid(train_residual.data(), PQ_centroids.data(), nt, dimension, nc_PQ);

    std::cout << "Assinging the residuals to the PQ centroids " << std::endl;
    assign_residual(train_residual.data(), PQ_centroids.data(), dimension, PQ_train_ids.data(), nc_PQ, nt);

    record_file << "Compression Loss distance" << std::endl;
    for (size_t iter = 0; iter < total_iter * PQ_iter; iter++){
        update_PQ_centroids(train_residual.data(), PQ_centroids.data(), PQ_train_ids.data(), nt, nc_PQ, dimension);
        assign_residual(train_residual.data(), PQ_centroids.data(), dimension, PQ_train_ids.data(), nc_PQ, nt);
        std::vector<float> distance_metric(2,0);
        metric_computation(train_vectors.data(), index_centroids.data(), train_ids.data(), PQ_centroids.data(), PQ_train_ids.data(), nt, nc_PQ, dimension, distance_metric.data(), false, true);
        record_file << distance_metric[1] << " ";
    }
    record_file << std::endl;
    

    std::string message = "Time for kmeans training: ";
    Trecorder.record_time_usage(record_file, message);
    Trecorder.print_time_usage(message);

    std::vector<idx_t> PQ_ids(nb * M);
    std::cout << "Assigning the vectors " << std::endl;
    assign_vector(index_centroids.data(), base_vectors.data(), dimension, base_ids.data(), nc, nb);
    std::cout << "Computing the residual " << std::endl;
    compute_index_residual(base_vectors.data(), index_centroids.data(), base_ids.data(), base_residual.data(), nb, dimension);
    std::cout << "Assinging the residuals to the PQ centroids " << std::endl;
    assign_residual(base_residual.data(), PQ_centroids.data(), dimension, PQ_ids.data(), nc_PQ, nb);

    std::vector<std::vector<size_t>> clusters(nc);
    for (size_t i = 0; i < nb; i++){
        clusters[base_ids[i]].push_back(i);
    }

    std::ifstream query_input(path_query, std::ios::binary);
    std::vector<float> query_vectors(nq * dimension);
    readXvecFvec<float>(query_input, query_vectors.data(), dimension, nq, true, false);

    PrintMessage("Loading groundtruth");
    std::vector<uint32_t> groundtruth(nq * ngt);
    {
        std::ifstream gt_input(path_gt, std::ios::binary);
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, true, true);
    }

    faiss::IndexFlatL2 centroid_index(dimension);
    centroid_index.add(nc, index_centroids.data());

    std::vector<float> compressed_vectors(nb * dimension);
    get_base_vectors(base_ids.data(), PQ_ids.data(), index_centroids.data(), PQ_centroids.data(), compressed_vectors.data(), dimension, nb, nc_PQ);


        size_t nc_to_visit = nc / 10;
        std::vector<std::vector<float>> correct_num1(nq);
        std::vector<std::vector<float>> correct_num10(nq);
        std::vector<std::vector<float>> correct_num100(nq);
        std::vector<std::vector<float>> visited_num(nq);

#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        correct_num1[i].resize(nc_to_visit, 0);
        correct_num10[i].resize(nc_to_visit, 0);
        correct_num100[i].resize(nc_to_visit, 0);
        visited_num[i].resize(nc_to_visit, 0);

        size_t visited_vectors = 0;

        std::unordered_set<idx_t> gt1;
        for (size_t j = 0; j < 1; j++){
            gt1.insert(groundtruth[i * ngt + j]);
        }

        std::unordered_set<idx_t> gt10;
        for (size_t j = 0; j < 10; j++){
            gt10.insert(groundtruth[i * ngt + j]);
        }

        std::unordered_set<idx_t> gt100;
        for (size_t j = 0; j < 100; j++){
            gt100.insert(groundtruth[i * ngt + j]);
        }


        std::vector<float> result_dists1(1);
        std::vector<idx_t> result_labels1(1);

        std::vector<float> result_dists10(10);
        std::vector<idx_t> result_labels10(10);

        std::vector<float> result_dists100(100);
        std::vector<idx_t> result_labels100(100);


        faiss::maxheap_heapify(1, result_dists1.data(), result_labels1.data());
        faiss::maxheap_heapify(10, result_dists10.data(), result_labels10.data());
        faiss::maxheap_heapify(100, result_dists100.data(), result_labels100.data());


        std::vector<float> distance(nc);
        std::vector<idx_t> query_ids(nc);
        centroid_index.search(1, query_vectors.data()+i * dimension, nc, distance.data(), query_ids.data());
        for (size_t j = 0; j < nc_to_visit; j++){
            idx_t centroid_id = query_ids[j];

            size_t cluster_size = clusters[centroid_id].size();
            visited_vectors += cluster_size;
            
            for (size_t k = 0; k < cluster_size; k++){
                float dist = faiss::fvec_L2sqr(query_vectors.data() + i * dimension, compressed_vectors.data() + clusters[centroid_id][k] * dimension, dimension);

                if (dist < result_dists1[0]){
                    faiss::maxheap_pop(1, result_dists1.data(), result_labels1.data());
                    faiss::maxheap_push(1, result_dists1.data(), result_labels1.data(), dist, clusters[centroid_id][k]);
                }

                if (dist < result_dists10[0]){
                    faiss::maxheap_pop(10, result_dists10.data(), result_labels10.data());
                    faiss::maxheap_push(10, result_dists10.data(), result_labels10.data(), dist, clusters[centroid_id][k]);
                }

                if (dist < result_dists100[0]){
                    faiss::maxheap_pop(100, result_dists100.data(), result_labels100.data());
                    faiss::maxheap_push(100, result_dists100.data(), result_labels100.data(), dist, clusters[centroid_id][k]);
                }
            }
            visited_num[i][j] += visited_vectors;

            for (size_t k = 0; k < 1; k++){
                if (gt1.count(result_labels1[k]) != 0){
                    correct_num1[i][j] += 1;
                }
            }

            for (size_t k = 0; k < 10; k++){
                if (gt10.count(result_labels10[k]) != 0){
                    correct_num10[i][j] += 1;
                }
            }
            for (size_t k = 0; k < 100; k++){
                if (gt100.count(result_labels100[k]) != 0){
                    correct_num100[i][j] += 1;
                }
            }
        }
    }


for (size_t recall_index = 0; recall_index < recall_k_list.size(); recall_index++){
    size_t recall_k = recall_k_list[recall_index];
    std::vector<float> sum_visited_num(nc_to_visit, 0);
    std::vector<float> sum_correct_num(nc_to_visit, 0);

    if (recall_k ==1)
    for (size_t i = 0; i < nc_to_visit; i++){
        for (size_t j = 0; j < nq; j++){
            sum_correct_num[i] += correct_num1[j][i];
            sum_visited_num[i] += visited_num[j][i];
        }
    }
    else if (recall_k == 10)
    for (size_t i = 0; i < nc_to_visit; i++){
        for (size_t j = 0; j < nq; j++){
            sum_correct_num[i] += correct_num10[j][i];
            sum_visited_num[i] += visited_num[j][i];
        }
    }
    else if (recall_k == 100)
    for (size_t i = 0; i < nc_to_visit; i++){
        for (size_t j = 0; j < nq; j++){
            sum_correct_num[i] += correct_num100[j][i];
            sum_visited_num[i] += visited_num[j][i];
        }
    }

    record_file << "result for recall@ " << recall_k << std::endl;
    std::cout << "result for recall@ " << recall_k << std::endl;
    for (size_t i = 0; i < nc_to_visit; i++){
        std::cout << (sum_visited_num[i] / nq) << " ";
        record_file << (sum_visited_num[i] / nq) << " ";
    }
    std::cout << std::endl;
    record_file << std::endl;

    for (size_t i = 0; i < nc_to_visit; i++){
        std::cout << sum_correct_num[i] / recall_k / nq << " ";
        record_file << sum_correct_num[i] / recall_k / nq<< " ";
    }
    std::cout << std::endl;
    record_file << std::endl;

}
}
}