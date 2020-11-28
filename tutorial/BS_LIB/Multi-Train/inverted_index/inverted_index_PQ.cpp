#include <iostream>
#include "../../utils/utils.h"
#include "../../parameters/parameter_tuning/inverted_index/Inverted_Index.h"
#include "inverted_index_PQ.h"
#include <unordered_set>


// ids: sizeof(idx) * nb * M
void assign_residual(const float * residuals, const float * centroid, size_t dimension, idx_t * PQ_ids, size_t nc_PQ, size_t nb){
    std::cout << "Entering the assign residual function " << std::endl;
    size_t dimension_sub = dimension / M;
    
    for (size_t PQ_index = 0; PQ_index < M; PQ_index++){
        faiss::IndexFlatL2 index_sub(dimension_sub);
        index_sub.add(nc_PQ, centroid + PQ_index * nc_PQ * dimension_sub);
        std::vector<float> distance(nb);
        for (size_t i = 0; i < nb; i++){
            index_sub.search(1, residuals + i * dimension + PQ_index * dimension_sub, 1, distance.data()+i, PQ_ids + PQ_index * nb + i);
        }
    }
}

using namespace bslib;


int main(){

    time_t now = time(0);
   
   // 把 now 转换为字符串形式
    char* dt = ctime(&now);

    std::vector<float> base_vectors(dimension * nb);
    std::vector<idx_t> base_ids(nb);
    std::vector<float> base_residual(dimension * nb);
    std::ifstream base_input(path_base, std::ios::binary);
    readXvecFvec<float>(base_input, base_vectors.data(), dimension, nb, true);
    std::string path_record = "./record/inverted_index_PQ" + std::string(dt) + ".txt";
    std::ofstream record_file(path_record);

    if (dimension % M != 0){
        std::cout << "The dimension " << dimension << " should be an integer multiple of M: " << M << std::endl;
        exit(0); 
    }

    for (size_t nc = nc_low; nc <= nc_up; nc += nc_step){
        record_file << "Kmeans with centroids: " << nc << std::endl;

    std::vector<float> PQ_centroids(dimension * nc_PQ);
    std::vector<idx_t> PQ_ids(nb * M);

    std::vector<float> index_centroids(dimension * nc);
    std::cout << "Initializing the index centroids " << std::endl;
    initialize_centroid(base_vectors.data(), index_centroids.data(), nb, dimension, nc);

    std::cout << "Assigning the vectors " << std::endl;
    assign_vector(index_centroids.data(), base_vectors.data(), dimension, base_ids.data(), nc, nb);

    std::cout << "Computing the residual " << std::endl;
    compute_index_residual(base_vectors.data(), index_centroids.data(), base_ids.data(), base_residual.data(), nb, dimension);

    std::cout << "Initializing the PQ centroids " << std::endl;
    initialize_PQ_centroid(base_residual.data(), PQ_centroids.data(), nb, dimension, nc_PQ);

    std::cout << "Assinging the residuals to the PQ centroids " << std::endl;
    assign_residual(base_residual.data(), PQ_centroids.data(), dimension, PQ_ids.data(), nc_PQ, nb);

    std::vector<float> PQ_residual(nb * dimension); // This is the residual between base vector and the PQ centroids

    record_file << "Distance computation" << std::endl;
    std::cout << "Start the training process " << std::endl;
    for (size_t iter = 0; iter < total_iter; iter++){
        compute_PQ_residual(base_vectors.data(), PQ_centroids.data(), PQ_ids.data(), PQ_residual.data(), nb, dimension, nc_PQ);
        for (size_t iter1 = 0; iter1 < index_iter; iter1++){
            update_index_centroids(base_vectors.data(), PQ_residual.data(), index_centroids.data(), base_ids.data(), nb, nc, dimension);
            assign_vector(index_centroids.data(), base_vectors.data(), dimension, base_ids.data(), nc, nb);
        }

        compute_index_residual(base_vectors.data(), index_centroids.data(), base_ids.data(), base_residual.data(), nb, dimension);
        for (size_t iter2 = 0; iter2 < PQ_iter; iter2++){
            update_PQ_centroids(base_residual.data(), PQ_centroids.data(), PQ_ids.data(), nb, nc_PQ, dimension);
            assign_residual(base_residual.data(), PQ_centroids.data(), dimension, PQ_ids.data(), nc_PQ, nb);
        }

        std::cout << "Start evaluation " << std::endl;
        std::vector<float> distance_metric(2,0);
        metric_computation(base_vectors.data(), index_centroids.data(), base_ids.data(), PQ_centroids.data(), PQ_ids.data(), nb, nc_PQ, dimension, distance_metric.data());
        record_file << distance_metric[0] << " " << distance_metric[1] << std::endl;
    }


    std::vector<std::vector<size_t>> clusters(nc);
    for (size_t i = 0; i < nb; i++){
        clusters[base_ids[i]].push_back(i);
    }

    std::vector<float> correct_num(nc, 0);
    std::vector<float> visited_num(nc, 0);

    std::ifstream query_input(path_query, std::ios::binary);
    std::vector<float> query_vectors(nq * dimension);
    readXvecFvec<float>(query_input, query_vectors.data(), dimension, nq, true, false);

    PrintMessage("Loading groundtruth");
    std::vector<uint32_t> groundtruth(nq * ngt);
    {
        std::ifstream gt_input(path_gt, std::ios::binary);
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, false, false);
    }

    faiss::IndexFlatL2 centroid_index(dimension);
    centroid_index.add(nc, index_centroids.data());

    std::vector<float> compressed_vectors(nb * dimension);
    get_base_vectors(base_ids.data(), PQ_ids.data(), index_centroids.data(), PQ_centroids.data(), compressed_vectors.data(), dimension, nb, nc_PQ);

    for (size_t recall_index = 0; recall_index < recall_k_list.size(); recall_index++){
        size_t recall_k = recall_k_list[recall_index];
    for (size_t i = 0; i < nq; i++){
        size_t visited_vectors = 0;
        std::unordered_set<idx_t> gt;
        for (size_t j = 0; j < recall_k; j++){
            gt.insert(groundtruth[i * ngt + j]);
        }
        std::vector<float> result_dists(recall_k);
        std::vector<idx_t> result_labels(recall_k);
        faiss::maxheap_heapify(recall_k, result_dists.data(), result_labels.data());

        std::vector<float> distance(nc);
        std::vector<idx_t> query_ids(nc);
        centroid_index.search(1, query_vectors.data()+i * dimension, nc, distance.data(), query_ids.data());
        for (size_t j = 0; j < nc; j++){
            idx_t centroid_id = query_ids[j];

            size_t cluster_size = clusters[centroid_id].size();
            visited_vectors += cluster_size;
            visited_num[j] += visited_vectors;
            for (size_t k = 0; k < cluster_size; k++){
                float dist = faiss::fvec_L2sqr(query_vectors.data() + i * dimension, compressed_vectors.data() + clusters[centroid_id][k] * dimension, dimension);
                if (dist < result_dists[0]){
                    faiss::maxheap_pop(recall_k, result_dists.data(), result_labels.data());
                    faiss::maxheap_push(recall_k, result_dists.data(), result_labels.data(), dist, clusters[centroid_id][k]);
                }
            }
            for (size_t k = 0; k < recall_k; k++){
                if (gt.count(result_labels[k]) != 0){
                    correct_num[j] += 1;
                }
            }
        }
    }

    record_file << "result for recall@ " << recall_k << std::endl;
    for (size_t i = 0; i < nc / 10; i++){
        std::cout << visited_num[i * 10] / nq << " ";
        record_file << visited_num[i * 10] / nq << " ";
    }
    std::cout << visited_num[nc-1] / nq << " " << std::endl;
    record_file << visited_num[nc-1] / nq << " " << std::endl;

    for (size_t i = 0; i < nc / 10; i++){
        std::cout << correct_num[i * 10] / recall_k / nq << " ";
        record_file << correct_num[i * 10] / recall_k / nq<< " ";
    }
    std::cout << correct_num[nc-1] / recall_k / nq << " " << std::endl;
    record_file << correct_num[nc-1] / recall_k / nq << " " << std::endl;
}
}
}