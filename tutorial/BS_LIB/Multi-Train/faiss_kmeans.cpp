


#include <iostream>
#include "../utils/utils.h"
#include "../parameters/parameter_tuning/inverted_index/Inverted_Index.h"
#include "parameter_multi.h"
#include <unordered_set>


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
    std::string path_record = "./record/faiss_kmeans_" + std::string(dt) + ".txt";
    std::ofstream record_file(path_record);

    if (dimension % M != 0){
        std::cout << "The dimension " << dimension << " should be an integer multiple of M: " << M << std::endl;
        exit(0); 
    }

    for (size_t nc = nc_low; nc <= nc_up; nc += nc_step){
        record_file << "Kmeans with centroids: " << nc << std::endl;

    std::vector<float> index_centroids(dimension * nc);
    faiss::kmeans_clustering(dimension, nb, nc, base_vectors.data(), index_centroids.data(), index_iter * total_iter);


    std::cout << "Assigning the vectors " << std::endl;
    assign_vector(index_centroids.data(), base_vectors.data(), dimension, base_ids.data(), nc, nb);

    float b_c_distance = 0;
    for (size_t i = 0; i < nb; i++){
        idx_t base_id = base_ids[i];
        b_c_distance += faiss::fvec_L2sqr(base_vectors.data() + i * dimension, index_centroids.data() + base_id * dimension, dimension);
    }
    
    std::cout << "Now the avergae b_c_dist is: " << b_c_distance / nb << std::endl;
    record_file << "Index distance: " << b_c_distance / nb << std::endl;
    
    std::cout << "Computing the residual " << std::endl;
    compute_index_residual(base_vectors.data(), index_centroids.data(), base_ids.data(), base_residual.data(), nb, dimension);

    faiss::ProductQuantizer pq = faiss::ProductQuantizer(dimension, M, nbits);

    pq.verbose = false;
    pq.train(nb, base_residual.data());

    size_t code_size = pq.code_size;
    std::vector<uint8_t> base_codes(code_size * nb);
    pq.compute_codes(base_residual.data(), base_codes.data(), nb);

    std::vector<float> compressed_vectors(nb * dimension);
    pq.decode(base_codes.data(), compressed_vectors.data(), nb);

    float pq_distance = 0;
    pq_distance = faiss::fvec_L2sqr(base_residual.data(), compressed_vectors.data(), nb * dimension);
    std::cout << "The pq distance: " << pq_distance / nb << std::endl;
    record_file << "PQ distance: " << pq_distance / nb << std::endl;


    for (size_t i = 0; i < nb; i++){
        idx_t centroid_id = base_ids[i];
        faiss::fvec_madd(dimension, compressed_vectors.data() + i * dimension, 1.0, index_centroids.data() + centroid_id * dimension, compressed_vectors.data() + i * dimension);
    }

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
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, false, false);
    }

    faiss::IndexFlatL2 centroid_index(dimension);
    centroid_index.add(nc, index_centroids.data());


        std::vector<std::vector<float>> correct_num1(nq);
        std::vector<std::vector<float>> correct_num10(nq);
        std::vector<std::vector<float>> correct_num100(nq);
        std::vector<std::vector<float>> visited_num(nq);

#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        correct_num1[i].resize(nc, 0);
        correct_num10[i].resize(nc, 0);
        correct_num100[i].resize(nc, 0);
        visited_num[i].resize(nc, 0);

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
        for (size_t j = 0; j < nc; j++){
            idx_t centroid_id = query_ids[j];

            size_t cluster_size = clusters[centroid_id].size();
            visited_vectors += cluster_size;
            visited_num[i][j] += visited_vectors;
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
    std::vector<float> sum_visited_num(nc, 0);
    std::vector<float> sum_correct_num(nc, 0);

    if (recall_k ==1)
    for (size_t i = 0; i < nc; i++){
        for (size_t j = 0; j < nq; j++){
            sum_correct_num[i] += correct_num1[j][i];
            sum_visited_num[i] += visited_num[j][i];
        }
    }
    else if (recall_k == 10)
    for (size_t i = 0; i < nc; i++){
        for (size_t j = 0; j < nq; j++){
            sum_correct_num[i] += correct_num10[j][i];
            sum_visited_num[i] += visited_num[j][i];
        }
    }
    else if (recall_k == 100)
    for (size_t i = 0; i < nc; i++){
        for (size_t j = 0; j < nq; j++){
            sum_correct_num[i] += correct_num100[j][i];
            sum_visited_num[i] += visited_num[j][i];
        }
    }

    record_file << "result for recall@ " << recall_k << std::endl;
    for (size_t i = 0; i < nc / 10; i++){
        std::cout << size_t(sum_visited_num[i * 10] / nq) << " ";
        record_file << size_t(sum_visited_num[i * 10] / nq) << " ";
    }
    std::cout << sum_visited_num[nc-1] / nq << " " << std::endl;
    record_file << sum_visited_num[nc-1] / nq << " " << std::endl;

    for (size_t i = 0; i < nc / 10; i++){
        std::cout << sum_correct_num[i * 10] / recall_k / nq << " ";
        record_file << sum_correct_num[i * 10] / recall_k / nq<< " ";
    }
    std::cout << sum_correct_num[nc-1] / recall_k / nq << " " << std::endl;
    record_file << sum_correct_num[nc-1] / recall_k / nq << " " << std::endl;

}
}
}