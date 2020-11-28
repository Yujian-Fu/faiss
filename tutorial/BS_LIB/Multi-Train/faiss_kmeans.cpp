


#include <iostream>
#include "../utils/utils.h"
#include "../parameters/parameter_tuning/inverted_index/Inverted_Index.h"
#include "inverted_index_PQ.h"
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

            size_t cluster_size = clusters[j].size();
            visited_vectors += cluster_size;
            visited_num[j] += visited_vectors;
            for (size_t k = 0; k < cluster_size; k++){
                float dist = faiss::fvec_L2sqr(query_vectors.data() + i * dimension, compressed_vectors.data() + clusters[j][k] * dimension, dimension);
                if (dist < result_dists[0]){
                    faiss::maxheap_pop(recall_k, result_dists.data(), result_labels.data());
                    faiss::maxheap_push(recall_k, result_dists.data(), result_labels.data(), dist, clusters[j][k]);
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