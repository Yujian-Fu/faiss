#include<faiss/IndexFlat.h>
#include<faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <unordered_set>
#include <faiss/utils/distances.h>
#include "../utils/utils.h"
#include <faiss/VectorTransform.h>
#include <algorithm>


typedef faiss::Index::idx_t idx_t;

using namespace bslib;
int main(){
    int dimension = 128;                   // dimension
    int nb = 1000000;                       // database size
    int nq = 1000;                         // nb of queries
    size_t nlist = 1000;
    size_t M = 16;
    size_t nbits = 8;
    size_t nprobe = 100;
    size_t sum_correctness = 0;
    size_t ksub = 0;
    size_t code_size = 0;

    std::string dataset = "SIFT1M";

    std::string path_base = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
    std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";
    std::string path_record = "/home/y/yujianfu/ivf-hnsw/models_VQ/" + dataset + "/recording_reranking_paras_" + std::to_string(M) + "_" + std::to_string(nbits) + "_" + ".txt";
    
    float *xb = new float[dimension * nb];
    float *xq = new float[dimension * nq];
    std::ifstream base_file(path_base, std::ios::binary); std::ifstream query_file(path_query, std::ios::binary);
    readXvec<float>(base_file, xb, dimension, nb, false, false);
    readXvec<float>(query_file, xq, dimension, nq, false, false);

    size_t k_result = 100;
    time_recorder Trecorder = time_recorder();

    std::ofstream record_file;
    record_file.open(path_record, std::ios::out);

    faiss::IndexFlatL2 index_flat(dimension);
    std::vector<idx_t> labels(k_result * nq);
    std::vector<float> dists(k_result * nq);
    
    Trecorder.reset();
    index_flat.add(nb, xb);
    index_flat.search(nq, xq, k_result, dists.data(), labels.data());
    Trecorder.print_time_usage("Searching for flat index");


    faiss::IndexFlatL2 quantizer(dimension);

    faiss::IndexIVFPQ index_pq(&quantizer, dimension, nlist, M, nbits);
    ksub = index_pq.pq.ksub;
    code_size = index_pq.pq.code_size;
    index_pq.verbose = true;
    index_pq.train(nb / 100, xb);
    index_pq.add(nb, xb);

    index_pq.nprobe = nprobe;
    Trecorder.reset();


    std::vector<idx_t> base_labels(nb);
    std::vector<float> base_dists(nb);

    quantizer.search(nb, xb, 1, base_dists.data(), base_labels.data());

    std::vector<std::vector<idx_t>> inverted_index(nlist);
    // Build inverted index list
    for (size_t i = 0; i < nb; i++){
        inverted_index[base_labels[i]].push_back(i);
    }

    // Compute the residual and encode the residual
    std::vector<float> residual(dimension * nb);
    quantizer.compute_residual_n(nb, xb, residual.data(), base_labels.data());

    std::vector<uint8_t> residual_code(code_size * nb);
    index_pq.pq.compute_codes(residual.data(), residual_code.data(), nb);


    /**
     * The size for one distance table is M * ksub 
     * every value is the distance between sub-query and 
     * sub-centroid. Add the index of one query to the 
     * distance table to get the sum distance
     * 
     * distance = ||query - (centroids + residual_PQ)||^2 
     *          = ||query - centroids||^2 + ||residual_PQ|| ^2 - 2 * (query - centroids) * residual_PQ 
     *          = ||query - centroids||^2 + ||residual_PQ|| ^2 + 2 * centroids * residual_PQ - 2 * query * residual_PQ
     *          = ||query - centroids||^2 + ||residual_PQ + centroids|| ^2 - ||centroids||^2 - 2 * query * residual_PQ
     *          = ... + sum{}
     * So you need the norm of centroid and base vector 
     **/ 

    std::vector<float> pre_computed_tables(nlist * M * ksub);
    size_t dsub = dimension / M;
    for (size_t i = 0; i < nlist; i++){
        std::vector<float> quantizer_centroid(dimension);
        quantizer.reconstruct(i, quantizer_centroid.data());
        for (size_t m = 0; m < M; m++){
            const float * quantizer_sub_centroid = quantizer_centroid.data() + m * dsub;
            
            for (size_t k = 0; k < ksub; k++){
                const float * pq_sub_centroid = index_pq.pq.get_centroids(m, k);
                float residual_PQ_norm = faiss::fvec_norm_L2sqr(pq_sub_centroid, dsub);
                float prod_quantizer_pq = faiss::fvec_inner_product(quantizer_sub_centroid, pq_sub_centroid, dsub);
                pre_computed_tables[i * M * ksub + m * ksub + k] = residual_PQ_norm + 2 * prod_quantizer_pq;
            }
        }
    }

    Trecorder.reset();

    std::vector <idx_t> result_labels(nq * k_result);
    std::vector <float> result_dists(nq * k_result);
    record_file << "M: " << M << " nbits: " << nbits << std::endl;
//#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        record_file << "Query: " << i << std::endl;
        const float * query = xq + i * dimension ;
        for (size_t j = 0; j < dimension; j++) {record_file << query[j] << " ";} record_file << std::endl;

        const size_t result_position = i * k_result;

        std::vector <idx_t> query_labels(nprobe);
        std::vector <float> query_dists(nprobe);
        std::vector<float> distance_table(M * ksub);
        index_pq.pq.compute_inner_prod_table(query, distance_table.data());

        faiss::maxheap_heapify(k_result, result_dists.data() + result_position, result_labels.data() + result_position);
        quantizer.search(1, query, nprobe, query_dists.data(), query_labels.data());
        std::vector<float> computed_distance;
        std::vector<idx_t> computed_label;
        size_t visited_vectors = 0;
        size_t update_times = 0;

        for (size_t j = 0; j < nprobe; j++){
            size_t group_label = query_labels[j];
            size_t group_size = inverted_index[group_label].size();
            float qc_dist = query_dists[j];
            visited_vectors += group_size;

            for (size_t k = 0; k < group_size; k++){
                float sum_distance = 0;
                float sum_prod_distance = 0;
                float table_distance = 0;
                idx_t sequence_id = inverted_index[group_label][k];

                uint8_t * base_code = residual_code.data() + sequence_id * code_size;

                for (size_t l = 0; l < M; l++){
                    sum_prod_distance += distance_table[l * ksub + base_code[l]];
                    table_distance += pre_computed_tables[group_label * M * ksub + l * ksub + base_code[l]];
                }

                sum_distance = qc_dist + table_distance - 2 * sum_prod_distance;
                computed_distance.push_back(sum_distance);
                computed_label.push_back(sequence_id);

                if (sum_distance < result_dists[result_position]){
                    update_times ++;
                    faiss::maxheap_pop(k_result, result_dists.data() + result_position, result_labels.data() + result_position);
                    faiss::maxheap_push(k_result, result_dists.data() + result_position, result_labels.data() + result_position, sum_distance, sequence_id);
                }
            }

            if (j == nprobe / 2){
                std::vector<idx_t> search_dist_index(visited_vectors);
                size_t x = 0;
                std::iota(search_dist_index.begin(), search_dist_index.end(), x++);
                std::sort(search_dist_index.begin(), search_dist_index.end(), [&](int i,int j){return computed_distance[i]<computed_distance[j];});

                record_file << "n / 2 d_1st: " << computed_distance[search_dist_index[0]] << std::endl;
                record_file << "n / 2 d_10th: " << computed_distance[search_dist_index[9]] << std::endl;
                record_file << "n / 2 d_1st / d_10th " << computed_distance[search_dist_index[0]] / computed_distance[search_dist_index[9]] << std::endl;
                record_file << "search times / n / 2 update times " << float(visited_vectors) / float(update_times) << std::endl;
            }
        }
        std::vector<idx_t> search_dist_index(visited_vectors);
        size_t x = 0;
        std::iota(search_dist_index.begin(), search_dist_index.end(), x++);
        std::sort(search_dist_index.begin(), search_dist_index.end(), [&](int i,int j){return computed_distance[i]<computed_distance[j];});

        record_file << "n d_1st: " << computed_distance[search_dist_index[0]] << std::endl;
        record_file << "n d_10th: " << computed_distance[search_dist_index[9]] << std::endl;
        record_file << "n d_1st / d_10th " << computed_distance[search_dist_index[0]] / computed_distance[search_dist_index[9]] << std::endl;
        record_file << "search times / n update times " << float(visited_vectors) / float(update_times) << std::endl;
        record_file << "Visited vectors: " << visited_vectors << std::endl;

        const size_t target_size = 13;
        size_t gt_target[target_size] = {1, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        for(size_t index = 0; index < target_size; index++){
            size_t visited_gt = 0;
            size_t reranking_space = visited_vectors;

            std::unordered_set<idx_t> gt_set;
            for (size_t j = 0; j < gt_target[index]; j++){
                gt_set.insert(labels[i * k_result + j]);
            }
            for (size_t j = 0; j < visited_vectors; j++){
                if (gt_set.count(computed_label[search_dist_index[j]]) != 0){
                    visited_gt ++;
                }
                if (visited_gt >= gt_target[index]){
                    reranking_space = j;
                    break;
                }
            }

            record_file << gt_target[index] << " " << reranking_space << " " << visited_gt << std::endl;
        }
    }
    Trecorder.print_time_usage("Finished IVFPQ search");
    sum_correctness = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt_set;

        for (size_t j = 0; j < k_result; j++){
            gt_set.insert(labels[i * k_result + j]);
        }

        for (size_t j = 0; j < k_result; j++){
            if (gt_set.count(result_labels[i * k_result + j]) != 0){
                sum_correctness ++;
            }
        }
    }
    std::cout << "The recall for IVFPQ implementation is: " << float(sum_correctness) / (nq * k_result) << "for k = " << k_result << std::endl;
}