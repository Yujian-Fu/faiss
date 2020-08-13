#include<faiss/IndexFlat.h>
#include<faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <unordered_set>
#include <faiss/utils/distances.h>
#include "../utils/utils.h"

typedef faiss::Index::idx_t idx_t;

/**
 * The result of the resize function:
 * If the second length is larger than the first length, then the other part is the provided value
 * If the second length is smaller than the first length, then all the part is the origin value
 * !!!!! fvec_madd the first parameter is dimension!!!!
 **/
using namespace bslib;
int main(){

    int dimension = 128;                   // dimension
    int nb = 100000;                       // database size
    int nq = 1000;                         // nb of queries
    size_t nlist = 100;
    size_t M = 8;
    size_t nbits = 8;
    size_t nprobe = 10;
    size_t sum_correctness = 0;

    float *xb = new float[dimension * nb];
    float *xq = new float[dimension * nq];
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < dimension; j++) xb[dimension * i + j] = drand48();
        xb[dimension * i] += i / 1000.;
    }
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < dimension; j++) xq[dimension * i + j] = drand48();
        xq[dimension * i] += i / 1000.;
    }

    size_t k_result = 10;
    time_recorder Trecorder = time_recorder();

    Trecorder.reset();
    faiss::IndexFlatL2 index_flat(dimension);
    std::vector<idx_t> labels(k_result * nq);
    std::vector<float> dists(k_result * nq);
    Trecorder.print_time_usage("Training flat index");
    

    index_flat.add(nb, xb);
    index_flat.search(nq, xq, k_result, dists.data(), labels.data());
    Trecorder.print_time_usage("Searching for flat index");

    /*
    size_t M_HNSW = 100;
    std::vector<idx_t> HNSW_labels(k_result * nq);
    std::vector<float> HNSW_dists(k_result * nq);
    faiss::IndexHNSWFlat index_HNSW(dimension, M_HNSW);
    
    index_HNSW.train(nb / 10, xb);
    index_HNSW.add(nb, xb);

    index_HNSW.search(nq, xq, k_result, HNSW_dists.data(), HNSW_labels.data());

    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt_set;
        for (size_t j = 0; j < k_result; j++){
            gt_set.insert(labels[i * k_result + j]);
        }
        for (size_t j = 0; j < k_result; j++){
            if (gt_set.count(HNSW_labels[i * k_result + j]) != 0){
                sum_correctness ++;
            }
        }
    }
    std::cout << "The recall for HNSW k = " << k_result << " is: " << float(sum_correctness) / (k_result * nq) << std::endl; 


    std::vector<idx_t> pq_labels(k_result * nq);
    std::vector<float> pq_dists(k_result * nq);
    faiss::IndexFlatL2 quantizer(dimension);
    faiss::IndexIVFPQ index_pq(&quantizer, dimension, nlist, M, nbits);
    index_pq.verbose = true;
    index_pq.train(nb / 10, xb);
    index_pq.add(nb, xb);
    Trecorder.print_time_usage("Training PQ index");


    index_pq.nprobe = nprobe;
    index_pq.search(nq, xq, k_result, pq_dists.data(), pq_labels.data());
    Trecorder.print_time_usage("Searching for PQ index");

    sum_correctness = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt_set;
        for (size_t j = 0; j < k_result; j++){
            gt_set.insert(labels[i * k_result + j]);
        }
        for (size_t j = 0; j < k_result; j++){
            if (gt_set.count(pq_labels[i * k_result + j]) != 0){
                sum_correctness ++;
            }
        }
    }

    std::cout << "The recall for PQ k = " << k_result << " is: " << float(sum_correctness) / (k_result * nq) << std::endl; 
    */

    // My implementation of IVFPQ
    faiss::ClusteringParameters CP; // Try different settings of CP
    CP.niter = 40;
    faiss::Clustering clus(dimension, nlist, CP);
    faiss::IndexFlatL2 index_assign(dimension);
    clus.verbose =  true;
    clus.train(nb / 10, xb, index_assign);
    
    faiss::ProductQuantizer PQ(dimension, M, nbits);
    size_t code_size = PQ.code_size;
    std::vector<idx_t> base_labels(nb);
    std::vector<float> base_dists(nb);

    Trecorder.reset();
    index_assign.search(nb, xb, 1, base_dists.data(), base_labels.data());
    Trecorder.print_time_usage("Assigned the base vectors");

    std::vector<std::vector<idx_t>> inverted_index(nlist);
    // Build inverted index list
    std::vector <size_t> inverted_index_accumulation(nlist, 0);
    for (size_t i = 0; i < nb; i++){
        inverted_index_accumulation[base_labels[i]] ++;
    }
    for (size_t i = 0; i < nlist; i++){
        inverted_index[i].resize(inverted_index_accumulation[i]);
    }
    std::vector<size_t> inverted_index_pointer(nlist, 0);
    
    for (size_t i = 0; i < nb; i++){
        size_t group_label = base_labels[i];
        inverted_index[group_label][inverted_index_pointer[group_label]] = i;
        inverted_index_pointer[group_label] ++;
    }
    Trecorder.print_time_usage("Constructed Inverted Index");

    // Compute the residual and encode the residual
    std::vector<float> residual(dimension * nb);
#pragma omp parallel for
    for (size_t i = 0; i < nb; i++){
        idx_t group_label = base_labels[i];
        faiss::fvec_madd(dimension, xb + i * dimension, -1.0, index_assign.xb.data() + group_label * dimension, residual.data() + i * dimension); 
    }

    PQ.verbose = true;
    PQ.train(nb / 10, residual.data());
    Trecorder.print_time_usage("Trained PQ");

    std::vector<uint8_t> residual_code(code_size * nb);
    PQ.compute_codes(residual.data(), residual_code.data(), nb);
    std::vector<float> centroid_norm(nlist, 0);
    std::vector<float> base_norm(nb, 0);

    faiss::fvec_norms_L2sqr(centroid_norm.data(), index_assign.xb.data(), dimension, nlist);
    faiss::fvec_norms_L2sqr(base_norm.data(), xb, dimension, nb);


    /**
     * The size for one distance table is M * ksub 
     * every value is the distance between sub-query and 
     * sub-centroid. Add the index of one query to the 
     * distance table to get the sum distance
     * 
     * distance = ||query - (centroids + residual_PQ)||^2 
     *            ||query - centroids||^2 + ||residual_PQ|| ^2 - 2 * (query - centroids) * residual_PQ 
     *            ||query - centroids||^2 + ||residual_PQ|| ^2 - 2 * query * residual_PQ + 2 * centroids * residual_PQ 
     *            ||query - centroids||^2 + ||residual_PQ + centroids||^2 - ||centroids||^2 - 2 * query * residual_PQ  
     * So you need the norm of centroid and base vector 
     **/ 

    std::vector<size_t> query_correctness(nq, 0);
//#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        const float * query = xq + i * dimension;
        std::vector <idx_t> result_labels(k_result);
        std::vector <float> result_dists(k_result);

        std::vector <idx_t> query_labels(nprobe);
        std::vector <float> query_dists(nprobe);
        std::vector<float> distance_table(PQ.M * PQ.ksub);
        PQ.compute_inner_prod_table(query, distance_table.data());

        faiss::maxheap_heapify(k_result, result_dists.data(), result_labels.data());
        index_assign.search(1, query, nprobe, query_dists.data(), query_labels.data());
        for (size_t j = 0; j < nprobe; j++){
            size_t group_label = query_labels[j];
            size_t group_size = inverted_index[group_label].size();
            float qc_dist = query_dists[j];
            float c_norm = centroid_norm[group_label];

            for (size_t k = 0; k < group_size; k++){
                float sum_distance = 0;
                float sum_prod_distance = 0;
                idx_t sequence_id = inverted_index[group_label][k];
                float b_norm = base_norm[sequence_id];

                uint8_t * base_code = residual_code.data() + sequence_id * code_size;

                for (size_t l = 0; l < M; l++){
                    sum_prod_distance += distance_table[l * PQ.ksub + base_code[l]];
                }
                sum_distance = qc_dist + b_norm - c_norm - 2 * sum_prod_distance;


                
                std::cout << qc_dist << " " << b_norm << " " << c_norm << " " << sum_prod_distance << " " << sum_distance << " " << std::endl;
                for (size_t l = 0; l < code_size; l++){std::cout << (float) base_code[l] << " ";} std::cout << std::endl;
            
                std::vector<float> reconstructed_residual(dimension);
                PQ.decode(base_code, reconstructed_residual.data());

                float test_prod_distance = 0;
                for (size_t l = 0; l < dimension; l++){
                    test_prod_distance += reconstructed_residual[l] * query[l];
                }

                float test_prod_actual_distance = faiss::fvec_inner_product(xb + sequence_id * dimension, query, dimension);
                float test_qc_dist = faiss::fvec_L2sqr(query, index_assign.xb.data() + group_label*dimension, dimension);
                float test_qb_dist = faiss::fvec_L2sqr(query, xb + sequence_id * dimension, dimension);
                float test_b_norm = faiss::fvec_norm_L2sqr(xb + sequence_id * dimension, dimension);
                float test_c_norm = faiss::fvec_norm_L2sqr(index_assign.xb.data() + group_label * dimension, dimension);

                std::cout <<  test_qc_dist << " " << test_b_norm << " " << test_c_norm << " " <<  test_prod_distance << " " << test_prod_actual_distance << " " << test_qb_dist << std::endl;
                


                if (sum_distance < result_dists[0]){
                    faiss::maxheap_pop(k_result, result_dists.data(), result_labels.data());
                    faiss::maxheap_push(k_result, result_dists.data(), result_labels.data(), sum_distance, sequence_id);
                }
            }
        }

        // Computing the correct num
        std::unordered_set<idx_t> gt_set;
        for (size_t j = 0; j < k_result; j++){
            gt_set.insert(labels[i * k_result + j]);
        }

        for (size_t j = 0; j < k_result; j++){
            if (gt_set.count(result_labels[j]) != 0){
                query_correctness[i] ++;
            }
        }
    }
    sum_correctness = 0;
    for (size_t i = 0; i < nq; i++){
        sum_correctness += query_correctness[i];
    }
    std::cout << "The recall for IVFPQ implementation is: " << float(sum_correctness) / (nq * k_result) << std::endl;

}