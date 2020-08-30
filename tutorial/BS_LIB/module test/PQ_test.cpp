#include<faiss/IndexFlat.h>
#include<faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <unordered_set>
#include <faiss/utils/distances.h>
#include "../utils/utils.h"
#include <faiss/VectorTransform.h>
#include <algorithm>

typedef faiss::Index::idx_t idx_t;

/**
 * The result of the resize function:
 * If the second length is larger than the first length, then the other part is the provided value
 * If the second length is smaller than the first length, then all the part is the origin value
 * !!!!! fvec_madd the first parameter is dimension!!!!
 **/

/**
 * The procession of use OPQ:
 * Train:
 * use residual to train the OPQ matrix
 * Use the rotated residual to train the PQ
 * 
 * Assign:
 * assign the origin vector and origin centroids
 * 
 * Add_batch:
 * compute and rotate the residuals
 * rotate the residual with pretrained PQ
 * 
 * rotate the centroids for further searching
 **/
using namespace bslib;
int main(){

    bool use_OPQ = false;

    int dimension = 128;                   // dimension
    int nb = 100000;                       // database size
    int nq = 100;                         // nb of queries
    size_t nlist = 100;
    size_t M = 8;
    size_t nbits = 8;
    size_t nprobe = 100;
    size_t sum_correctness = 0;
    size_t ksub = 0;
    size_t code_size = 0;
    faiss::LinearTransform * OPQMatrix;

    std::string dataset = "SIFT1M";

    std::string path_base = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
    std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";
    std::string path_record = "/home/y/yujianfu/ivf-hnsw/models_VQ/" + dataset + "/recording_reranking_space_" + std::to_string(M) + "_" + std::to_string(nbits) + "_" + ".txt";
    
    
    float *xb = new float[dimension * nb];
    float *xq = new float[dimension * nq];
    std::ifstream base_file(path_base, std::ios::binary); std::ifstream query_file(path_query, std::ios::binary);
    readXvec<float>(base_file, xb, dimension, nb, false, false);
    readXvec<float>(query_file, xq, dimension, nq, false, false);
    /*
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < dimension; j++) xb[dimension * i + j] = drand48();
        xb[dimension * i] += i / 1000.;
    }
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < dimension; j++) xq[dimension * i + j] = drand48();
        xq[dimension * i] += i / 1000.;
    }*/

    size_t k_result = 10;
    time_recorder Trecorder = time_recorder();

    std::ofstream record_file;
    record_file.open(path_record, std::ios::app);

    faiss::IndexFlatL2 index_flat(dimension);
    std::vector<idx_t> labels(k_result * nq);
    std::vector<float> dists(k_result * nq);
    
    Trecorder.reset();
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
    */


    std::vector<idx_t> pq_labels(k_result * nq);
    std::vector<float> pq_dists(k_result * nq);
    faiss::IndexFlatL2 quantizer(dimension);

    faiss::IndexIVFPQ index_pq(&quantizer, dimension, nlist, M, nbits);
    ksub = index_pq.pq.ksub;
    code_size = index_pq.pq.code_size;
    index_pq.verbose = true;
    index_pq.train(nb / 10, xb);
    index_pq.add(nb, xb);

    index_pq.nprobe = nprobe;
    Trecorder.reset();
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
    
    // My implementation of IVFPQ
    //faiss::ClusteringParameters CP;
    //CP.niter = 100;
    //faiss::Clustering clus(dimension, nlist, CP);
    //faiss::IndexFlatL2 quantizer_assign(dimension);

    //clus.verbose = true;
    //clus.train(nb / 10, xb, quantizer_assign);

    faiss::ProductQuantizer * PQ = new faiss::ProductQuantizer(dimension, M, nbits);

    code_size = PQ->code_size;
    ksub = PQ->ksub;
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
    for (size_t i = 0; i < 100; i++){std::cout << residual[i] << " ";} std::cout << std::endl << std::endl;
    for (size_t i = 0; i < 100; i++){std::cout << quantizer.xb[i] << " ";} std::cout << std::endl << std::endl;


    if (use_OPQ){
        faiss::OPQMatrix * matrix = new faiss::OPQMatrix(dimension, M);
        matrix->verbose = false; matrix->max_train_points = nb / 10;
        matrix->train(nb / 10, residual.data());
        for (size_t i = 0; i < 100; i++){std::cout << matrix->A[i] << " ";} std::cout << std::endl << std::endl;

        std::vector<float> copy_residual(dimension * nb);
        memcpy(copy_residual.data(), residual.data(), nb * dimension * sizeof(float));
        OPQMatrix = matrix;
        OPQMatrix->apply_noalloc(nb, copy_residual.data(), residual.data());

        std::vector<float> copy_centrodis(nlist * dimension);
        OPQMatrix->apply_noalloc(nlist, quantizer.xb.data(), copy_centrodis.data());
        memcpy(quantizer.xb.data(), copy_centrodis.data(), nlist * dimension * sizeof(float));
    }

    for (size_t i = 0; i < 100; i++){std::cout << residual[i] << " ";} std::cout << std::endl << std::endl;
    for (size_t i = 0; i < 100; i++){std::cout << quantizer.xb[i] << " ";} std::cout << std::endl << std::endl;
    PQ->verbose = true;
    PQ->train(nb / 10, residual.data());

    
    std::vector<uint8_t> residual_code(code_size * nb);
    PQ->compute_codes(residual.data(), residual_code.data(), nb);


    /**
     * The size for one distance table is M * ksub 
     * every value is the distance between sub-query and 
     * sub-centroid. Add the index of one query to the 
     * distance table to get the sum distance
     * 
     * distance = ||query - (centroids + residual_PQ)||^2 
     *          = ||query - centroids||^2 + ||residual_PQ|| ^2 - 2 * (query - centroids) * residual_PQ 
     *          = ||query - centroids||^2 + ||residual_PQ|| ^2 + 2 * centroids * residual_PQ - 2 * query * residual_PQ
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
                const float * pq_sub_centroid = PQ->get_centroids(m, k);
                float residual_PQ_norm = faiss::fvec_norm_L2sqr(pq_sub_centroid, dsub);
                float prod_quantizer_pq = faiss::fvec_inner_product(quantizer_sub_centroid, pq_sub_centroid, dsub);
                pre_computed_tables[i * M * ksub + m * ksub + k] = residual_PQ_norm + 2 * prod_quantizer_pq;
            }
        }
    }

    Trecorder.reset();
    std::vector <idx_t> result_labels(nq * k_result);
    std::vector <float> result_dists(nq * k_result);
#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        const float * query = use_OPQ ? OPQMatrix->apply(1, xq + i * dimension) : xq + i * dimension ;
        const size_t result_position = i * k_result;

        std::vector <idx_t> query_labels(nprobe);
        std::vector <float> query_dists(nprobe);
        std::vector<float> distance_table(M * ksub);
        PQ->compute_inner_prod_table(query, distance_table.data());

        faiss::maxheap_heapify(k_result, result_dists.data() + result_position, result_labels.data() + result_position);
        quantizer.search(1, query, nprobe, query_dists.data(), query_labels.data());
        //std::vector<float> computed_distance;
        std::vector<idx_t> computed_label;
        size_t visited_vectors = 0;
        

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
                //computed_distance.push_back(sum_distance);
                //computed_label.push_back(sequence_id);

                if (sum_distance < result_dists[result_position]){
                    faiss::maxheap_pop(k_result, result_dists.data() + result_position, result_labels.data() + result_position);
                    faiss::maxheap_push(k_result, result_dists.data() + result_position, result_labels.data() + result_position, sum_distance, sequence_id);
                }
            }
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
    std::cout << "The recall for IVFPQ implementation is: " << float(sum_correctness) / (nq * k_result) << std::endl;

}