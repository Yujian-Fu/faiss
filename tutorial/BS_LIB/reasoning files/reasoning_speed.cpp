#include "../utils/utils.h"
#include <unordered_set>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/ProductQuantizer.h>
#include <algorithm>
#include <cassert>
//Parameters
    
    const std::string dataset = "SIFT1M";
    const size_t dimension = 128;
    size_t train_set_size = 100000;
    const size_t base_set_size = 1000000;
    const size_t query_set_size = 1000;
    const size_t ngt = 100;
    const size_t recall_test_size = 3;
    const float MIN_DISTANCE = 1e8;
    
    
    /*
    const std::string dataset = "DEEP1M";
    const size_t dimension = 256;
    size_t train_set_size =  100000;
    const size_t base_set_size = 1000000;
    const size_t query_set_size = 1000;
    const size_t ngt = 100;
    const size_t recall_test_size = 3;
    const float MIN_DISTANCE = 100;
    */
    

    const std::string model = "models_VQ_VQ";
    const bool use_fast_assign = false;

    const std::string path_learn = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_learn.fvecs";
    const std::string path_base = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_base.fvecs";
    const std::string path_gt = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_groundtruth.ivecs";
    const std::string path_query = "/home/y/yujianfu/ivf-hnsw/data/" + dataset + "/" + dataset +"_query.fvecs";
    std::string path_record = "/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset + "/reasoning_speed.txt";
    

    std::vector<float> train_set(dimension * train_set_size);
    std::vector<float> base_set(dimension * base_set_size);
    std::vector<float> query_set(dimension * query_set_size);
    std::vector<uint32_t> gt_set(ngt * base_set_size);

    std::ifstream train_input(path_learn, std::ios::binary);
    std::ifstream base_input(path_base, std::ios::binary);
    std::ifstream gt_input(path_gt, std::ios::binary);
    std::ifstream query_input(path_query, std::ios::binary);
    std::ofstream record_output;

float pq_L2sqr(const uint8_t *code, const float * precomputed_table, size_t code_size, size_t ksub)
{
    float result = 0.;
    const size_t dim = code_size >> 2;
    size_t m = 0;
    for (size_t i = 0; i < dim; i++) {
        result += precomputed_table[ksub * m + code[m]]; m++;
        result += precomputed_table[ksub * m + code[m]]; m++;
        result += precomputed_table[ksub * m + code[m]]; m++;
        result += precomputed_table[ksub * m + code[m]]; m++;
    }
    return result;
}

typedef faiss::Index::idx_t idx_t;
using namespace bslib;
int main(){
    PrepareFolder((char *) ("/home/y/yujianfu/ivf-hnsw/" + model + "/" + dataset).c_str());
    record_output.open(path_record, std::ios::out);
    readXvec<float>(train_input, train_set.data(), dimension, train_set_size, false, false);
    readXvec<float>(base_input, base_set.data(), dimension, base_set_size, false, false);
    readXvec<uint32_t>(gt_input, gt_set.data(), ngt, query_set_size, false, false);
    readXvec<float> (query_input, query_set.data(), dimension, query_set_size, false, false);

    const size_t centroid_num[2] = {1000, 2000};
    const size_t centroid_keep_space[2] = {100, 200};
    const size_t centroid_num1[4] = {10, 100, 20, 100};
    const size_t centroid_num2[4] = {100, 10, 100, 20};
    const size_t centroid_keep_space1[4] = {2, 50, 4, 50};
    const size_t centroid_keep_space2[4] = {50, 2, 50, 4};
    const size_t repeat_time = 1000;
    record_output << "The repeat times for measuring is " << repeat_time << std::endl;

    time_recorder trecorder;
    record_output << "The searching time for " << base_set_size << " vectors" << std::endl;
    faiss::ProductQuantizer PQ(dimension, 8, 8);
    size_t code_size = PQ.code_size;

    // The 1 layer structure
    for (size_t i = 0; i < 2; i++){
        faiss::Clustering clus (dimension, centroid_num[i]);
        clus.verbose = true;
        faiss::IndexFlatL2 index(dimension);
        clus.train(train_set_size, train_set.data(), index);

        trecorder.reset();
        std::vector<idx_t> base_assigned_ids(base_set_size * centroid_keep_space[i]); std::vector<float> base_assigned_dists(base_set_size * centroid_keep_space[i]);
        index.search(base_set_size, base_set.data(), centroid_keep_space[i], base_assigned_dists.data(), base_assigned_ids.data());
        std::string message = "Finished searching 1 layer:  ncentroids: " + std::to_string(centroid_num[i]) + " search_space: " + std::to_string(centroid_keep_space[i]);
        trecorder.record_time_usage(record_output, message);
        PrintMessage("Finished 1 layer searching");

        trecorder.reset();
        std::vector<uint8_t> base_vector_codes(10000 * code_size);
        std::vector<float> base_vectors(10000 * dimension);
        const float * query = query_set.data();
        std::vector<float> precomputed_table(PQ.M * PQ.ksub);
        PQ.compute_inner_prod_table(query, precomputed_table.data());
        for (size_t repeat = 0; repeat < repeat_time; repeat++){
#pragma omp parallel for
            for (size_t j = 0; j < 10000; j++){
                pq_L2sqr(base_vector_codes.data() + j * code_size, precomputed_table.data(), code_size, PQ.ksub);
            }
        }
        message = "Finish computing PQ distance for 10000 vectors";
        trecorder.record_time_usage(record_output, message);

        PrintMessage("Finished compute PQ distance");

        trecorder.reset();
        query = query_set.data();
        for (size_t repeat = 0; repeat < repeat_time; repeat++){
#pragma omp parallel for
            for (size_t j = 0; j < 10000; j++){
                std::vector<float> residual(dimension);
                const float * base_vector = base_set.data() + j * dimension;
                faiss::fvec_madd(dimension, query, -1.0, base_vector, residual.data());
                faiss::fvec_norm_L2sqr(residual.data(), dimension);
            }
        }
        message = "Finished computing actual distance for 10000 vectors";
        trecorder.record_time_usage(record_output, message);

        PrintMessage("Finished compute L2 distance");
    
    }

    for (size_t i = 0; i < 4; i++){
        faiss::Clustering clus1(dimension, centroid_num1[i]);
        clus1.verbose = true;
        faiss::IndexFlatL2 index1(dimension);
        clus1.train(train_set_size, train_set.data(), index1);

        std::vector<idx_t> train_set_ids(train_set_size);
        std::vector<float> train_set_dists(train_set_size);
        index1.search(train_set_size, train_set.data(), 1, train_set_dists.data(), train_set_ids.data());

        std::vector<std::vector<float>> train_set_second(centroid_num1[i]);
        for (size_t j = 0; j < train_set_size; j++){
            idx_t second_layer_id = train_set_ids[j];
            for (size_t k = 0; k < dimension; k++){
                train_set_second[second_layer_id].push_back(train_set[j * dimension + k]);
            }
        }

        std::vector<faiss::Clustering *> clus2(centroid_num1[i]);
        std::vector<faiss::IndexFlatL2> index2(centroid_num1[i]);

#pragma omp parallel for
        for (size_t j = 0; j < centroid_num1[i]; j++){
            size_t sub_train_set_size = train_set_second[j].size() / dimension;
            clus2[j] = new faiss::Clustering(dimension, centroid_num2[i]);
            clus2[j]->verbose = false;
            index2[j] = faiss::IndexFlatL2(dimension);
            clus2[j]->train(sub_train_set_size, train_set_second[j].data(), index2[j]);
        }

        trecorder.reset();
        std::vector<idx_t> base_assigned_id_1st(base_set_size * centroid_keep_space1[i]);
        std::vector<float> base_assigned_dist_1st(base_set_size * centroid_keep_space1[i]);

        index1.search(base_set_size, base_set.data(), centroid_keep_space1[i], base_assigned_dist_1st.data(), base_assigned_id_1st.data());
        std::vector<idx_t> base_assigned_ids_2nd(base_set_size * centroid_keep_space1[i] * centroid_keep_space2[i]);
        std::vector<float> base_assigned_dists_2nd(base_set_size * centroid_keep_space1[i] * centroid_keep_space2[i]);
#pragma omp parallel for
        for (size_t j = 0; j < base_set_size; j++){
            const float * base_vector = base_set.data() + j * dimension;
            for (size_t k = 0; k < centroid_keep_space1[i]; k++){
                const size_t search_id = base_assigned_id_1st[j * centroid_keep_space1[i] + k];
                index2[search_id].search(1, base_vector, centroid_keep_space2[i], base_assigned_dists_2nd.data() + j * centroid_keep_space1[i] * centroid_keep_space2[i] + k * centroid_keep_space2[i], base_assigned_ids_2nd.data() + j * centroid_keep_space1[i] * centroid_keep_space2[i] + k * centroid_keep_space2[i]);
            }
        }
        std::string message = "Finished searching 2 layer:  ncentroids: " + std::to_string(centroid_num1[i]) + " " + std::to_string(centroid_num2[i]) + " search_space: " + std::to_string(centroid_keep_space1[i]) + " " + std::to_string(centroid_keep_space2[i]);
        trecorder.record_time_usage(record_output, message);
        PrintMessage("Finished compute 2 layer distance");
    }
}



    

