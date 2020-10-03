#include "../VQ_quantizer.h"
#include "../LQ_quantizer.h"
#include "../PQ_quantizer.h"
#include "../parameters/parameters_millions_PQ.h"
#include "../utils/utils.h"

using namespace bslib;

int main(){
    
    const std::string path_learn =     "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/sift_learn.fvecs";
    const std::string path_base =      "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/sift_base.fvecs";
    //const std::string path_record_VQ =  "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/recording/VQ_VQ.txt";
    const std::string path_record_LQ =  "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/recording/VQ_LQ_no_fast.txt";
    //const std::string path_record_PQ =  "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/recording/VQ_PQ.txt";

    std::ofstream VQ_record;
    std::ofstream LQ_record;
    std::ofstream PQ_record;

    //VQ_record.open(path_record_VQ, std::ios::binary);
    LQ_record.open(path_record_LQ, std::ios::binary);
    //PQ_record.open(path_record_PQ, std::ios::binary);
    nb = 100000;
    size_t search_space = 2000;
    size_t dimension = 128;
    std::vector<float> base_set(dimension * nb);
    std::vector<float> train_set(dimension * train_size);
    
    std::ifstream learn_file(path_learn, std::ios::binary);
    std::ifstream base_file(path_base, std::ios::binary);

    readXvec<float>(learn_file, train_set.data(), dimension, train_size);
    readXvec<float> (base_file, base_set.data(), dimension, nb);
    time_recorder Trecorder;

    for (size_t nc_1st = 100; nc_1st < 500; nc_1st+= 100){
    std::vector<idx_t> train_ids(train_size, 0);
    std::vector<float> first_layer_centroid(dimension * nc_1st);
    VQ_quantizer vq_quantizer(dimension,1, nc_1st);
    vq_quantizer.build_centroids(train_set.data(), train_size, train_ids.data());
    vq_quantizer.search_all(train_size, 1, train_set.data(), train_ids.data());

    for (size_t nc_2nd = 20; nc_2nd < 80; nc_2nd+= 20){
    VQ_record << "VQ structure: " << nc_1st << " " << nc_2nd << std::endl;
    LQ_record << "LQ structure: " << nc_1st << " " << nc_2nd << std::endl;
    std::cout << "Structure: " << nc_1st << " " << nc_2nd;



    std::vector<float> upper_centroids(dimension * nc_1st);
    std::vector<float> nn_centroids_dists(nc_1st * nc_2nd);
    std::vector<idx_t> nn_centroids_idxs(nc_1st * nc_2nd);
    
    vq_quantizer.compute_nn_centroids(nc_2nd, upper_centroids.data(), nn_centroids_dists.data(), nn_centroids_idxs.data());
    LQ_quantizer lq_quantizer(dimension, nc_1st, nc_2nd, upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data());
    lq_quantizer.build_centroids(train_set.data(), train_size, train_ids.data());

    VQ_quantizer vq_quantizer_2nd(dimension, nc_1st, nc_2nd);
    vq_quantizer_2nd.build_centroids(train_set.data(), train_size, train_ids.data());

    /**
     * To be tested:
     * Layer type
     * Num of queries
     * upper space
     * num centroids
     * keep space
     * dimension
     * 
     * Search time: 
     * vq layer:
     * Layer: 
     * a * x + b, x is the number of query, (also need to consider use HNSW or L2)
     * a = f(upper_keep_space, nc_per_group, dimension) = upper_keep_space * nc_per_group * alpha + belta
     * alpha = dimension * gamma
     * b = f(upper_keep_space, keep_space)
     * 
     * vq selection:
     * Selection:
     * m = upper_keep_space * nc_per_group
     * n = upper_keep_space * keep_space
     * c * m * n + d * m + e * n + f
     * 
     * lq_layer: a1 * x1 + a2 * x2 + b
     * The lq_layer should provide 
    **/

    //Time for LQ layer:

    for (size_t first_keep_space = 10; first_keep_space <= 30; first_keep_space += 10){
        for (size_t second_keep_space = 2; second_keep_space <= 8; second_keep_space += 2){


    for (size_t nb = 100; nb <= search_space; nb += 100){
        std::vector<float> time_saver(4, 0);

        for (size_t i = 0; i < nb; i++){
            Trecorder.reset();
            std::vector<idx_t> query_id(1, 0);
            std::vector<float> query_dist(1, 0);
            std::vector<float> result_dist(nc_1st);
            std::vector<idx_t> result_labels(nc_1st);
            vq_quantizer.search_in_group(1, base_set.data()+i * dimension, query_id.data(), result_dist.data(), result_labels.data(), first_keep_space);

            time_saver[0] += Trecorder.get_time_usage(); Trecorder.reset();

            query_id.resize(first_keep_space);
            query_dist.resize(first_keep_space);
            keep_k_min(nc_1st, first_keep_space, result_dist.data(), result_labels.data(), query_dist.data(), query_id.data());
            time_saver[1] += Trecorder.get_time_usage(); Trecorder.reset();

            std::vector<idx_t> upper_result_labels(nc_1st);
            std::vector<float> upper_result_dists(nc_1st);
            memcpy(upper_result_labels.data(), result_labels.data(), nc_1st * sizeof(idx_t));
            memcpy(upper_result_dists.data(), result_dist.data(), nc_1st * sizeof(float));
            Trecorder.reset();

            result_dist.resize(first_keep_space * nc_2nd);
            result_labels.resize(first_keep_space * nc_2nd);
            for (size_t j = 0; j < first_keep_space; j++){
                lq_quantizer.search_in_group(1, base_set.data() + i * dimension, upper_result_labels.data(), upper_result_dists.data(), nc_1st, query_id.data() + j,   result_dist.data() + j * nc_2nd, result_labels.data() + j * nc_2nd);
            }
            time_saver[2] += Trecorder.get_time_usage(); Trecorder.reset();

            query_id.resize(first_keep_space * second_keep_space);
            query_dist.resize(first_keep_space * second_keep_space);
            keep_k_min(first_keep_space * nc_2nd, first_keep_space * second_keep_space, result_dist.data(), result_labels.data(), query_dist.data(), query_id.data());
        
            time_saver[3] += Trecorder.get_time_usage(); Trecorder.reset();
        }
        float sum = time_saver[0] + time_saver[1] + time_saver[2] + time_saver[3];
        std::cout << "Time for " << nb << " LQ queries with search para: " << first_keep_space << " " << second_keep_space << " " << time_saver[0] << " " <<
         time_saver[1] << " " << time_saver[2] << " " << time_saver[3] << " sum: " << sum << std::endl;
        LQ_record << "Time for " << nb << " LQ queries with search para: " << first_keep_space << " " << second_keep_space << " " << time_saver[0] << " " <<
         time_saver[1] << " " << time_saver[2] << " " << time_saver[3] << " sum: " << sum << std::endl;
    }

    std::cout << "\n\n";

    /*
    for (size_t nb = 100; nb <= search_space; nb += 100){
        
        std::vector<float> time_saver(4, 0);
        for (size_t i = 0; i < nb; i++){
            Trecorder.reset();
            std::vector<idx_t> query_id(1, 0);
            std::vector<float> query_dist(1, 0);
            std::vector<float> result_dist(nc_1st);
            std::vector<idx_t> result_labels(nc_1st);
            vq_quantizer.search_in_group(1, base_set.data()+i * dimension, query_id.data(), result_dist.data(), result_labels.data(), first_keep_space);

            time_saver[0] += Trecorder.get_time_usage(); Trecorder.reset();

            query_id.resize(first_keep_space);
            query_dist.resize(first_keep_space);
            keep_k_min(nc_1st, first_keep_space, result_dist.data(), result_labels.data(), query_dist.data(), query_id.data());
            time_saver[1] += Trecorder.get_time_usage(); Trecorder.reset();
            
            result_dist.resize(first_keep_space * nc_2nd);
            result_labels.resize(first_keep_space * nc_2nd);

            for (size_t j = 0; j < first_keep_space; j++){
                vq_quantizer_2nd.search_in_group(1, base_set.data() + i * dimension,  query_id.data() + j, result_dist.data() + j * nc_2nd, result_labels.data() + j * nc_2nd, second_keep_space);
            }
            time_saver[2] += Trecorder.get_time_usage(); Trecorder.reset();

            query_id.resize(first_keep_space * second_keep_space);
            query_dist.resize(first_keep_space * second_keep_space);
            keep_k_min(first_keep_space * nc_2nd, first_keep_space * second_keep_space, result_dist.data(), result_labels.data(), query_dist.data(), query_id.data());
            
            time_saver[3] += Trecorder.get_time_usage(); Trecorder.reset();
        }
        float sum = time_saver[0] + time_saver[1] + time_saver[2] + time_saver[3];
        std::cout << "Time for " << nb << " VQ queries with search para: " << first_keep_space << " " << second_keep_space << " " << time_saver[0] << " " <<
         time_saver[1] << " " << time_saver[2] << " " << time_saver[3] << " sum: " << sum << std::endl;   
        VQ_record << "Time for " << nb << " VQ queries with search para: " << first_keep_space << " " << second_keep_space << " " << time_saver[0] << " " <<
         time_saver[1] << " " << time_saver[2] << " " << time_saver[3] << " sum: " << sum << std::endl;
    }

    std::cout << "\n\n";
    */

    }
    }
    }

    /* The number of centroids for PQ is to large to be computed in 2 layer 
    for (size_t M = 2; M <= 4; M += 2){
        for (size_t nbits = 4; nbits <= 8; nbits += 2){
            PQ_quantizer pq (dimension, nc_1st, M, nbits);
            pq.build_centroids(train_set.data(), train_size, train_ids.data());
            PQ_record << "PQ structure: " << nc_1st << " " << M <<" " << nbits << std::endl;
            
            for (size_t first_keep_space = 10; first_keep_space <= 50; first_keep_space += 10){
                for (size_t pq_keep_space = 10; pq_keep_space <= 100; pq_keep_space+=10){
                std::vector<float> time_saver(3, 0);
                for (size_t i = 0; i < nb; i++){
                    Trecorder.reset();
                    std::vector<idx_t> query_id(1, 0);
                    std::vector<float> query_dist(1, 0);
                    std::vector<float> result_dist(nc_1st);
                    std::vector<idx_t> result_labels(nc_1st);
                    vq_quantizer.search_in_group(1, base_set.data()+i * dimension, query_id.data(), result_dist.data(), result_labels.data(), first_keep_space);

                    time_saver[0] += Trecorder.get_time_usage(); Trecorder.reset();

                    query_id.resize(first_keep_space);
                    query_dist.resize(first_keep_space);
                    keep_k_min(nc_1st, first_keep_space, result_dist.data(), result_labels.data(), query_dist.data(), query_id.data());
                    time_saver[1] += Trecorder.get_time_usage(); Trecorder.reset();

                    result_dist.resize(first_keep_space * pq_keep_space);
                    result_labels.resize(first_keep_space * pq_keep_space);

                    for (size_t j = 0; j < first_keep_space; j++){
                        pq.search_in_group(1, base_set.data()+i * dimension, query_id.data() + j, result_dist.data() + j * pq_keep_space, result_labels.data() + j * pq_keep_space, pq_keep_space);
                    }
                    time_saver[2] += Trecorder.get_time_usage(); Trecorder.reset();  
                    std::cout << i << std::endl;
            }
            float sum = time_saver[0] + time_saver[1] + time_saver[2];
            std::cout << "Time for " << nb << " PQ queries with search para: " << first_keep_space << " " << pq_keep_space << " " << time_saver[0] << " " <<
            time_saver[1] << " " << time_saver[2] << " sum: " << sum << std::endl;
            PQ_record << "Time for " << nb << " PQ queries with search para: " << first_keep_space << " " << pq_keep_space << " " << time_saver[0] << " " <<
            time_saver[1] << " " << time_saver[2] << " sum: " << sum << std::endl;

        }
        std::cout << "\n\n";
    }
}
}
*/
}
}


