#include <random>
#include <time.h>
#include <iostream>
#include<cstdlib>
#include <algorithm>
#include "faiss/utils/utils.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/utils/distances.h"
#include "faiss/Clustering.h"
#include "faiss/impl/ProductQuantizer.h"

typedef faiss::Index::idx_t idx_t;

//const size_t nc = 12;
const size_t nc_low = 100;
const size_t nc_up = 3000;
const size_t nc_step = 50;
const size_t M = 16;
const size_t nc_PQ = 256;
const float alpha = 0.1;
const size_t N_random = 999;
const bool use_kmeansplusplus = false;

const size_t index_iter = 1;
const size_t PQ_iter = 1;
const size_t total_iter = 30;
std::vector<size_t> recall_k_list = {1, 10, 100};


void kmeansplusplus(const float * vectors, float * centroids, size_t nb, size_t dimension, size_t nc){
    srand((unsigned)time(NULL));
    size_t start_index = rand() % (nb);
    for (size_t i = 0; i < dimension; i++){
        centroids[i] = vectors[start_index * dimension + i];
    }

    std::vector<float> distances(nb);
    std::vector<idx_t> labels(nb);
    for (size_t i = 1; i < nc; i++){
        faiss::IndexFlatL2 centroid_index(dimension);
        centroid_index.add(i, centroids);
        float total_dist = 0;
        centroid_index.search(nb, vectors, 1, distances.data(), labels.data());
        for (size_t j = 0; j < nb; j++){
            total_dist += distances[j];
        }
        total_dist *= float(rand() % (N_random + 1)) / (float)(N_random + 1);
        for (size_t j = 0; j < nb; j++){
            total_dist -= distances[j];
            if (total_dist < 0){
                for (size_t m = 0; m < dimension; m++){
                    centroids[i * dimension + m] = vectors[j * dimension + m];
                }
                break;
            }
        }
    }
}


void initialize_centroid(const float * vectors, float * centroids, size_t nb, size_t dimension, size_t nc){
    srand((unsigned)time(NULL));
    if (use_kmeansplusplus){
        kmeansplusplus(vectors, centroids, nb, dimension, nc);
    }
    else{
        std::vector<int> nc_index(nb);
    
        for(size_t i = 0; i < nb; i++){
            nc_index[i] = i;
        }
        std::random_shuffle(nc_index.begin(), nc_index.end());
        for (size_t i = 0; i < nc; i++){
            for (size_t j = 0; j < dimension; j++){
                centroids[i * dimension + j] = vectors[nc_index[i] * dimension + j];
            }
        }
    }
}

//ids: nb
void assign_vector(const float * centroid, const float * vectors, size_t dimension, idx_t * ids, size_t nc, size_t nb, size_t k = 1){
    faiss::IndexFlatL2 index(dimension);
    index.add(nc, centroid);
    std::vector<float> distance(k * nb);
    index.search(nb, vectors, k, distance.data(), ids);
}


void compute_index_residual(const float * base_vectors, const float * centroids, const idx_t * base_ids, float * residual, size_t nb, size_t dimension){
#pragma omp parallel for
    for (size_t i = 0; i < nb; i++){
        faiss::fvec_madd(dimension, base_vectors + i * dimension, -1.0, centroids + base_ids[i] * dimension, residual+ i * dimension);
    }
}

// PQ_centroids: nc_PQ * dimension_sub * M
void initialize_PQ_centroid(const float * residual_vectors, float * PQ_centroids, size_t nb, size_t dimension, size_t nc_PQ){
    srand((unsigned)time(NULL));
    size_t dimension_sub = dimension / M;


    if (use_kmeansplusplus){
#pragma omp parallel for
        for (size_t PQ_index = 0; PQ_index < M; PQ_index++){
            std::vector<float> subset(dimension_sub * nb);
            
            for (size_t i = 0; i < nb; i++){
                for (size_t j = 0; j < dimension_sub; j++){
                    subset[i * dimension_sub + j] = residual_vectors[i * dimension + PQ_index * dimension_sub + j];
                }
            }
            kmeansplusplus(subset.data(), PQ_centroids + PQ_index * nc_PQ * dimension_sub, nb, dimension_sub, nc_PQ);
        }
    }
    else {
#pragma omp parallel for
        for (size_t PQ_index = 0; PQ_index < M; PQ_index++){
            std::vector<int> nc_index(nb);
            for(size_t i = 0; i < nb; i++){
                nc_index[i] = i;
            }
            std::random_shuffle(nc_index.begin(), nc_index.end());

            for (size_t i = 0; i < nc_PQ; i++){
                for (size_t j = 0; j < dimension_sub; j++){
                    PQ_centroids[PQ_index * nc_PQ * dimension_sub + i * dimension_sub + j] = residual_vectors[nc_index[i] * dimension + PQ_index * dimension_sub + j];
                }
            }
        }
    }
}

void residual_PQ_dist(const float * residuals, const float * centroid, const idx_t * PQ_ids, size_t dimension){
    size_t dimension_sub = dimension / M;
    float dist = 0;
    for (size_t i = 0; i < nb; i++){
        for (size_t j = 0; j < M; j++){
            dist += faiss::fvec_L2sqr(residuals + i * dimension + j * dimension_sub, centroid + j * nc_PQ * dimension_sub + PQ_ids[j * nb + i] * dimension_sub, dimension_sub);
        }
    }
    std::cout << "Dist between residual and PQ centroids: " << dist << std::endl;
}



void compute_PQ_residual(const float * base_vectors, const float * PQ_centroids, const idx_t * PQ_ids, float * PQ_residual, size_t nb, size_t dimension, size_t nc_PQ){
    size_t dimension_sub = dimension / M;
#pragma omp parallel for
    for (size_t i = 0; i < nb; i++){
        for (size_t PQ_index = 0; PQ_index < M; PQ_index++){
            idx_t PQ_id = PQ_ids[PQ_index * nb + i];
            faiss::fvec_madd(dimension_sub, base_vectors + i * dimension + PQ_index * dimension_sub, -1.0, PQ_centroids + PQ_index * nc_PQ * dimension_sub + PQ_id * dimension_sub, PQ_residual + i * dimension + PQ_index * dimension_sub);
        }
    }
}


void update_index_centroids(const float * base_vector, const float * PQ_residual, float * centroids, const idx_t * base_ids, size_t nb, size_t nc, size_t dimension){
    std::vector<size_t> cluster_size(nc, 0);
    std::vector<float> inter_centroids(nc * dimension, 0);

    for (size_t i = 0; i < nb; i++){
        idx_t base_id = base_ids[i];
        cluster_size[base_id] ++;
        for (size_t j = 0; j < dimension; j++){
            inter_centroids[base_id * dimension + j] += alpha * PQ_residual[i * dimension + j] + (1 - alpha) * base_vector[i * dimension + j];
        }
    }

    for (size_t i = 0; i < nc; i++){
        if(cluster_size[i] > 0){
            for (size_t j = 0; j < dimension; j++){
                centroids[i * dimension + j] = inter_centroids[i * dimension + j] / cluster_size[i];
            }
        }
    }
}

void kmeans_update_centroids(const float * base_vector, float * centroids, const idx_t * base_ids, size_t nb, size_t nc, size_t dimension){
    std::vector<size_t> cluster_size(nc, 0);
    std::vector<float> inter_centroids(nc * dimension, 0);

    for (size_t i = 0; i < nb; i++){
        idx_t base_id = base_ids[i];
        cluster_size[base_id] ++;
        for (size_t j = 0; j < dimension; j++){
            inter_centroids[base_id * dimension + j] += base_vector[i * dimension + j];
        }
    }
    for (size_t i = 0; i < nc; i++){
        if (cluster_size[i] > 0){
            for (size_t j = 0; j < dimension; j++){
                centroids[i * dimension + j] = inter_centroids[i * dimension + j] / cluster_size[i];
            }
        }
    }
}


void update_PQ_centroids(const float * residual, float * PQ_centroid, idx_t * residual_ids, size_t nb, size_t nc_PQ, size_t dimension){
    size_t dimension_sub = dimension / M;
    //residual_PQ_dist(residual, PQ_centroid, residual_ids, dimension);
    for (size_t PQ_index = 0; PQ_index < M; PQ_index++){
        std::vector<size_t> cluster_size(nc_PQ, 0);
        std::vector<float> inter_centroids(nc_PQ * dimension_sub);
        for (size_t i = 0; i < nb; i++){
            idx_t sub_vector_id = residual_ids[PQ_index * nb + i];
            assert(sub_vector_id < nc_PQ);
            cluster_size[sub_vector_id] += 1;
            for (size_t j = 0; j < dimension_sub; j++){
                inter_centroids[sub_vector_id * dimension_sub + j] += residual[i * dimension + PQ_index * dimension_sub + j];
            }
        }

        for (size_t i = 0; i < nc_PQ; i++){
            if (cluster_size[i] > 0){
                for (size_t j = 0; j < dimension_sub; j++){
                    PQ_centroid[PQ_index * nc_PQ * dimension_sub + i * dimension_sub + j] = inter_centroids[i * dimension_sub + j] / cluster_size[i];
                }
            }
        }
    }
    //residual_PQ_dist(residual, PQ_centroid, residual_ids, dimension);
}


void metric_computation(const float * base_vector, const float * index_centroids, const idx_t * base_ids, const float * PQ_centroids, 
                        const idx_t * PQ_ids, size_t nb, size_t nc_PQ, size_t dimension, float * distance_result, bool compute_b_c = true, bool compute_b_PQ = true){
    // The distance to the index centroids
    if (compute_b_c){
        float b_c_distance = 0;
        for (size_t i = 0; i < nb; i++){
            idx_t base_id = base_ids[i];
            b_c_distance += faiss::fvec_L2sqr(base_vector + i * dimension, index_centroids + base_id * dimension, dimension);
        }
        std::cout << "Now the avergae b_c_dist is: " << b_c_distance / nb << std::endl;
        distance_result[0] = b_c_distance / nb;
    }

    if (compute_b_PQ){
        std::vector<float> index_residual(nb * dimension);
        compute_index_residual(base_vector, index_centroids, base_ids, index_residual.data(), nb, dimension);
        std::vector<float> index_PQ_residual(nb * dimension);
        compute_PQ_residual(index_residual.data(), PQ_centroids, PQ_ids, index_PQ_residual.data(), nb, dimension, nc_PQ);
        float compression_loss = faiss::fvec_norm_L2sqr(index_PQ_residual.data(), nb * dimension) / nb;
        std::cout << "Now the average compression loss is: " << compression_loss << std::endl;
        distance_result[1] = compression_loss;
    }
}


void get_base_vectors(const idx_t * base_ids, const idx_t * PQ_ids, const float * index_centroids, const float * PQ_centroids, float * compressed_vectors, size_t dimension, size_t nb, size_t nc_PQ){
    size_t dimension_sub = dimension / M;
#pragma omp parallel for
    for (size_t i = 0; i < nb; i++){
        idx_t base_id = base_ids[i];
        for (size_t j = 0; j < dimension; j++){
            compressed_vectors[i * dimension + j] += index_centroids[base_id * dimension + j];
        }

        for (size_t PQ_index = 0; PQ_index < M; PQ_index++){
            idx_t PQ_id = PQ_ids[PQ_index * nb + i];
            for (size_t j = 0; j < dimension_sub; j++){
                compressed_vectors[i * dimension + PQ_index * dimension_sub + j] += PQ_centroids[PQ_index * nc_PQ * dimension_sub + PQ_id * dimension_sub + j];
            }
        }
    }
}









