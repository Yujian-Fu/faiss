#ifndef BS_LIB_VQ_LQ_H
#define BS_LIB_VQ_LQ_H

#include <iostream>
#include <fstream>
#include <cstdio>


#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>

#include "index_VQ_LQ_utils.h"


namespace bslib_VQ_LQ{
    struct BS_LIB_VQ_LQ{
    typedef faiss::Index::idx_t idx_t;

    size_t dimension;
    size_t nc;
    size_t code_size;
    bool use_quantized_distance;

    size_t nsubc;         ///< Number of sub-centroids per group
    bool do_pruning;      ///< Turn on/off pruning

    faiss::ProductQuantizer * pq;
    faiss::ProductQuantizer * norm_pq;
    faiss::IndexFlatL2 * quantizer;

    size_t nprobe;
    size_t max_vectors;


    std::vector<std::vector<idx_t> > nn_centroid_idxs;    ///< Indices of the <nsubc> nearest centroids for each centroid
    std::vector<std::vector<idx_t> > subgroup_sizes;      ///< Sizes of sub-groups for each group
    std::vector<float> alphas;    ///< Coefficients that determine the location of sub-centroids


    std::vector<std::vector<idx_t>> ids;
    //This is the inverted list for indexes
    std::vector<std::vector<uint8_t>> base_codes;
    //This is the PQ codes of residuals
    std::vector<std::vector<uint8_t>> base_norm_codes;
    //This is the PQ codes of norms of reconstructed base vectors
    std::vector<std::vector<float>> base_norms;
    //This is the norms of reconstructed base vectors

    protected:
        std::vector<float> norms;
        std::vector<float> centroid_norms;
        std::vector<float> precomputed_table;
        float pa_L2sqr(const uint8_t * code);
        /// Distances to the coarse centroids. Used for distance computation between a query and base points
        std::vector<float> query_centroid_dists;

        /// Distances between coarse centroids and their sub-centroids
        std::vector<std::vector<float>> inter_centroid_dists;

    public:
        explicit BS_LIB_VQ_LQ(size_t dimension, size_t ncentroids, size_t bytes_per_code, size_t nbits_per_idx, size_t nsubcentroids, bool use_quantized_distance);

        virtual void build_quantizer(const char * centroid_path);
        virtual void assign(size_t n, const float * x, idx_t * label, size_t k = 1);
        virtual void train_pq(size_t n, const float * x, bool train_pq, bool train_norm_pq);
        virtual void reconstruct(size_t n, float * x, const float * decoded_residuals, const float * subcentroids, const idx_t * keys);
        virtual void add_group(size_t centroid_idx, size_t group_size, const float * data, const idx_t * idxs);
        virtual void search(size_t nq, size_t k, const float * x, float * distances, idx_t * labels);
        virtual float pq_L2sqr(const uint8_t *code);
        virtual void write(const char * path_index);
        virtual void read(const char * path_index);
        virtual float compute_alpha(const float * centroid_vectors, const float * points, const float * centroid, const float * centroid_vector_norms_L2sqr, size_t group_size);
        virtual void compute_subcentroid_idxs(idx_t * subcentroid_idxs, const float * subcentroids, const float * x, size_t group_size);
        virtual void compute_residuals(size_t n, const float * x, float * residuals, const float * subcentroids, const idx_t * keys);
        virtual void compute_centroid_norms();
        virtual void compute_inter_centroid_dists();
    };
}

#endif