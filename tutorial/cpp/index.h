#ifndef BS_LIB_H
#define BS_LIB_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_map>


#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>

#include "index_utils.h"

namespace bslib{
struct BS_LIB
{
    typedef faiss::Index::idx_t idx_t;

    size_t dimension;
    size_t nc;
    size_t code_size;
    bool use_quantized_distance;

    faiss::ProductQuantizer * pq;
    faiss::ProductQuantizer * norm_pq;
    faiss::IndexFlatL2 * quantizer;

    size_t nprobe;
    size_t max_codes;

    std::vector<std::vector<idx_t>> ids;
    //This is the inverted list for indexes
    std::vector<std::vector<uint8_t>> base_codes;
    //This is the PQ codes of residuals
    std::vector<std::vector<uint8_t>> base_norm_codes;
    //This is the PQ codes of norms of reconstructed base vectors
    std::vector<std::vector<float>> base_norms;

    protected:
        std::vector<float> norms;
        std::vector<float> centroid_norms;
        std::vector<float> precomputed_table;
        float pa_L2sqr(const uint8_t * code);

    public:
        explicit BS_LIB(size_t dimension, size_t ncentroids, size_t byte_per_code,
                        size_t nbits_per_idx, bool use_quantized_distance, size_t max_group_size = 65536);

        virtual ~BS_LIB();

        virtual void build_quantizer(const char * centroid_path);
        virtual void assign(size_t n, const float * x, idx_t * label, size_t k = 1);
        virtual void train_pq(size_t n, const float * x, bool train_pq, bool train_norm_pq);
        virtual void compute_residuals(size_t n, const float * x, float * residuals, const idx_t * keys);
        virtual void reconstruct(size_t n, float * x, const float * decoded_residuals, const idx_t * keys);
        virtual void add_batch(size_t n, const float * x, const idx_t * origin_ids, const idx_t * quantization_ids);
        virtual void search(size_t nq, size_t k, const float * x, float * distances, idx_t * labels);
        virtual float pq_L2sqr(const uint8_t *code);
        virtual void write(const char * path_index);
        virtual void read(const char * path_index);
        virtual void compute_centroid_norms();
};
}
#endif