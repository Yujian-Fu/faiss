#ifndef BS_LIB_VQ_VQ_H
#define BS_LIB_VQ_VQ_H

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

#include "index_VQ_VQ_utils.h"

/*
// The construction phase can be divided into 4 parts:
// 1. Train the Product Quantizer to support the compression.
// 2. Assign the base vectors into specific clusters and store the idx.
// 3. Build the index and store the base vector codes, vector norms, centroid norms
// 4. Do the search
*/
namespace bslib_VQ_VQ{
    struct BS_LIB_VQ_VQ
    {
        typedef faiss::Index::idx_t idx_t;
        typedef uint32_t bslib_idx_t;

        //Part 1: Product Quantizer
        size_t code_size;
        size_t dimension;
        faiss::ProductQuantizer * pq;

        //Part 2:Assigning base vectors
        faiss::IndexFlatL2 * quantizer;
        std::vector<faiss::IndexFlatL2 *> quantizers;
        size_t nc1;
        size_t nc2;

        //Part 3:Indexing with all base vectors
        std::vector<std::vector<std::vector<idx_t>>> ids;
        std::vector<std::vector<std::vector<uint8_t>>> base_codes;
        std::vector<std::vector<std::vector<float>>> base_norms;
        std::vector<std::vector<float>> centroid_norms;

        //Part 4:Searching
        size_t nprobe1;
        size_t nprobe2;
        size_t max_vectors;
        std::vector<float> precomputed_table;


        public:
            explicit BS_LIB_VQ_VQ(size_t dimension, size_t nc1, size_t nc2, size_t bytes_per_code, size_t nbits_per_idx);
            
            //Part 1:
            virtual void train_pq(size_t n, const float * x);
            virtual void build_quantizer(const char * centroid_path, const char * subcentroid_path);
            virtual void compute_residuals(size_t n, const float * x, float * residuals, const idx_t * labels, const idx_t * sub_labels);

            //Part 2: 
            virtual void assign(size_t n, const float * x, idx_t * labels, idx_t * sub_labels);

            //Part 3:
            virtual void reconstruct(size_t n, float * x, const float * decoded_residuals, const idx_t * idxs, const idx_t * sub_idxs);
            virtual void compute_centroid_norms();
            virtual void read(const char * path_index);
            virtual void write(const char * path_index);
            virtual void add_batch(size_t n, const float * x, const idx_t * origin_ids, const idx_t * idxs, const idx_t * sub_idxs);

            //Part 4:
            virtual float pq_L2sqr(const uint8_t * code);
            virtual void search(size_t nq, size_t k, const float * x, float * dists, idx_t * labels);     
    };
}
#endif