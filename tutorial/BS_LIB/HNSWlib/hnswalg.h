#pragma once

#include "visited_list_pool.h"
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>

#include <faiss/utils/Heap.h>
#include <faiss/Index.h>
#include "../utils/utils.h"

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif

#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

namespace hnswlib {
    //typedef faiss::Index::idx_t idx_t;
    typedef uint32_t idx_t;
    
    struct HierarchicalNSW
    {

        size_t maxelements_;
        size_t cur_element_count;
        size_t efConstruction_;

        VisitedListPool *visitedlistpool;

        idx_t enterpoint_node;

        size_t dist_calc;

        const bool PQ_dist_flag;
        const bool use_vector_alpha;

        size_t d_;
        size_t code_size;
        size_t ksub;

        size_t data_size_;
        size_t offset_data;
        size_t size_data_per_element;
        size_t M_;
        size_t maxM_;
        size_t size_links_level0;
        size_t efSearch;
        float centroid_norm;

        uint8_t * base_code_point;
        faiss::Index::idx_t * base_sequece_id_list;

        float q_c_dist;
        float q_alpha;

        char * data_level0_memory_;

        float * vector_alpha;
        float * base_norms;
        float nn_dist;

    public:
        HierarchicalNSW(const bool PQ_flag = false, const bool use_vector_alpha = false);

        HierarchicalNSW(size_t d, size_t maxelements, size_t M, size_t maxM, size_t efConstruction, bool PQ_flag = false,
        const bool use_vector_alpha = false, size_t code_size = 8, size_t ksub = 256);

        ~HierarchicalNSW();

        inline float *getDataByInternalId(idx_t internal_id) const {
            return (float *) (data_level0_memory_ + internal_id * size_data_per_element + offset_data);
        }

        inline uint8_t *get_linklist0(idx_t internal_id) const {
            return (uint8_t *) (data_level0_memory_ + internal_id * size_data_per_element);
        }

        inline uint8_t *getcodeByInternalId(idx_t internal_id) const {
            return (uint8_t *) (base_code_point + internal_id * code_size);
        }

        std::priority_queue<std::pair<float, idx_t>> searchBaseLayer(const float *x, size_t ef);
        std::vector<float> vector_norm; // Use for storing the vector norm (term 2)

        void getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN);

        float getDistance(const float * point, idx_t id);
        void mutuallyConnectNewElement(idx_t id, std::priority_queue<std::pair<float, idx_t>> topResults);

        void addPoint(const float *point);

        std::priority_queue<std::pair<float, idx_t >> searchKnn(const float *query_data, size_t k);

        void SaveInfo(const std::string &location);
        void SaveEdges(const std::string &location);

        void LoadInfo(const std::string &location);
        void LoadData(const std::string &location);
        void LoadEdges(const std::string &location);
        
        float fstdistfunc(const float *x, const float *y);
        
        float PQdistfunc(const float * PQ_dist_table, const idx_t id);
    };
}
