#include "../HNSWlib/hnswalg.h"
#include <faiss/IndexFlat.h>

/**
 * 
 * This is for testing the time and speed for HNSW lib and flatL2 search
 **/

typedef faiss::Index::idx_t idx_t;
int main(){
    size_t dimension = 128;
    size_t M_HNSW = 100;
    size_t efConstruction = 200;
    size_t nb = 1000;
    size_t nq = 100;
    size_t k_result = 10;
    size_t efSearch = 50;
    std::vector<float> xb(dimension * nb);
    std::vector<float> xq(dimension * nq);

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < dimension; j++) xb[dimension * i + j] = drand48();
        xb[dimension * i] += i / 1000.;
    }
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < dimension; j++) xq[dimension * i + j] = drand48();
        xq[dimension * i] += i / 1000.;
    }


    faiss::IndexFlatL2 index_flat(dimension);
    index_flat.add(nb, xb.data());

    std::vector<idx_t> flat_ids(nq * k_result);
    std::vector<float> flat_dists(nq * k_result);
    index_flat.search(nq, xq.data(), k_result, flat_dists.data(), flat_ids.data());

    std::vector<idx_t> HNSW_ids(nq * k_result);
    std::vector<float> HNSW_dists(nq * k_result);
    hnswlib::HierarchicalNSW * quantizer = new hnswlib::HierarchicalNSW(dimension, nb, M_HNSW, 2 * M_HNSW, efConstruction);
    for (size_t i = 0; i < nb; i++){
        quantizer->addPoint(xb.data() + i * dimension);
    }

#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        const float * query = xq.data() + i * dimension;
        auto result_queue = quantizer->searchBaseLayer(query, efSearch);
        for (size_t j = 0; j < k_result; j++){
            HNSW_dists[i * k_result + j] = result_queue.top().first;
            HNSW_ids[i * k_result + j] = result_queue.top().second;
            result_queue.pop();
        }
    }

    size_t sum_correctness = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt_set;
        for (size_t j = 0; j < k_result; j++){
            gt_set.insert(flat_ids[i * k_result + j]);
        }
        
        for (size_t j = 0; j < k_result; j++){
            if (gt_set.count(HNSW_ids[i * k_result + j]) != 0){
                sum_correctness ++;
            }
        }
    }
    std::cout << "The avg correctness is: " << (float) sum_correctness / (nq * k_result) << std::endl;

}



