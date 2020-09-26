#include "../HNSWlib/hnswalg.h"
#include <faiss/IndexFlat.h>
#include "../utils/utils.h"
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

/**
 * 
 * This is for testing the time and speed for HNSW lib and flatL2 search
 **/

typedef faiss::Index::idx_t idx_t;
using namespace bslib;

void keep_k_min(const size_t m, const size_t k, const float * all_dists, const idx_t * all_labels, float * sub_dists, idx_t * sub_labels){
    
    if (k < m){
        faiss::maxheap_heapify(k, sub_dists, sub_labels);
        for (size_t i = 0; i < m; i++){
            if (all_dists[i] < sub_dists[0]){
                faiss::maxheap_pop(k, sub_dists, sub_labels);
                faiss::maxheap_push(k, sub_dists, sub_labels, all_dists[i], all_labels[i]);
            }
        }
    }
    else if (k == m){
        memcpy(sub_dists, all_dists, k * sizeof(float));
        memcpy(sub_labels, all_labels, k * sizeof(idx_t));
    }
    else{
        std::cout << "k should be at least as large as m in keep_k_min function " << std::endl;
        exit(0);
    }
}


int main(){
    size_t dimension = 128;
    size_t M_HNSW = 100;
    size_t efConstruction = 200;
    size_t nb = 100;
    size_t nq = 1000;
    size_t k_result = 10;
    size_t efSearch = k_result;
    std::vector<float> xb(dimension * nb);
    std::vector<float> xq(dimension * nq);
    time_recorder Trecorder  = time_recorder();


    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < dimension; j++) xb[dimension * i + j] = drand48();
        xb[dimension * i] += i / 1000.;
    }
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < dimension; j++) xq[dimension * i + j] = drand48();
        xq[dimension * i] += i / 1000.;
    }


    Trecorder.reset();
    std::vector<idx_t> flat_ids(nq * k_result);
    std::vector<float> flat_dists(nq * k_result);
    std::vector<float> each_result_dist(nb);
    std::vector<idx_t> each_result_label(nb);

    for (size_t i = 0; i < nq; i++){
        float * query = xq.data() + i * dimension;
        for (size_t j = 0; j < nb; j++){
            float * target = xb.data() + j * dimension;
            std::vector<float> query_centroid_vector(dimension);
            each_result_dist[j] = faiss::fvec_L2sqr(target, query, dimension);
            each_result_label[j] = j;
        }
        keep_k_min(nb, k_result, each_result_dist.data(), each_result_label.data(), flat_dists.data() + i * k_result, flat_ids.data() + i * k_result);
    }
    Trecorder.print_time_usage("IndexFlat search finished: ");


    //faiss::IndexFlatL2 index_flat(dimension);
    //index_flat.add(nb, xb.data());
    //for (size_t i = 0; i < nq; i++){
    //    index_flat.search(1, xq.data() + i * dimension, k_result, flat_dists.data() + i * k_result, flat_ids.data() + i * k_result);
    //}
    

    std::vector<idx_t> HNSW_ids(nq * efSearch);
    std::vector<float> HNSW_dists(nq * efSearch);
    hnswlib::HierarchicalNSW * quantizer = new hnswlib::HierarchicalNSW(dimension, nb, M_HNSW, 2 * M_HNSW, efConstruction);
    for (size_t i = 0; i < nb; i++){
        quantizer->addPoint(xb.data() + i * dimension);
    }

    Trecorder.reset();
//#pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        const float * query = xq.data() + i * dimension;
        auto result_queue = quantizer->searchBaseLayer(query, efSearch);
        for (size_t j = 0; j < efSearch; j++){
            HNSW_dists[(i + 1) * efSearch - j - 1] = result_queue.top().first;
            HNSW_ids[(i + 1) * efSearch - j - 1] = result_queue.top().second;
            result_queue.pop();
        }
    }
    Trecorder.print_time_usage("HNSW search finished: ");

    size_t sum_correctness = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt_set;
        for (size_t j = 0; j < k_result; j++){
            gt_set.insert(flat_ids[i * k_result + j]);
        }
        
        for (size_t j = 0; j < k_result; j++){
            if (gt_set.count(HNSW_ids[i * efSearch + j]) != 0){
                sum_correctness ++;
            }
        }
    }
    std::cout << "The avg correctness is: " << (float) sum_correctness / (nq * k_result) << std::endl;

}



