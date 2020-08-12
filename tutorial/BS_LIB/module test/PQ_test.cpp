#include<faiss/IndexFlat.h>
#include<faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <unordered_set>
#include "../utils/utils.h"

typedef faiss::Index::idx_t idx_t;

using namespace bslib;
int main(){

    int dimension = 128;                            // dimension
    int nb = 100000;                       // database size
    int nq = 1000;                        // nb of queries
    float *xb = new float[dimension * nb];
    float *xq = new float[dimension * nq];
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < dimension; j++) xb[dimension * i + j] = drand48();
        xb[dimension * i] += i / 1000.;
    }
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < dimension; j++) xq[dimension * i + j] = drand48();
        xq[dimension * i] += i / 1000.;
    }

    size_t k = 10;
    time_recorder Trecorder = time_recorder();

    Trecorder.reset();
    faiss::IndexFlatL2 index_flat(dimension);
    std::vector<idx_t> labels(k * nq);
    std::vector<float> dists(k * nq);
    Trecorder.print_time_usage("Training flat index");
    

    index_flat.add(nb, xb);
    index_flat.search(nq, xq, k, dists.data(), labels.data());
    Trecorder.print_time_usage("Searching for flat index");

    size_t M_HNSW = 100;
    std::vector<idx_t> HNSW_labels(k * nq);
    std::vector<float> HNSW_dists(k * nq);
    faiss::IndexHNSWFlat index_HNSW(dimension, M_HNSW);
    
    index_HNSW.train(nb / 10, xb);
    index_HNSW.add(nb, xb);

    index_HNSW.search(nq, xq, k, HNSW_dists.data(), HNSW_labels.data());
    size_t sum_correctness = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt_set;
        for (size_t j = 0; j < k; j++){
            gt_set.insert(labels[i * k + j]);
        }
        for (size_t j = 0; j < k; j++){
            if (gt_set.count(HNSW_labels[i * k + j]) != 0){
                sum_correctness ++;
            }
        }
    }
    std::cout << "The recall for HNSW k = " << k << " is: " << float(sum_correctness) / (k * nq) << std::endl; 



    size_t nlist = 100;
    size_t M = 8;
    size_t nbits = 10;
    std::vector<idx_t> pq_labels(k * nq);
    std::vector<float> pq_dists(k * nq);
    faiss::IndexFlatL2 quantizer(dimension);
    faiss::IndexIVFPQ index_pq(&quantizer, dimension, nlist, M, nbits);
    index_pq.verbose = true;
    index_pq.train(nb / 10, xb);
    index_pq.add(nb, xb);
    Trecorder.print_time_usage("Training PQ index");


    index_pq.nprobe = 10;
    index_pq.search(nq, xq, k, pq_dists.data(), pq_labels.data());
    Trecorder.print_time_usage("Searching for PQ index");

    sum_correctness = 0;
    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt_set;
        for (size_t j = 0; j < k; j++){
            gt_set.insert(labels[i * k + j]);
        }
        for (size_t j = 0; j < k; j++){
            if (gt_set.count(pq_labels[i * k + j]) != 0){
                sum_correctness ++;
            }
        }
    }

    std::cout << "The recall for PQ k = " << k << " is: " << float(sum_correctness) / (k * nq) << std::endl; 


    faiss::ClusteringParameters CP; // Try different settings of CP
    CP.niter = 40;
    std::vector<float> centroids(dimension * nlist);
    faiss::Clustering clus(dimension, nlist);
    faiss::IndexFlatL2 index_assign(dimension);
    clus.train(nb / 10, xb, index_assign);
    
    faiss::ProductQuantizer PQ(dimension, M, nbits);
    





}