#include <cstdio>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/utils/utils.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#include "whole.h"

typedef uint8_t data_t;
int main(int argc, char** argv) {
  // Reserves 18% of GPU memory for temporary work by default; the
  // size can be adjusted, or your own implementation of GpuResources
  // can be made to manage memory in a different way.
    const char * LearnPath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    const char * CentroidsSavePath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/centroids_VQ.fvecs";
    
    std::ifstream LearnSet;
    LearnSet.open(LearnPath, std::ios::binary);
    if(!LearnSet)
        ReportFileError("Learn set open error");

    size_t Dimension = 128;
    size_t LearnNum;
    LearnSet.read((char *) &Dimension, sizeof(uint32_t));
    std::cout << "The dimension of this dataset is " << Dimension << std::endl;
    LearnSet.seekg(0, std::ios::end);
    size_t fsize = (size_t) LearnSet.tellg();
    std::cout << "The learn file size is " << fsize << std::endl;
    LearnNum = (unsigned) (fsize / (Dimension + sizeof(uint32_t)/sizeof(data_t)) / sizeof(data_t));
    std::cout << "The learn set size is " << LearnNum << std::endl;
    LearnSet.seekg(0, std::ios::beg);
    std::vector<float> LearnVectors(Dimension * LearnNum);
    std::cout << "Loading Learn Set " << std::endl;
    readXvecFvec<data_t>(LearnSet, LearnVectors.data(), Dimension, LearnNum, true);
    std::cout << "Loaded Learn Set " << std::endl;
    LearnSet.close();


    faiss::gpu::StandardGpuResources res;

    int numberOfEMIterations = 50;
    size_t numberOfClusters = 1000000;

    // generate a bunch of random vectors; note that this is on the CPU!
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = 0;            // this is the default
    config.useFloat16 = false;    // this is the default
    faiss::gpu::GpuIndexFlatL2 index(&res, Dimension, config);

    faiss::ClusteringParameters cp;
    cp.niter = numberOfEMIterations;
    cp.verbose = true; // print out per-iteration stats

    // For spherical k-means, use GpuIndexFlatIP and set cp.spherical = true

    // By default faiss only samples 256 vectors per centroid, in case
    // you are asking for too few centroids for too many vectors.
    // e.g., numberOfClusters = 1000, numVecsToCluster = 1000000 would
    // only sample 256000 vectors.
    //
    // You can override this to use any number of clusters
    // cp.max_points_per_centroid =
    //   ((numVecsToCluster + numberOfClusters - 1) / numberOfClusters);

    faiss::Clustering kMeans(Dimension, numberOfClusters, cp);

    // do the work!
    kMeans.train(LearnNum, LearnVectors.data(), index);

    // kMeans.centroids contains the resulting cluster centroids (on CPU)
    std::ofstream CentroidsOutput;
    CentroidsOutput.open(CentroidsSavePath, std::ios::binary);
    writeXvec<float>(CentroidsOutput, kMeans.centroids.data(), Dimension, LearnNum);

    printf("centroid 3 dim 6 is %f\n", kMeans.centroids[3 * Dimension + 6]);
    return 0;
}