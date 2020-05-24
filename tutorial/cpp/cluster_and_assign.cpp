#include <fstream>
#include <iostream>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
#include <faiss/Clustering.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/random.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
typedef uint8_t data_t;

template<typename T>
void PrintVector(T *data, const size_t dimension, const size_t num_vector){
    std::cout << "Printing sample (2 vectors) of the dataset " << std::endl;
    for (size_t i= 0; i < 2 * dimension; i++)
    {
        std::cout << (float)data[i];
        if (!i)
        {
            std::cout << " ";
            continue;
        }

        if (i % (dimension - 1) != 0)
        {
            std::cout << " ";
        }
        else
        {
            std::cout << std::endl << std::endl;
        }
    }
    std::cout << std::endl;
}

/// Read fvec/ivec/bvec format vectors and convert them to the float array
template<typename T>
void readXvecFvec(std::ifstream & in, float *data, const size_t dimension, 
                  const size_t num_vector = 1, const size_t num_subvector = 1,
                  bool print_flag = false)
{
    long seed = 1234;
    std::vector<int> perm(num_vector);
    faiss::rand_perm(perm.data(), num_subvector, seed);

    std::cout << "Loading data with " << num_vector << " vectors in " << dimension << std::endl;
    uint32_t dim = dimension;
    T mass[dimension];
    size_t print_every = num_subvector / 10;
    for (size_t i = 0; i < num_subvector; i++) {
        in.seekg(0, (sizeof(uint32_t)+sizeof(T)*dim) *perm[i]);
        in.read((char *) &dim, sizeof(uint32_t));
        if (dim != dimension) {
            std::cout << dim << " " << dimension << " dimension error \n";
            exit(1);
        }
        in.read((char *) mass, dim * sizeof(T));
        for (size_t j = 0; j < dimension; j++){
            data[i * dim + j] = 1. * mass[j];
        }
        if ( i % (print_every) == 0)
            std::cout << "[Finished loading " << i << " / " << num_subvector << "]" << std::endl; 
    }
    if (print_flag)
        PrintVector<float>(data, dimension, num_vector);
    //PrintVector(data, dimension, num_vector);
}

int main(int argc, char** argv) {
  // Reserves 18% of GPU memory for temporary work by default; the
  // size can be adjusted, or your own implementation of GpuResources
  // can be made to manage memory in a different way.
  faiss::gpu::StandardGpuResources res;

  size_t LearnNum;
  int Dimension = 128;
  int numberOfEMIterations = 20;
  size_t numberOfClusters = 1500000;
  size_t numVecsToCluster = 10000000;
  const char * LearnPath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
  std::ifstream LearnSet;
  LearnSet.open(LearnPath, std::ios::binary);
  LearnSet.read((char *) &Dimension, sizeof(uint32_t));
  std::cout << "The dimension of this dataset is " << Dimension << std::endl;
  LearnSet.seekg(0, std::ios::end);
  size_t fsize = (size_t) LearnSet.tellg();
  std::cout << "The learn file size is " << fsize << std::endl;
  LearnNum = (unsigned) (fsize / (Dimension + sizeof(uint32_t)/sizeof(data_t)) / sizeof(data_t));
  std::cout << "The learn set size is " << LearnNum << std::endl;

  //LearnNum = LearnNum / 5;
  // generate a bunch of random vectors; note that this is on the CPU!
  LearnSet.seekg(0, std::ios::beg);
  std::vector<float> vecs(Dimension * numVecsToCluster);
  std::cout << "Loading Learn Set " << std::endl;
  readXvecFvec<data_t>(LearnSet, vecs.data(), Dimension, LearnNum, numVecsToCluster, true);
  std::cout << "Loaded Learn Set " << std::endl;
  LearnSet.close();


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
  kMeans.train(numVecsToCluster, vecs.data(), index);

  // kMeans.centroids contains the resulting cluster centroids (on CPU)
  // printf("centroid 3 dim 6 is %f\n", kMeans.centroids[3 * dim + 6]);

  return 0;
}
