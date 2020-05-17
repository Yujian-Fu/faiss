#include <stdio.h>
#include <iostream>
#include <whole.h>
#include <faiss/Clustering.h>

typedef uint8_t data_t;

int main(){
    //******************
    // Parameters
    //******************
    //load base set
    size_t BaseNum;
    size_t Dimension;
    const char * BasePath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs";;
    float * BaseVectors;
    
    //Load Dataset

    //learn set parameter
    size_t LearnNum;
    const char * LearnPath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";;

    //read_fvecs(BasePath, BaseVectors, BaseNum, Dimension);
    std::ifstream LearnSet(LearnPath, std::ios::binary);

    //Get the size and dimension
    LearnSet.read((char *) &Dimension, sizeof(uint32_t));
    LearnSet.seekg(0, std::ios::end);
    size_t fsize = (size_t) LearnSet.tellg();
    LearnNum = (unsigned) (fsize / (Dimension + 4) / sizeof(data_t));

    //load learn set 
    std::cout << "Loading Learn Set " << std::endl;
    std::vector<float> LearnVectors(Dimension * LearnNum);
    readXvecFvec<data_t>(LearnSet, LearnVectors.data(), Dimension, LearnNum);
    std::cout << "Loaded Learn Set " << std::endl;
    LearnSet.close();

    //Vector Quantization Parameter
    size_t ncentroids = 1000000;
    std::vector<float> centroids (ncentroids * Dimension) ;
    float obj;
    const char * CentroidsSavePath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/vector_quantization_centroids.fvecs";

    //Generate kmeans centroids
    std::cout << "Training Centroids " << std::endl;
    obj = faiss::kmeans_clustering(Dimension, LearnNum, ncentroids, LearnVectors.data(), centroids.data());
    std::cout << "Trained Centroids " << std::endl;
    std::ofstream CentroidSave(CentroidsSavePath, std::ios::binary);
    std::cout << "Saving Centroids " << std::endl;
    writeXvec<float>(CentroidSave, centroids.data(), Dimension, ncentroids);
    std::cout << "Saved Centroids " << std::endl;
    CentroidSave.close();

    std::cout << "Loading Centroids " << std::endl;
    std::ifstream CentroidRead(CentroidsSavePath, std::ios::binary);
    readXvec<float>(CentroidRead, centroids.data(), ncentroids, Dimension, true);
    std::cout << "Loaded Centroids " << std::endl;

}