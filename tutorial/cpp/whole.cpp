#include <stdio.h>
#include <iostream>
#include <faiss/Clustering.h>

#include "whole.h"

// Parameters on server

typedef uint8_t data_t;
const char * LearnPath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
const char * CentroidsSavePath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/vector_quantization_centroids.fvecs";
const char * BasePath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_base.bvecs";
const char * ComputedVQIdsPath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/computed_vq_ids.fvecs";
size_t ncentroids = 1000000;


/*
//Parameters on laptop
typedef float data_t;
const char * LearnPath = "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/sift_learn.fvecs";
const char * CentroidsSavePath = "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/vector_quantization_centroids.fvecs";
const char * BasePath = "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/sift_base.fvecs";
const char * ComputedVQIdsPath = "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/computed_vq_ids.fvecs";
size_t ncentroids = 2000;
*/


int main(){

    //learn set parameter
    size_t LearnNum;
    size_t Dimension = 128;
    bool train_vector_quantization = true;
    bool assign_vertor_quantization = true;

    //read_fvecs(BasePath, BaseVectors, BaseNum, Dimension);
    std::ifstream LearnSet;
    LearnSet.open(LearnPath, std::ios::binary);
    if(!LearnSet)
        ReportFileError("Learn set open error");

    //Get the size and dimension
    LearnSet.read((char *) &Dimension, sizeof(uint32_t));
    std::cout << "The dimension of this dataset is " << Dimension << std::endl;
    LearnSet.seekg(0, std::ios::end);
    size_t fsize = (size_t) LearnSet.tellg();
    std::cout << "The learn file size is " << fsize << std::endl;
    LearnNum = (unsigned) (fsize / (Dimension + sizeof(uint32_t)/sizeof(data_t)) / sizeof(data_t));
    std::cout << "The learn set size is " << LearnNum << std::endl;

    //load learn set 
    LearnSet.seekg(0, std::ios::beg);
    std::vector<float> LearnVectors(Dimension * LearnNum);
    std::cout << "Loading Learn Set " << std::endl;
    readXvecFvec<data_t>(LearnSet, LearnVectors.data(), Dimension, LearnNum, true);
    std::cout << "Loaded Learn Set " << std::endl;
    LearnSet.close();

    //Vector Quantization Parameter
    std::vector<float> centroids (ncentroids * Dimension) ;
    std::cout << "Starting building centroids for vector quantization " << std::endl;
    
    //Generate kmeans centroids
    if (train_vector_quantization){
        std::cout << "Training Centroids " << std::endl;
        float obj = faiss::kmeans_clustering(Dimension, LearnNum, ncentroids, LearnVectors.data(), centroids.data());
        PrintVector<float>(centroids.data(), Dimension, ncentroids);
        std::cout << "Trained Centroids with obj " << obj << std::endl;
        std::ofstream CentroidSave;
        CentroidSave.open(CentroidsSavePath, std::ios::binary);
        std::cout << "Saving Centroids " << std::endl;
        writeXvec<float>(CentroidSave, centroids.data(), Dimension, ncentroids);
        std::cout << "Saved Centroids " << std::endl;
        CentroidSave.close();
    }

    std::cout << "Loading Centroids " << std::endl;
    std::ifstream CentroidRead;
    CentroidRead.open(CentroidsSavePath, std::ios::binary);
    readXvec<float>(CentroidRead, centroids.data(), Dimension, ncentroids, true);
    std::cout << "Loaded Centroids " << std::endl;
    CentroidRead.close();

    //load base set and assign
    size_t BaseNum;
    std::ifstream BaseSet;
    BaseSet.open(BasePath, std::ios::binary);
    BaseSet.seekg(0, std::ios::end);
    fsize = (size_t) BaseSet.tellg();
    std::cout << "The base set file size is " << fsize << std::endl;
    BaseNum = (unsigned) (fsize / (Dimension + sizeof(uint32_t)/sizeof(data_t)) / sizeof(data_t));
    std::cout << "The base set size is " << BaseNum << std::endl;

    std::vector<ID_T> VectorQuantID (BaseNum);
    std::cout << "Building ID for vector quantization" << std::endl; 
    size_t IDDimension = 100;
    size_t IDNum = BaseNum / IDDimension;
    if (assign_vertor_quantization){
        std::cout << "Assigning all base vectors " << std::endl;
        BaseSet.seekg(0, std::ios::beg);
        assign<data_t>(BaseSet, centroids, VectorQuantID.data(), 
                            Dimension, BaseNum, ncentroids);
        std::cout << "Saving computed ID" << std::endl;
        std::ofstream VqIDsWrite;
        VqIDsWrite.open(ComputedVQIdsPath,std::ios::binary);
        writeXvec<ID_T>(VqIDsWrite, VectorQuantID.data(), IDDimension, IDNum);
    }
    BaseSet.close();

    std::cout << "Loading computed VQ ID" << std::endl;
    std::ifstream VqIDsRead;
    VqIDsRead.open(ComputedVQIdsPath, std::ios::binary);
    VqIDsRead.seekg(0, std::ios::end);
    fsize = (size_t) VqIDsRead.tellg();
    std::cout << "The ID set file size is " << fsize << std::endl;
    IDNum = (unsigned) (fsize / (IDDimension * sizeof(ID_T) + sizeof(uint32_t)));
    std::cout << "The ID set size is " << IDNum << std::endl;
    readXvec<ID_T>(VqIDsRead, VectorQuantID.data(), IDDimension, IDNum, false);

    std::cout << "Generating Line quantization Layer " << std::endl;


}