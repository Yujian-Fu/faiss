#include <iostream>
#include <fstream>
#include <stdlib.h>
#include<faiss/IndexFlat.h>
#include<faiss/utils/utils.h>
#include<string.h>



int main(){

    size_t ncentroids = 1000000;
    size_t dimension = 10;
    std::vector<float> centroids (ncentroids * dimension);
    size_t k_neighbors = 5;


    for (int i = 0; i < ncentroids; i++){
        for (int j = 0; j < dimension; j++)
            centroids[ dimension * i + j ] = drand48();
        centroids[dimension * i] += i / 1000;
    }

    faiss::IndexFlatL2 index (dimension);

    index.add(ncentroids, centroids.data());
    std::vector<float> alphas (ncentroids);

    for (int i = 0; i < ncentroids; i++){
        alphas[i] = drand48();
    }

    long * ids = new long[1 * (k_neighbors + 1)];
    float * dis = new float[1 * (k_neighbors + 1)];
    for (int i = 0; i < ncentroids; i++){
        std::vector<float> subcentroids (k_neighbors * dimension);
        std::cout << "The centroid is : " << std::endl;
        for (int temp = 0; temp < dimension; temp ++){
            std::cout << centroids[i * dimension + temp] << " ";
        }
        std::cout << std::endl;
        index.search(1, centroids.data() + i * dimension, k_neighbors+1, dis, ids);
        std::vector<float> centroid_vector(dimension);
        for (int j = 1; j < k_neighbors+1; j++)
        {
            std::cout << "The  is the nearest neighbor: " << std::endl;
            for (int temp = 0; temp < dimension; temp ++){
                std::cout << centroids[ids[j] * dimension + temp] << " ";
            }
            std::cout << std::endl;
            faiss::fvec_madd(dimension, centroids.data() + ids[j] * dimension, -1.0, centroids.data() + i * dimension, centroid_vector.data());
            std::cout << "The  is the neighbor vector: " << std::endl;
            for (int temp = 0; temp < dimension; temp ++){
                std::cout << centroid_vector[temp] << " ";
            }
            std::cout << std::endl;
            faiss::fvec_madd(dimension, centroids.data() + i * dimension, alphas[i], centroid_vector.data(), subcentroids.data() + (j-1) * dimension);
            std::cout << "The  is the subcentroid: " << std::endl;
            for (int temp = 0; temp < dimension; temp ++){
                std::cout << subcentroids[(j-1)*dimension + temp] << " ";
            }
            std::cout << std::endl << std::endl;;
        }
    }
    delete[] ids;
    delete[] dis; 
}


