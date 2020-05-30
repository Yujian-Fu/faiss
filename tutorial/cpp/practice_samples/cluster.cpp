#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <faiss/Clustering.h>

int main(){
    
    size_t n = 1000000;
    size_t ncentroids = n / 1000;
    size_t dimension = 128;
    float obj;

    float * xb = new float [dimension * n];
    float * centroids = new float [dimension * ncentroids];

    for (int i = 0; i < n; i++){
        for (int j = 0; j < dimension; j++)
            xb[ dimension * i + j ] = drand48();
        xb[dimension * i] += i / 1000;
    }

    obj = faiss::kmeans_clustering(dimension, n, ncentroids, xb, centroids);

    
    for (int i = 0; i < dimension; i++){
        std::cout << centroids[i] << " ";
        if ((i % dimension) == 0)
            std::cout << std::endl;
    }
    

   std::cout << std::endl << "The imbalance is " << obj << std::endl;
   delete [] xb;
   delete [] centroids;
}