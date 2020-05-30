#include <iostream>
#include <fstream>
#include <cstdio>
#include <unordered_map>

#include<faiss/impl/ProductQuantizer.h>

int main(){
    size_t dimension = 128;
    size_t code_size = 4;
    size_t train_size = 100000;
    size_t n = 200000;

    float * x_train = new float [dimension * train_size];
    float * x_base = new float [dimension * n];
    float * x_code = new float [dimension * n];

    for (int i = 0; i < train_size; i++){
        for (int j = 0; j < dimension; j++)
            x_train[ dimension * i + j ] = drand48();
        x_train[dimension * i] += i / 1000;
    }

    for (int i = 0; i < train_size; i++){
        for (int j = 0; j < dimension; j++)
            x_base[ dimension * i + j ] = drand48();
        x_base[dimension * i] += i / 1000;
    }

    std::cout << "Data generated successfully" << std::endl;

    faiss::ProductQuantizer pq = faiss::ProductQuantizer(dimension, code_size, 8);

    std::cout << "Start training " << std::endl;
    pq.train(train_size, x_train);

    uint8_t * codes = new uint8_t [dimension * n];
    
    pq.compute_codes(x_base, codes, n);

    pq.decode(codes, x_code);

}

