#include<iostream>
#include <queue>
#include <limits>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/time.h>


class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

/// Read fvec/ivec/bvec format vectors
template<typename T>
void readXvec(std::ifstream &in, T *data, const size_t d, const size_t n = 1)
{
    uint32_t dim = d;
    for (size_t i = 0; i < n; i++) {
        in.seekg(sizeof(uint32_t)+sizeof(T)*dim), std::ios::beg);
        in.read((char *) &dim, sizeof(uint32_t));
        if (dim != d) {
            std::cout << "file error\n";
            exit(1);
        }
        in.read((char *) (data + i * dim), dim * sizeof(T));
    }
}

int main(){
    size_t Dimension = 128;
    size_t LearnNum;
    const char * LearnPath = "/home/y/yujianfu/ivf-hnsw/data/SIFT1B/bigann_learn.bvecs";
    std::ifstream LearnSet;
    LearnSet.open(LearnPath, std::ios::binary);
    LearnSet.read((char *) &Dimension, sizeof(uint32_t));
    std::cout << "The dimension of this dataset is " << Dimension << std::endl;
    LearnSet.seekg(0, std::ios::end);
    size_t fsize = (size_t) LearnSet.tellg();
    std::cout << "The learn file size is " << fsize << std::endl;
    LearnNum = (unsigned) (fsize / (Dimension + sizeof(uint32_t)/sizeof(uint8_t)) / sizeof(uint8_t));
    std::cout << "The learn set size is " << LearnNum << std::endl;
    LearnSet.seekg(0, std::ios::beg);
    std::vector<uint8_t> LearnVectors(Dimension * 10);
    StopW stopw = StopW();
    size_t subset = 10;
    readXvec<uint8_t>(LearnSet, LearnVectors.data(), Dimension, subset);
    std::cout << "The time for reading " << subset << " instances is " << stopw.getElapsedTimeMicro() << " us" << std::endl;
}
