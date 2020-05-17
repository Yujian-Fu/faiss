#include <fstream>
#include <iostream>
#include <stdio.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/random.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
#include <string>

typedef faiss::Index::idx_t ID_T;


void ReportFileError(std::string s){
    std::cerr<< s << std::endl;
    exit(1);
}


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


void random_subset(const float *x, float *x_out, size_t d, 
                   size_t nx, size_t sub_nx) {
    long seed = 1234;
    std::vector<int> perm(nx);
    faiss::rand_perm(perm.data(), nx, seed);

    for (size_t i = 0; i < sub_nx; i++)
        memcpy(x_out + i * d, x + perm[i] * d, sizeof(x_out[0]) * d);
}

template<typename T>
void readXvec(std::ifstream & in, T *data, const size_t dimension, 
              const size_t num_vector = 1, bool print_flag = false){
    size_t dim = dimension;
    for (size_t i = 0; i < num_vector; i++){
        in.read((char *) &dim, sizeof(uint32_t));
        if (dim != dimension){
            std::cout << dim << " " << dimension << " dimension error \n";
            exit(1);
        }
        in.read((char *) (data + i * dim), dim*sizeof(T));
        if ( i % (num_vector / 100) == 0)
            std::cout << "[Finished loading " << i << " / " << num_vector << "]" << std::endl; 
    }
    if (print_flag)
        PrintVector<T>(data, dimension, num_vector);
}

/// Read fvec/ivec/bvec format vectors and convert them to the float array
template<typename T>
void readXvecFvec(std::ifstream & in, float *data, const size_t dimension, 
                  const size_t num_vector = 1, bool print_flag = false)
{
    std::cout << "Loading data with " << num_vector << " vectors in " << dimension << std::endl;
    uint32_t dim = dimension;
    T mass[dimension];
    size_t print_every = num_vector / 10;
    for (size_t i = 0; i < num_vector; i++) {
        in.read((char *) &dim, sizeof(uint32_t));
        if (dim != dimension) {
            std::cout << "file error\n";
            exit(1);
        }
        in.read((char *) mass, dim * sizeof(T));
        for (size_t j = 0; j < dimension; j++){
            data[i * dim + j] = 1. * mass[j];
        }
        if ( i % (print_every) == 0)
            std::cout << "[Finished loading " << i << " / " << num_vector << "]" << std::endl; 
    }
    if (print_flag)
        PrintVector<float>(data, dimension, num_vector);
    //PrintVector(data, dimension, num_vector);
}

template<typename T>
void writeXvec(std::ofstream & output, const T * vector, size_t dimension, 
               size_t num){
    for (size_t i = 0;  i< num; i++){
        const uint32_t size = dimension;
        output.write((char *) &size, sizeof(uint32_t));
        output.write((char *) (vector + dimension * i), dimension * sizeof(T));
    }
}

template<typename Data_T>
void assign(std::ifstream & Dataset, std::vector<float> codewords, ID_T * IDs,
            size_t Dimension, size_t num_vector, size_t NumCodewords){
    ID_T KNeighbors = 1;
    Data_T mass[Dimension];
    float data[Dimension];
    size_t dim = Dimension;
    std::vector<float> dis (KNeighbors);
    std::vector<ID_T> ids (KNeighbors);

    std::cout << "Building FlatL2 Search Index " << std::endl;
    faiss::IndexFlatL2 index (Dimension);
    index.add(NumCodewords, codewords.data());

    size_t print_every = num_vector / 10;

    for (size_t i = 0; i < num_vector; i++){
        Dataset.read((char *) &dim, sizeof(uint32_t));
        if (dim != Dimension){
            std::cout << dim << " " << Dimension << "Dimension error\n";
            exit(1);
        }
        Dataset.read((char *) mass, dim * sizeof(Data_T));
        for (size_t j = 0; j < dim; j++){
            data[j] = 1. * mass[j];
        }
        index.search(1, data, KNeighbors, dis.data(),ids.data());
        IDs[i] = ids[0];
        if (i % print_every == 0)
            std::cout << "[Finished: " << i << " / " << num_vector << "] " <<std::endl; 
    }
}

/*
void read_fvecs(const char* filename, float* &data, 
                size_t &num_points, size_t &dim, bool print_flag = false){
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open())
    {
        std::cout << "file open error" << std::endl;
        exit(-1);
    }

    infile.read((char*)&dim, 4);
    infile.seekg(0, std::ios::end);
    std::ios::pos_type ss = infile.tellg();
    size_t fsize = (size_t) ss;
    num_points =  (unsigned)(fsize / (dim + 1) /4);
    data = new float[(size_t)num_points * (size_t)dim];

    infile.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_points; i++)
    {
        infile.seekg(4, std::ios::cur);
        infile.read((char*)(data + i*dim), dim * 4);
    }

    if (print_flag)
    for (size_t i= 0; i < num_points * dim; i++)
    {
        std::cout << (float)data[i];
        if (!i)
        {
            std::cout << " ";
            continue;
        }

        if (i % (dim - 1) != 0)
        {
            std::cout << " ";
        }
        else
        {
            std::cout << std::endl;
        }
    }

    infile.close();
}
*/