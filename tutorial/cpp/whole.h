#include <fstream>
#include <iostream>
#include <stdio.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/random.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
#include <string>
#include <time.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

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
    std::cout << "Loading data with " << num_vector << " vectors in " << dimension << std::endl;
    size_t dim = dimension;
    size_t print_every = num_vector / 10;
    for (size_t i = 0; i < num_vector; i++){
        in.read((char *) &dim, sizeof(uint32_t));
        if (dim != dimension){
            std::cout << dim << " " << dimension << " dimension error \n";
            exit(1);
        }
        in.read((char *) (data + i * dim), dim * sizeof(T));
        if ( i % print_every == 0)
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
            std::cout << dim << " " << dimension << " dimension error \n";
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

//using gpu
template<typename Data_T>
void assign(std::ifstream & Dataset, std::vector<float> codewords, ID_T * IDs,
            size_t Dimension, size_t num_vector, size_t NumCodewords){
    ID_T KNeighbors = 1;
    Data_T mass[Dimension];
    float data[Dimension];
    size_t dim = Dimension;
    std::vector<float> dis (KNeighbors);
    std::vector<ID_T> ids (KNeighbors);
    clock_t start,end;
    /*
    bool use_gpu = false;

    if (use_gpu){
        //Build index in gpus
        std::cout << "Building FlatL2 Search Index with GPU" << std::endl;
        int ngpus = faiss::gpu::getNumDevices();
        std::cout << "Number of GPUs: " <<  ngpus << std::endl;
        std::vector<faiss::gpu::GpuResources*> res;
        std::vector<int> devs;
        for(int i = 0; i < ngpus; i++) {
            res.push_back(new faiss::gpu::StandardGpuResources);
            devs.push_back(i);
        }
        faiss::IndexFlatL2 cpu_index(Dimension);
        faiss::Index *index =
            faiss::gpu::index_cpu_to_gpu_multiple(
                res,
                devs,
                &cpu_index
            );
        std::cout << "is_trained = " << index->is_trained ? "true" : "false" << std::endl;
        index->add(NumCodewords, codewords.data());  // vectors to the index
        std::cout << "ntotal = " << index->ntotal <<std::endl;
    }
    else{
        faiss::IndexFlatL2 index (Dimension);
        index.add(NumCodewords, codewords.data());
    }
    */

    faiss::IndexFlatL2 index (Dimension);
    index.add(NumCodewords, codewords.data());
    size_t print_every = num_vector / 1000;

    start = clock();
#pragma omp parallel for
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
        if (i % 100 == 0)
            std::cout << std::endl << "[Finished: " << i << " / " << num_vector << "] in " << (double)(clock()-start)/CLOCKS_PER_SEC << std::endl; 
    }

/*
    delete index;

    if (use_gpu)
        for(int i = 0; i < ngpus; i++) {
            delete res[i];
        }
*/
}

template<typename Data_T> 
void compute_aphas(float * centroids, std::ifstream & Dataset, ID_T * IDs, 
                   float * alphas, size_t Dimension, size_t ncentroids){
    for (int i = 0; i < ncentroids; i++){
        alphas[i] = 0.1;
    }
}


template<typename Data_T>
void build_subcentroids(float * centroids, std::ifstream & Dataset, ID_T * IDs,
                        float * alphas, size_t Dimension, size_t ncentroids, size_t nsubc){
    for (size_t i =0; i <ncentroids; i++){
        std::vector<float> subcentroids (nsubc * Dimension);
        std::cout << "The centroids is : " << std::endl;
        for (size_t temp = 0; temp < )
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