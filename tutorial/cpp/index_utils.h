#include <iostream>
#include<faiss/utils/random.h>
#include <sys/time.h>
#include <chrono>



class StopW{
    std::chrono::steady_clock::time_point time_begin;
    public:
        StopW(){
            time_begin = std::chrono::steady_clock::now();
        }

        float getElapsedTimeMicro(){
            std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
            return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
        }

        void reset(){
            time_begin = std::chrono::steady_clock::now();
        }

};


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

template<typename T>
void XvecSize(std::ifstream & in, size_t & size, const size_t dimension){
    in.seekg(0, std::ios::end);
    size_t fsize = (size_t) in.tellg();
    std::cout << "The ID set file size is " << fsize << std::endl;
    size_t datasize = (unsigned) (fsize / (dimension * sizeof(T) + sizeof(uint32_t)));
    std::cout << "The ID set size is " << datasize << std::endl;
    in.seekg(0, std::ios::beg);
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

/// Check if file exists
inline bool exists(const char *path) {
    std::ifstream f(path);
    return f.good();
}

void random_subset(const float *x, float *x_out, size_t d, size_t nx, size_t sub_nx) {
    long seed = 1234;
    std::vector<int> perm(nx);
    faiss::rand_perm(perm.data(), nx, seed);

    for (size_t i = 0; i < sub_nx; i++)
        memcpy(x_out + i * d, x + perm[i] * d, sizeof(x_out[0]) * d);
}

template<typename T>
void write_variable(std::ofstream & out, const T & val){
    out.write((char *) & val, sizeof(T));
}

template<typename T>
void write_vector(std::ostream & out, std::vector<T> & vec){
    const uint32_t size = vec.size();
    out.write((char *) & size, sizeof(uint32_t));
    out.write((char *) vec.data(), size * sizeof(T));
}

template<typename T>
void read_variable(std::ifstream & in, T & val){
    in.read((char * ) & val, sizeof(T));
}

template<typename T>
void read_vector(std::ifstream & in, std::vector<T> & vec){
    uint32_t size;
    in.read((char *) & size, sizeof(uint32_t));
    vec.resize(size);
    in.read((char *)vec.data(), size * sizeof(T));
}

