#include "fstream"
#include "iostream"
#include <faiss/utils/random.h>
#include <chrono>
#include <string.h>
#include <fstream>
#include <sys/resource.h>
#include <sys/stat.h>
#include <dirent.h>

namespace bslib{
    struct time_recorder{
        std::chrono::steady_clock::time_point start_time;
        public:
            time_recorder(){
                start_time = std::chrono::steady_clock::now();
            }

            float getTimeConsumption(){
                std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
                return (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
            }

            void reset(){
                start_time = std::chrono::steady_clock::now();
            }

            void record_time_usage(std::ofstream & output_record, std::string s){
                output_record << s << "The time usage: " << getTimeConsumption() / 1000000 << " s " << std::endl;
            }

            void print_time_usage(std::string s){
                std::cout << s << " The time usage: " << getTimeConsumption() / 1000000 << " s "<< std::endl; 
            }

            float get_time_usage(){
                return getTimeConsumption() / 1000000;
            }
    };

    struct memory_recorder{
        public:
            
        void record_memory_usage(std::ofstream & output_record, std::string s){
            rusage r_usage;
            getrusage(RUSAGE_SELF, &r_usage);
            output_record << s << " The memory usage: " <<  r_usage.ru_ixrss << " KB / " << r_usage.ru_isrss << " KB / " << r_usage.ru_idrss << " KB / " << r_usage.ru_maxrss <<  " KB " << std::endl;
        }

        void print_memory_usage(std::string s){
            rusage r_usage;
            getrusage(RUSAGE_SELF, &r_usage);
            std::cout << s << " The memory usage: " <<  r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;
        }
    };

    struct recall_recorder{
        public:

        void print_recall_performance(size_t n_query, float recall, size_t recall_k, std::string mode, size_t layers, const size_t * search_paras, size_t max_visited_vectors){
            std::cout << "The recall@ " << recall_k << " for " << n_query << " queries in " << mode << " mode is: " << recall <<  std::endl;
            std::cout << "The search parameters is: ";
            for (size_t i = 0; i < layers; i++){std::cout << search_paras[i] << " ";}
            std::cout << "The max ivisited vectors is: " << max_visited_vectors << std::endl;
        }

        void record_recall_performance(std::ofstream & output_record, size_t n_query, float recall, size_t recall_k, std::string mode, size_t layers, const size_t * search_paras, size_t max_visited_vectors){
            output_record << "The recall@" << recall_k << " for " << n_query << " queries in " << mode << " mode is: " << recall <<  std::endl;
            output_record << "The search parameters is ";
            for (size_t i = 0; i < layers; i++){output_record << search_paras[i] << " ";}
            output_record << "The max ivisited vectors is: " << max_visited_vectors << std::endl;
        }
    };

    template<typename T>
    void CheckResult(T * data, const size_t dimension, size_t dataset_size = 2){
        std::cout << "Printing sample (2 vectors) of the dataset " << std::endl;

        for (size_t i = 0; i < dimension; i++){
            std::cout << data[i] << " ";
        }
        std::cout << std::endl << std::endl;
        for (size_t i = 0; i< dimension; i++){
            std::cout << data[(dataset_size-1) * dimension+i] << " ";
        }
        std::cout << std::endl << std::endl;
    }

    template<typename T>
    void readXvec(std::ifstream & in, T * data, const size_t dimension, const size_t n,
                  bool CheckFlag = false, bool ShowProcess = false){
        if (ShowProcess)
        std::cout << "Loading data with " << n << " vectors in " << dimension << std::endl;
        uint32_t dim = dimension;
        size_t print_every = n / 10;
        for (size_t i = 0; i < n; i++){
            in.read((char *) & dim, sizeof(uint32_t));
            if (dim != dimension){
                std::cout << dim << " " << dimension << " dimension error \n";
                exit(1);
            }
            in.read((char *) (data + i * dim), dim * sizeof(T));
            if ( ShowProcess && print_every != 0 && i % print_every == 0)
                std::cout << "[Finished loading " << i << " / " << n << "]"  << std::endl; 
        }
        if (CheckFlag)
            CheckResult<T>(data, dimension, n);
    }

    template<typename T>
    uint32_t GetXvecSize(std::ifstream & in, const size_t dimension){
        in.seekg(0, std::ios::end);
        size_t FileSize = (size_t) in.tellg();
        std::cout << "The file size is " << FileSize / 1000 << " KB " << std::endl;
        size_t DataSize = (unsigned) (FileSize / (dimension * sizeof(T) + sizeof(uint32_t)));
        std::cout << "The data size is " << DataSize << std::endl;
        return DataSize;
    }

    template<typename T>
    void readXvecFvec(std::ifstream & in, float * data, const size_t dimension, const size_t n = 1,
                      bool CheckFlag = false, bool ShowProcess = false){
        if (ShowProcess)
        std::cout << "Loading data with " << n << " vectors in " << dimension << std::endl;
        uint32_t dim = dimension;
        T origin_data[dimension];
        size_t print_every = n / 10;
        for (size_t i = 0; i < n; i++){
            in.read((char * ) & dim, sizeof(uint32_t));
            if (dim != dimension) {
                std::cout << dim << " " << dimension << " dimension error \n";
                exit(1);
            }
            in.read((char * ) & origin_data, dim * sizeof(T));
            for (size_t j = 0; j < dimension; j++){
                data[i * dim + j] = 1.0 * origin_data[j];
            }
            if ( ShowProcess && print_every != 0 && i % (print_every) == 0)
                std::cout << "[Finished loading " << i << " / " << n << "]" << std::endl; 
        }
        if (CheckFlag)
            CheckResult<float>(data, dimension, n);
    }


    inline bool exists(const std::string FilePath){
        std::ifstream f (FilePath);
        return f.good();
    }

    template<typename T>
    void RandomSubset(const T * x, T * output, size_t dimension, size_t n, size_t sub_n){
        long RandomSeed = 1234;
        std::vector<int> RandomId(n);
        faiss::rand_perm(RandomId.data(), n, RandomSeed);

        for (size_t i = 0; i < sub_n; i++)
            memcpy(output + i * dimension, x + RandomId[i] * dimension, sizeof(T) * dimension);
    }

    inline void PrintMessage(std::string s){
        std::cout << s << std::endl;
    }

    inline void PrepareFolder(const char * FilePath){
        if(NULL==opendir(FilePath))
        mkdir(FilePath, S_IRWXU); //Have the right to read, write and execute
    }

    template<typename T>
    inline void HashMapping(size_t n, const T * group_ids, T * hash_ids, size_t hash_size){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            hash_ids[i] = group_ids[i] % hash_size;
    }


}
