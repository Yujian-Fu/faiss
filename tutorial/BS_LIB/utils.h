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
                output_record << s << "The time usage: " << getTimeConsumption() << std::endl;
            }

            void print_time_usage(std::string s){
                std::cout << s << " The time usage: " << getTimeConsumption() << std::endl; 
            }
    };

    struct memory_recorder{
        public:
            
        void record_memory_usage(std::ofstream & output_record, std::string s){
            rusage r_usage;
            getrusage(RUSAGE_SELF, &r_usage);
            output_record << s << "The memory usage: " <<  r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;
        }

        void print_memory_usage(std::string s){
            rusage r_usage;
            getrusage(RUSAGE_SELF, &r_usage);
            std::cout << s << "The memory usage: " <<  r_usage.ru_ixrss << " / " << r_usage.ru_isrss << " / " << r_usage.ru_idrss << " / " << r_usage.ru_maxrss << std::endl;
        }
    };

    template<typename T>
    void CheckResult(T * data, const size_t dimension){
        std::cout << "Printing sample (1 vector) of the dataset " << std::endl;
        for (size_t i= 0; i < 10; i++)
        {
            for (size_t j = 0; j < dimension; j++){
                std::cout << data[i * dimension + j] << " ";
            }
            std::cout << std::endl << std::endl;
        }
    }

    template<typename T>
    void readXvec(std::ifstream & in, T * data, const size_t dimension, const size_t n,
                  bool CheckFlag = false, bool ShowProcess = false){
        if (ShowProcess)
        std::cout << "Loading data with " << n << " vectors in " << dimension << std::endl;
        uint32_t dim = dimension;
        size_t print_every = n / 10;
        for (size_t i = 0; i < n; i++){
            in.read((char *) &dim, sizeof(uint32_t));
            if (dim != dimension){
                std::cout << dim << " " << dimension << " dimension error \n";
                exit(1);
            }
            if ( ShowProcess && print_every != 0 && i % print_every == 0)
                std::cout << "[Finished loading " << i << " / " << n << "]"  << std::endl; 
        }
        if (CheckFlag)
            CheckResult<T>(data, dimension);
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
            CheckResult<float>(data, dimension);
    }

    inline bool exists(const char * FilePath){
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

    inline void ShowMessage(std::string s){
        std::cout << s << std::endl;
    }

    inline void PrepareFolder(const char * FilePath){
        if(NULL==opendir(FilePath))
        mkdir(FilePath, S_IRWXU); //Have the right to read, write and execute
    }
    
}