#include <fstream>
#include <unordered_set>
#include <sys/resource.h>
#include <string>

#include "bslib_index.h"
#include "parameters/parameters_millions_VQ.h"

/**
 * 
 * This is the test file for multiple modules in my program.
 * 
 **/

using namespace bslib;
int main(){
    bool TEST_VQ = false;
    bool TEST_LQ = false;
    bool TEST_PQ = false;
    bool TEST_ALGO = false;
    bool TEST_DATA = true; 

    /**
     * Testing VQ part
    **/

   /**
    * Testing LQ part
    **/

   /**
    * Testing PQ part
    **/
   
   /**
    * Testing algorithm part
    **/

   /**
    * Testing data generated
    **/
   if (TEST_DATA){

       std::vector<idx_t> origin_ids(nb);
       std::ifstream ids_input(path_ids, std::ios::binary);
       readXvec<idx_t>(ids_input, origin_ids.data(), batch_size, nbatches);
       

       std::string path_seq_ids = "/home/y/yujianfu/ivf-hnsw/models_VQ/SIFT1M/base_idxs_5000_seq.ivecs";
       std::vector<idx_t> seq_origin_ids(nb);
       std::ifstream seq_ids_input(path_seq_ids, std::ios::binary);
       readXvec<idx_t> (seq_ids_input, seq_origin_ids.data(), batch_size, nbatches);
       //for (size_t i = 0; i < nb / 10; i++){if (seq_origin_ids[i] == origin_ids[i]) std::cout << "t" << i; else std::cout << "f";}
    for (size_t i = 0; i < 4; i++){
        for (size_t j = 0; j < batch_size; j++){
            std::cout << origin_ids[i * batch_size + j] << " ";
        }
        std::cout << std::endl;
    }


        //Bslib_Index index = Bslib_Index(dimension, layers, index_type, use_HNSW_VQ, use_norm_quantization);
        //index.read_index(path_index);

        /*
        std::ifstream seq_base_input(path_base, std::ios::binary);
        std::vector<float> seq_batch(batch_size * dimension);
        for (size_t i = 0; i < 3; i++){
            readXvecFvec<base_data_type> (seq_base_input, seq_batch.data(), dimension, batch_size);
        }

        size_t i = 2;
        std::vector<float> batch(batch_size * dimension);
        std::ifstream base_input(path_base, std::ios::binary);
        base_input.seekg(i * batch_size * dimension * sizeof(base_data_type) + i * batch_size * sizeof(uint32_t), std::ios::beg);
        readXvecFvec<base_data_type> (base_input, batch.data(), dimension, batch_size);

        for (size_t i = 0; i < batch_size * dimension; i++){
            std::cout << i << " " << batch[i] - seq_batch[i] << " ";
        }
        */




   

   }
   
   


}