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
       for (size_t i = 0; i < 100; i++){std::cout << origin_ids[i] << " ";}
   }
   
   


}