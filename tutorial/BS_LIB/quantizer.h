#ifndef _QUANTIZER_H
#define _QUANTIZER_H
#include<iostream>
#include<fstream>
#include <cstdio>
#include <queue>
#include <math.h>
#include <map>

#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>



namespace bslib{
struct Base_quantizer
{

    typedef uint32_t idx_t;
    size_t dimension;
    size_t nc_upper;
    size_t nc_per_group;
    size_t nc;
    
    
    
    std::vector<idx_t> CentroidDistributionMap;
    
    explicit Base_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group);
};
}
#endif