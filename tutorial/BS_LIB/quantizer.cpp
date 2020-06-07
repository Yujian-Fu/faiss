#include "quantizer.h"

namespace bslib{
    Base_quantizer::Base_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group):
        dimension(dimension), nc_upper(nc_upper), nc_per_group(nc_per_group){
            nc = nc_upper * nc_per_group;
            CentroidDistributionMap.resize(nc_upper);
            size_t num_centroids = 0;
            for (size_t i = 0; i < nc_upper; i++){
                size_t group_size = nc_per_group;
                this->CentroidDistributionMap[i] = num_centroids;
                num_centroids += group_size;
            }
        }
}