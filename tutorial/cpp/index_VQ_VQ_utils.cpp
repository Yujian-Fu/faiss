#include "index_VQ_VQ_utils.h"

namespace bslib_VQ_VQ{

void random_subset(const float *x, float *x_out, size_t d, size_t nx, size_t sub_nx) {
    long seed = 1234;
    std::vector<int> perm(nx);
    faiss::rand_perm(perm.data(), nx, seed);

    for (size_t i = 0; i < sub_nx; i++)
        memcpy(x_out + i * d, x + perm[i] * d, sizeof(x_out[0]) * d);
}
}