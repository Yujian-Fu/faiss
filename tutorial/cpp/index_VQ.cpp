#include "index_VQ.h"

namespace bslib{

    BS_LIB::BS_LIB(size_t dimension, size_t ncentroids, size_t bytes_per_code, size_t nbits_per_idx, bool use_quantized_distance, size_t max_group_size):
            dimension(dimension), nc(ncentroids),  use_quantized_distance(use_quantized_distance), pq(nullptr), norm_pq(nullptr)
    {
        this->pq = new faiss::ProductQuantizer(dimension, bytes_per_code, nbits_per_idx);
        this->norm_pq = new faiss::ProductQuantizer(1, 1, nbits_per_idx);

        code_size = pq->code_size;
        norms.resize(max_group_size);
        precomputed_table.resize(pq->ksub * pq->M);

        base_codes.resize(nc);
        base_norm_codes.resize(nc);
        ids.resize(nc);
        centroid_norms.resize(nc);
    }

    BS_LIB::~BS_LIB(){
        if (pq) delete pq;
        if (norm_pq) delete norm_pq;
    }

    void BS_LIB::build_quantizer(const char * centroid_path){
        this->quantizer = new faiss::IndexFlatL2(dimension);
        std::cout << "Building quantizer " << std::endl;
        std::ifstream centroid_input(centroid_path, std::ios::binary);
        std::vector<float> centroids(dimension * nc);
        readXvec<float> (centroid_input, centroids.data(), dimension, nc, true);
        faiss::IndexFlatL2 quantizer (dimension);
        this->quantizer->add(nc, centroids.data());
    }

    void BS_LIB::compute_residuals(size_t n, const float * x, float * residuals, const idx_t * keys){
        for (size_t i = 0; i < n; i++){
            const float * centroid = this->quantizer->xb.data()+ keys[i] * dimension;
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, centroid, residuals+i*dimension);
        }
    }

    void BS_LIB::assign(size_t n, const float * x, idx_t * labels, size_t k){
        this->quantizer->assign(n, x, labels, k);
    }

    void BS_LIB::reconstruct(size_t n, float * x, const float * decoded_residuals, const idx_t * keys){
        for (size_t i = 0; i < n; i++){
            const float * centroid = this->quantizer->xb.data()+keys[i] * dimension;
            faiss::fvec_madd(dimension, decoded_residuals + i * dimension, 1.0, centroid, x + i*dimension);
        }
    }

    void BS_LIB::train_pq(size_t n, const float * x, bool train_pq, bool train_norm_pq){
        std::cout << "The inserted centroids size is " << this->quantizer->xb.size() / n << std::endl;
        std::cout << "Assigning train data points " << std::endl;
        std::vector<idx_t> assigned_ids(n);
        assign(n, x, assigned_ids.data());

        std::cout << "The assigned data examples are: " << std::endl;
        for (int i = 0; i < 10; i++)
        {
            std::cout << assigned_ids[i] << " "; 
        }
        std::vector<float> residuals (n * dimension);
        compute_residuals(n, x, residuals.data(), assigned_ids.data());

        if (train_pq){
            std::cout << "Training residual PQ with parameter setting: M: " << pq->M << " number of centroids: " << pq->ksub << std::endl;
            this->pq->verbose = true;
            this->pq->train(n, residuals.data());
        }

        if (train_norm_pq){
            //Compute the code of vector
            std::cout << "Compute the code of the train vectors " << std::endl;
            std::vector <uint8_t> residual_codes(n * code_size);
            this->pq->compute_code(residuals.data(), residual_codes.data());

            std::cout << "Decode the code of the train vectors " << std::endl;
            //Decode residuals, compute the centroid representation of 
            std::vector<float> decoded_residuals(n * dimension);
            this->pq->decode(residual_codes.data(), decoded_residuals.data());

            std::cout << "Reconstructing the vectors " << std::endl;
            std::vector<float> reconstructed_x(n * dimension);
            reconstruct(n, reconstructed_x.data(), decoded_residuals.data(), assigned_ids.data());
            
            std::cout << "Computing the norm of vectors " << std::endl;
            std::vector<float> norms(n);
            faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), dimension, n);

            std::cout << "Using quantized distance, training norm_pq with parameter: " << "M: " << norm_pq->M << "nc: " << norm_pq->ksub << std::endl;
            this->norm_pq->verbose = true;
            this->norm_pq->train(n, norms.data());
        }
    }

    void BS_LIB::add_batch(size_t n, const float * x, const idx_t * origin_ids, const idx_t * quantization_ids){
        std::cout << "Adding a batch" << std::endl;
        const idx_t  * idx;
        if (quantization_ids)
            idx = quantization_ids;
        else{
            idx = new idx_t[n];
            assign(n, x, const_cast<idx_t *> (idx));
        }

        //std::cout << "Computing residuals " << std::endl;
        std::vector<float> residuals(n * dimension);
        compute_residuals(n, x, residuals.data(), idx);

        //std::cout << "Encoding the residual " << std::endl;
        std::vector<uint8_t> residual_codes(n * code_size);
        this->pq->compute_codes(residuals.data(), residual_codes.data(), n);

        //std::cout << "Decoding the residual " << std::endl;
        std::vector<float> decoded_residuals(n * dimension);
        this->pq->decode(residual_codes.data(), decoded_residuals.data(), n);

        //std::cout << "reconstructing the base vector " << std::endl;
        std::vector<float> reconstructed_x(n * dimension);
        reconstruct(n, reconstructed_x.data(), decoded_residuals.data(), idx);

        //std::cout << "Computing the norm of the vector " << std::endl;
        std::vector<float> norms(n);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), dimension, n);

        //std::cout << "Adding norm to codes " << std::endl;
        if (use_quantized_distance){
            std::vector<uint8_t> x_norm_codes(n);
            this->norm_pq->compute_codes(norms.data(), x_norm_codes.data(), n);

            for (size_t i = 0; i < n; i++){
                const idx_t key = idx[i];
                base_norm_codes[key].push_back(x_norm_codes[i]);
            }
        }
        else{
            for (size_t i = 0; i < n; i++){
                const idx_t key = idx[i];
                base_norms[key].push_back(norms[i]);
            }
        }

        //std::cout << "Adding codes to groups " << std::endl;
        for (size_t i = 0; i < n; i++){
            const idx_t key = idx[i];
            const idx_t id = origin_ids[i];
            ids[key].push_back(id);
            const uint8_t * code = residual_codes.data() + i * code_size;
            for (size_t j = 0; j < code_size; j++)
                base_codes[key].push_back(code[j]);
        }

        // Free memory, if it is allocated 
        if (idx != quantization_ids)
            delete idx;
    }


    float BS_LIB::pq_L2sqr(const uint8_t *code)
    {
        float result = 0.;
        const size_t dim = code_size >> 2;
        size_t m = 0;
        for (size_t i = 0; i < dim; i++) {
            result += precomputed_table[this->pq->ksub * m + code[m]]; m++;
            result += precomputed_table[this->pq->ksub * m + code[m]]; m++;
            result += precomputed_table[this->pq->ksub * m + code[m]]; m++;
            result += precomputed_table[this->pq->ksub * m + code[m]]; m++;
        }
        return result;
    }


    void BS_LIB::search(size_t nq, size_t k, const float * x, float * centroid_dists, idx_t * centroid_idxs){
        //float centroid_dists[nq * k];
        //idx_t centroid_idxs[nq * k];

        this->quantizer->search(nq, x, k, centroid_dists, centroid_idxs);

        for(size_t query_id = 0; query_id < nq; query_id++){
            float query_distances[k];
            idx_t query_labels[k];
            const float * query = x + query_id * dimension;
            this->pq->compute_inner_prod_table(query, precomputed_table.data());
            faiss::maxheap_heapify(k, query_distances, query_labels);
            size_t ncode = 0;
            for (int i = 0; i < nprobe; i++){
                const idx_t centroid_idx = centroid_idxs[query_id * nprobe + i];
                const size_t group_size = (use_quantized_distance) ? base_norm_codes[centroid_idx].size() : base_norms[centroid_idx].size();
                if (group_size == 0)
                    continue;
                
                const uint8_t * code = base_codes[centroid_idx].data();
                const idx_t * id = ids[centroid_idx].data();
                const float term1 = centroid_dists[query_id * k + i] - centroid_norms[centroid_idx];

                if (use_quantized_distance){
                    const uint8_t * norm_code = base_norm_codes[centroid_idx].data();
                    this->norm_pq->decode(norm_code, norms.data(), group_size);
                }
                else{
                    norms = base_norms[centroid_idx];
                }

                for(size_t j = 0; j < group_size; j++){
                    const float term3 = 2 * pq_L2sqr(code + j * code_size);
                    const float dist = term1 + norms[j] - term3;
                    if (dist < query_distances[0]){
                        faiss::maxheap_pop(k, query_distances, query_labels);
                        faiss::maxheap_push(k, query_distances, query_labels, dist, id[j]);
                    }
                }

                ncode += group_size;
                if(ncode >= max_codes)
                    break;
            }

            for (int i = 0; i < k; i++){
                centroid_dists[query_id * k + i] = query_distances[i];
                centroid_idxs[query_id * k + i] = query_labels[i];
            }
        }
    }

    

    void BS_LIB::write(const char * path_index){
        std::ofstream output (path_index, std::ios::binary);

        write_variable(output, dimension);
        write_variable(output, nc);

        for (size_t i = 0; i < nc; i++)
            write_vector(output, ids[i]);
        
        for (size_t i = 0; i < nc; i++)
            write_vector(output, base_codes[i]);
        
        if (use_quantized_distance)
        for (size_t i = 0; i < nc; i++)
            write_vector(output, base_norm_codes[i]);

        write_vector(output, centroid_norms);
    }

    void BS_LIB::read(const char * path_index){
        std::ifstream input (path_index, std::ios::binary);

        read_variable(input, dimension);
        read_variable(input, nc);

        for(size_t i = 0; i < nc; i++)
            read_vector(input, ids[i]);

        for (size_t i = 0; i < nc; i++)
            read_vector(input, base_codes[i]);

        if (use_quantized_distance)
        for(size_t i = 0; i < nc; i++)
            read_vector(input, base_norm_codes[i]);

        read_vector(input, centroid_norms);

    }

    void BS_LIB::compute_centroid_norms(){
    for (size_t i = 0; i < nc; i++){
        const float * centroid = this->quantizer->xb.data()+ i * dimension;
        centroid_norms[i] = faiss::fvec_norm_L2sqr(centroid, dimension);
    }
    }
}

