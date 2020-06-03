#include "index_VQ_VQ.h"

namespace bslib_VQ_VQ{
    BS_LIB_VQ_VQ::BS_LIB_VQ_VQ(size_t dimension, size_t nc1, size_t nc2, size_t bytes_per_code, size_t nbits_per_idx):
        dimension(dimension), pq(nullptr), nc1(nc1), nc2(nc2) 
    {
        this->pq = new faiss::ProductQuantizer(dimension, bytes_per_code, nbits_per_idx);

        this->code_size = pq->code_size;
        
        ids.resize(nc1);
        base_codes.resize(nc1);
        base_norms.resize(nc1);
        centroid_norms.resize(nc1);
        quantizers.resize(nc1);
        precomputed_table.resize(pq->ksub * pq->M);
    }


    void BS_LIB_VQ_VQ::build_quantizer(const char * centroid_path, const char * subcentroid_path){
        this->quantizer = new faiss::IndexFlatL2(dimension);

        std::cout << "Building first level quantizer with centroids " << nc1 << std::endl;
        std::ifstream centroid_input(centroid_path, std::ios::binary);
        std::vector<float> centroids(nc1 * dimension);
        readXvec<float>(centroid_input, centroids.data(), dimension, nc1, true);
        this->quantizer->add(nc1, centroids.data());


        std::cout << "Building second level quantizer with centroids " << nc2 << std::endl;
        std::ifstream subcentroid_input(subcentroid_path, std::ios::binary);
        std::vector<float> subcentroids (nc2 * nc1 * dimension);
        readXvec<float>(subcentroid_input, subcentroids.data(), dimension, nc2 * nc1, true);
        for (size_t i = 0; i < nc1; i++){
            faiss::IndexFlatL2 * sub_quantizer = new faiss::IndexFlatL2(dimension);
            sub_quantizer->add(nc2, subcentroids.data() + i * nc2 * dimension);
            this->quantizers[i] = sub_quantizer;
            
            //if (i % 100 == 0)
                //std::cout << "Finished " << i << " / " << nc1 << "with quantizers size " << this->quantizers.size() << std::endl;
        }
        std::cout << "The size of quatizers is " << quantizers.size() << std::endl;
        
        // Checking the correctness of quantizer initialization
        assert(this->quantizer->xb.size() / dimension == nc1);
        for (size_t i = 0; i < nc1; i++)
        {
            assert(this->quantizers[i]->xb.size() / dimension == nc2);
        }
        std::cout << "All quantizers are added correctly " << std::endl;
    }

    void BS_LIB_VQ_VQ::assign(size_t n, const float * x, idx_t * labels, idx_t * sub_labels){
        this->quantizer->assign(n, x, labels);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            this->quantizers[labels[i]]->assign(1, x + i * dimension, sub_labels + i);
        }
    }

    void BS_LIB_VQ_VQ::compute_residuals(size_t n, const float * x, float * residuals, const idx_t * idxs, const idx_t * sub_idxs){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            const float * centroid = this->quantizers[idxs[i]]->xb.data()+ sub_idxs[i] * dimension;
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, centroid, residuals + i * dimension);
        }
    }

    void BS_LIB_VQ_VQ::train_pq(size_t n, const float * x){
        std::cout << "The existing first level centroid size is " << this->quantizer->xb.size() / dimension << std::endl;
        std::cout << "The existing second level centroid size is " << this->quantizers.size() * this->quantizers[0]->xb.size() / dimension << std::endl;
        std::vector<idx_t> assigned_labels(n);
        std::vector<idx_t> assigned_sub_labels(n);
        assign(n, x, assigned_labels.data(), assigned_sub_labels.data());

        std::vector<float> residuals (n * dimension);
        compute_residuals(n, x, residuals.data(), assigned_labels.data(), assigned_sub_labels.data());

        std::cout << "Training residual PQ with parameter setting: M: " << pq->M << "Number of centroids: " << pq->ksub << std::endl;
        this->pq->verbose = true;
        this->pq->train(n, residuals.data());
    }


    void BS_LIB_VQ_VQ::reconstruct(size_t n, float * x, const float * decoded_residuals, const idx_t * idxs, const idx_t * sub_idxs){
        for (size_t i=  0; i < n; i++){
            const float * centroid = this->quantizers[idxs[i]]->xb.data() + sub_idxs[i] * dimension;
            faiss::fvec_madd(dimension, decoded_residuals + i*dimension, 1.0, centroid, x + i*dimension);
        }
    }


    void BS_LIB_VQ_VQ::add_batch(size_t n, const float * x, const idx_t * origin_ids, const idx_t * idxs, const idx_t * sub_idxs){
        
        //Computing the residuals
        std::vector<float> residuals(n * dimension);
        compute_residuals(n, x, residuals.data(), idxs, sub_idxs);

        //Encoding the residuals
        std::vector<uint8_t> residual_codes(n * code_size);
        this->pq->compute_codes(residuals.data(), residual_codes.data(), n);

        //Decoding the residuals
        std::vector<float> decoded_residuals(n * dimension);
        this->pq->decode(residual_codes.data(), decoded_residuals.data(), n);

        //Reconstructing the vector
        std::vector<float> reconstructed_x(n * dimension);
        reconstruct(dimension, reconstructed_x.data(), decoded_residuals.data(), idxs, sub_idxs);

        //Computing the norm of the vector
        std::vector<float> norms(n);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), dimension, n);

        for (size_t i = 0; i < n; i++){
            base_codes[idxs[i]][sub_idxs[i]].push_back(residual_codes[i]);
            base_norms[idxs[i]][sub_idxs[i]].push_back(norms[n]);
            ids[idxs[i]][sub_idxs[i]].push_back(origin_ids[i]);
        }
    }

    void BS_LIB_VQ_VQ::write(const char * path_index){
        std::ofstream output (path_index, std::ios::binary);

        write_variable(output, dimension);
        write_variable(output, nc1);
        write_variable(output, nc2);

        for (size_t i = 0; i < nc1; i++)
            for (size_t j = 0; j < nc2; j++)
                write_vector(output, ids[i][j]);
        
        for (size_t i = 0; i < nc1; i++)
            for (size_t j = 0; j < nc2; j++)
                write_vector(output, base_codes[i][j]);

        for (size_t i = 0; i < nc1; i++)
            for (size_t j = 0; j < nc2; j++)
                write_vector(output, base_norms[i][j]);

        for (size_t i = 0; i < nc1; i++)
            write_vector(output, centroid_norms[i]);
    }

    void BS_LIB_VQ_VQ::read(const char * path_index){
        std::ifstream input (path_index, std::ios::binary);

        read_variable<size_t>(input, dimension);
        read_variable<size_t> (input, nc1);
        read_variable<size_t> (input, nc2);

        for(size_t i = 0; i < nc1; i++)
            for (size_t j = 0; j < nc2; j++)
                read_vector(input, ids[i][j]);

        for (size_t i = 0; i < nc1; i++)
            for (size_t j = 0; j < nc2; j++)
                read_vector(input, base_codes[i][j]);
        
        for(size_t i = 0; i < nc1; i++)
            for (size_t j = 0; j < nc2; j++)
                read_vector(input, base_norms[i][j]);
        
        for (size_t i = 0; i < nc1; i++)
            read_vector(input, centroid_norms[i]);
    }


    void BS_LIB_VQ_VQ::search(size_t nq, size_t k, const float * x, float * dists, idx_t * labels){
        std::vector<float> centroid_dis (nq * nprobe1);
        std::vector<idx_t> centroid_idxs(nq * nprobe1);

        this->quantizer->search(nq, x, nprobe1, centroid_dis.data(), centroid_idxs.data());
        size_t visited_vectors = 0;

#pragma omp parallel for
        for (size_t query_id = 0; query_id < nq; query_id++){
            const float * query = x + query_id * dimension;
            this->pq->compute_inner_prod_table(query, precomputed_table.data());
            
            float query_dis[k];
            idx_t query_labels[k];
            faiss::maxheap_heapify(k, query_dis, query_labels);

            for (size_t i = 0; i < nprobe1; i++){
                if (visited_vectors > max_vectors)
                    break;
                std::vector<float> sub_centroid_dis(nprobe2);
                std::vector<idx_t> sub_centroid_idx(nprobe2);

                const idx_t centroid_idx1 = centroid_idxs[query_id * nprobe1 + i];
                this->quantizers[centroid_idx1]->search(1, query, nprobe2, sub_centroid_dis.data(), sub_centroid_idx.data());
                
                for (size_t j = 0; j < nprobe2; j++){
                    const idx_t centroid_idx2 = sub_centroid_idx[j];

                    size_t group_size = base_codes[centroid_idx1][centroid_idx2].size();
                    assert(group_size == base_norms[centroid_idx1][centroid_idx2].size());
                    assert(group_size == ids[centroid_idx1][centroid_idx2].size());
                    if (group_size == 0)
                        continue;
                    
                    const uint8_t * code = base_codes[centroid_idx1][centroid_idx2].data();
                    const idx_t * origin_id = ids[centroid_idx1][centroid_idx2].data();
                    const float term1 = sub_centroid_dis[j] - centroid_norms[centroid_idx1][centroid_idx2];
                    
                    for (size_t group_id = 0; group_id < group_size; group_id++){
                        const float term2 = base_norms[centroid_idx1][centroid_idx2][group_id];
                        
                        const float term3 = 2 * pq_L2sqr(code + group_id * code_size);

                        const float dis = term1 + term2 - term3;

                        if (dis < query_dis[0]){
                            faiss::maxheap_pop(k, query_dis, query_labels);
                            faiss::maxheap_push(k, query_dis, query_labels, dis, origin_id[group_id]);
                        }
                    }

                    visited_vectors += group_size;
                    if (visited_vectors > max_vectors)
                        break;

                }
            }
            for (size_t result_id = 0; result_id < k; result_id++){
                dists[query_id * k + result_id] = query_dis[result_id];
                labels[query_id * k + result_id] = query_labels[result_id];
            }
        }
    }


    float BS_LIB_VQ_VQ::pq_L2sqr(const uint8_t * code){
        float result = 0.0;
        const size_t dim = code_size;
        for (size_t i = 0; i < dim; i++){
            result += precomputed_table[pq->ksub * i + code[i]];
        }
        return result;
    }

    void BS_LIB_VQ_VQ::compute_centroid_norms(){
        for (size_t i = 0; i < nc1; i++){
            for (size_t j = 0; j < nc2; j++){
                const float * centroid = this->quantizers[i]->xb.data() + j * dimension;
                const float norm = faiss::fvec_norm_L2sqr(centroid, dimension);
                centroid_norms[i].push_back(norm);
            }
        }
    }

}