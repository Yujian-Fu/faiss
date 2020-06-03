#include "index_VQ_LQ.h"

namespace bslib_VQ_LQ{
    BS_LIB_VQ_LQ::BS_LIB_VQ_LQ(size_t dimension, size_t ncentroids, size_t bytes_per_code,
                                                   size_t nbits_per_idx, size_t nsubcentroids, bool use_quantized_distance):
           dimension(dimension), nc(ncentroids), nsubc(nsubcentroids), use_quantized_distance(use_quantized_distance), pq(nullptr), norm_pq(nullptr){
               
        this->pq = new faiss::ProductQuantizer(dimension, bytes_per_code, nbits_per_idx);
        this->norm_pq = new faiss::ProductQuantizer(1, 1, nbits_per_idx);

        code_size = pq->code_size;


        norms.resize(65536);
        precomputed_table.resize(pq->ksub * pq->M);
        base_codes.resize(nc);
        base_norm_codes.resize(nc);
        ids.resize(nc);
        centroid_norms.resize(nc);
        alphas.resize(nc);
        nn_centroid_idxs.resize(nc);
        subgroup_sizes.resize(nc);
        query_centroid_dists.resize(nc);

        std::fill(query_centroid_dists.begin(), query_centroid_dists.end(), 0);
        inter_centroid_dists.resize(nc);
    }

    void BS_LIB_VQ_LQ::assign(size_t n, const float * x, idx_t * labels, size_t k){
        this->quantizer->assign(n, x, labels, k);
    }


    float BS_LIB_VQ_LQ::compute_alpha(const float * centroid_vectors, const float * points, const float * centroid,
                                      const float * centroid_vector_norms_L2sqr, size_t group_size){
        float group_numerator = 0.0;
        float group_denominator = 0.0;

        std::vector<float> point_vectors(group_size * dimension);
        for (size_t i = 0; i < group_size; i++)
            faiss::fvec_madd(dimension, points + i * dimension, -1.0, centroid, point_vectors.data() + i * dimension);
        
        for (size_t i = 0; i < group_size; i++){
            const float * point_vector = point_vectors.data() + i * dimension;
            const float * point = points + i * dimension;

            std::priority_queue<std::pair<float, std::pair<float, float>>> maxheap;

            for (size_t subc = 0; subc < nsubc; subc++){
                const float * centroid_vector = centroid_vectors + subc * dimension;
                float numerator = faiss::fvec_inner_product(centroid_vector, point_vector, dimension);
                numerator = (numerator > 0) ? numerator : 0.0;

                const float denominator = centroid_vector_norms_L2sqr[subc];
                const float alpha = numerator / denominator;

                std::vector<float> subcentroid(dimension);
                faiss::fvec_madd(dimension, centroid, alpha, centroid_vector, subcentroid.data());

                const float dist = faiss::fvec_L2sqr(point, subcentroid.data(), dimension);
                maxheap.emplace(-dist, std::make_pair(numerator, denominator));
            }

            group_numerator += maxheap.top().second.first;
            group_denominator += maxheap.top().second.second;
        }
        return (group_denominator > 0) ? group_numerator / group_denominator : 0.0;
    }


    void BS_LIB_VQ_LQ::build_quantizer(const char * centroid_path){
        this->quantizer = new faiss::IndexFlatL2(dimension);
        std::cout << "Building quantizer " << std::endl;
        std::ifstream centroid_input(centroid_path, std::ios::binary);
        std::vector<float> centroids(dimension * nc);
        readXvec<float> (centroid_input, centroids.data(), dimension, nc, true);
        faiss::IndexFlatL2 quantizer (dimension);
        this->quantizer->add(nc, centroids.data());
    }

    void BS_LIB_VQ_LQ::compute_subcentroid_idxs(idx_t * subcentroid_idxs, const float * subcentroids,
                                                const float * x, size_t group_size){
        for (size_t i = 0; i < group_size; i++){
            float min_dist = 0.0;
            idx_t min_idx = -1;
            for (size_t subc = 0; subc < nsubc; subc++){
                const float * subcentroid = subcentroids + subc * dimension;
                float dist = faiss::fvec_L2sqr(subcentroid, x + i * dimension, dimension);
                if (min_idx == -1 || dist < min_dist){
                    min_dist = dist;
                    min_idx = subc;
                }
            }
            subcentroid_idxs[i] = min_idx;
        }
    }

    void BS_LIB_VQ_LQ::compute_residuals(size_t n, const float * x, float * residuals, const float * subcentroids, const idx_t * keys){
#pragma omp parallel for        
        for (size_t i = 0; i < n; i++){
            const float * subcentroid = subcentroids + keys[i] * dimension;
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, subcentroid, residuals + i * dimension);
        }
    }


    void BS_LIB_VQ_LQ::train_pq(size_t n, const float * x, bool train_pq, bool train_norm_pq){
        std::vector<float> train_subcentroids(n * dimension);    
        std::vector<float> train_residuals(n * dimension);

        std::vector<idx_t> assigned_ids(n);
        assign(n, x, assigned_ids.data());

        std::cout << "The assigned data examples are: " << std::endl;
        for (int i = 0; i < 10; i++)
        {
            std::cout << assigned_ids[i] << " "; 
        }

        std::unordered_map<idx_t, std::vector<float>> group_map;

        for (size_t i = 0; i < n; i++){
            const idx_t key = assigned_ids[i];
            for (size_t j = 0; j < dimension; j++)
                group_map[key].push_back(x[i * dimension + j]);
        }

        std::cout << "Training Residual PQ codebook " << std::endl;

        for (auto group : group_map){
            const idx_t centroid_idx = group.first;
            const float * centroid = this->quantizer->xb.data() + centroid_idx * dimension;
            const std::vector<float> data = group.second;
            const size_t group_size = data.size() / dimension;

            std::vector<idx_t> nn_centroid_idxs(nsubc);
            std::vector<float> centroid_vector_norms(nsubc);
            std::vector<idx_t> searching_idxs(nsubc + 1);
            std::vector<float> searching_dis(nsubc + 1);
            this->quantizer->search(1, this->quantizer->xb.data() + centroid_idx * dimension, nsubc+1, searching_dis.data(), searching_idxs.data());

            assert(searching_idxs[0] == centroid_idx);
            for(size_t i = 0; i < nsubc; i++){
                nn_centroid_idxs[i] = searching_idxs[i + 1];
                centroid_vector_norms[i] = searching_dis[i + 1];
            }
            
            //Compute the centroid-neighbor_centroid and centroid-group_point vectors
            std::vector<float> centroid_vectors(nsubc * dimension);
            for (size_t subc = 0; subc < nsubc; subc ++){
                const float * nn_centroid = this->quantizer->xb.data() + nn_centroid_idxs[subc] * dimension;
                faiss::fvec_madd(dimension, nn_centroid, -1.0, centroid, centroid_vectors.data()+subc * dimension);
            }

            const float alpha = compute_alpha(centroid_vectors.data(), data.data(), centroid, 
                                              centroid_vector_norms.data(), group_size);
            
            std::vector<float> subcentroids(nsubc * dimension);

            for (size_t subc = 0; subc < nsubc; subc++)
                faiss::fvec_madd(dimension, centroid, alpha, centroid_vectors.data() + subc * dimension, subcentroids.data() + subc * dimension);
            
            std::vector<idx_t> subcentroid_idxs(group_size);
            compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data.data(), group_size);

            //Compute residuals
            std::vector<float> residuals(group_size * dimension);
            compute_residuals(group_size, data.data(), residuals.data(), subcentroids.data(), subcentroid_idxs.data());

            for(size_t i = 0; i < group_size; i++){
                const idx_t subcentroid_idx = subcentroid_idxs[i];
                for (size_t j = 0; j < dimension; j++){
                    train_subcentroids.push_back(subcentroids[subcentroid_idx * dimension + j]);
                    train_residuals.push_back(residuals[i * dimension + j]);
                }
            }
        }

        if (train_pq){
            std::cout << "Training PQ codebook with parameter setting: M: " << pq->M << " ksub: " << pq->ksub << " and trainset size:" << train_residuals.size();
            this->pq->verbose = true;
            this->pq->train(n, train_residuals.data());
        }

        if (train_norm_pq){
        std::cout << "Training Norm PQ codebook " << std::endl;
        std::vector<float> train_norms;
        const float * residuals = train_residuals.data();
        const float * subcentroids = train_subcentroids.data();

        for (auto group : group_map){
            const std::vector<float> data = group.second;
            const size_t group_size = data.size() / dimension;

            //Compute the codes
            std::vector<uint8_t> residual_codes(group_size * code_size);
            this->pq->compute_codes(residuals, residual_codes.data(), group_size);

            //Decode codes
            std::vector<float> decoded_residuals(group_size * dimension);
            pq->decode(residual_codes.data(), decoded_residuals.data(), group_size);

            //Reconstruct the base vector data
            std::vector<float> reconstruct_x (group_size * dimension);
            for (size_t i = 0; i < group_size; i++)
                faiss::fvec_madd(dimension, decoded_residuals.data() + i * dimension, 1.0, subcentroids + i * dimension, reconstruct_x.data() + i *dimension);

            //Compute norms
            std::vector<float> group_norms(group_size);
            faiss::fvec_norms_L2sqr(group_norms.data(), reconstruct_x.data(), dimension, group_size);

            //Add norm in each group to the final group
            for (size_t i = 0; i << group_size; i++)
                train_norms.push_back(group_norms[i]);
            
            residuals += group_size * dimension;
            subcentroids += group_size * dimension;
        }
        std::cout << "Training Norm PQ codebook with parameter setting: M: " << norm_pq->M << " ksub: " << norm_pq->ksub << " and trainset size:" << train_norms.size();
        norm_pq->verbose = true;
        norm_pq->train(n, train_norms.data());
        }
    }

    void BS_LIB_VQ_LQ::reconstruct(size_t n, float * x, const float * decoded_residuals, const float * subcentroids, const idx_t * keys){
#pragma omp parallel for        
        for (size_t i = 0; i < n; i++){
            const float * subcentroid = subcentroids  + dimension * keys[i];
            faiss::fvec_madd(dimension, decoded_residuals + i * dimension, 1.0, subcentroid, x + i * dimension);
        }
    }

    void BS_LIB_VQ_LQ::add_group(size_t centroid_idx, size_t group_size, const float * data, const idx_t * idxs, bool use_quantized_dis){
        const float * centroid = quantizer->xb.data() + centroid_idx * dimension;
        std::vector<idx_t> searching_idxs(nsubc + 1);
        std::vector<float> searching_distance(nsubc+1);
        this->quantizer->search(1, centroid, nsubc+1, searching_distance.data(), searching_idxs.data());

        std::vector<float> centroid_vector_norms_L2sqr(nsubc);
        nn_centroid_idxs[centroid_idx].resize(nsubc);

        for (size_t i = 0; i < nsubc; i++){
            nn_centroid_idxs[centroid_idx][i] = searching_idxs[i+1];
            centroid_vector_norms_L2sqr[i] = searching_distance[i+1];
        }
        if (group_size == 0)
            return;

        const float * centroid_vector_norms = centroid_vector_norms_L2sqr.data();
        const idx_t * nn_centroids = nn_centroid_idxs[centroid_idx].data();

        //Compute centroid-neighbor_centroid and centroid-group_poin vectors
        std::vector<float> centroid_vectors(nsubc * dimension);
        for (size_t subc = 0; subc < nsubc; subc++){
            const float * neighbor_centroid = this->quantizer->xb.data()+nn_centroids[subc] * dimension;
            faiss::fvec_madd(dimension, neighbor_centroid, -1.0, centroid, centroid_vectors.data() + subc * dimension);
        }

        //Compute alpha for group vectors
        alphas[centroid_idx] = compute_alpha(centroid_vectors.data(), data, centroid, 
                                             centroid_vector_norms, group_size);
        
        //Compute final subcentroids
        std::vector<float> subcentroids(nsubc * dimension);
        for (size_t subc = 0; subc < nsubc; subc++){
            const float * centroid_vector = centroid_vectors.data() + subc * dimension;
            float * subcentroid = subcentroids.data() + subc * dimension;
            faiss::fvec_madd(dimension, centroid, alphas[centroid_idx], centroid_vector, subcentroid);
        }

        //Compute the subcentroid idx for vectors
        std::vector<idx_t> subcentroid_idxs (group_size);
        compute_subcentroid_idxs(subcentroid_idxs.data(), subcentroids.data(), data, group_size);

        //Compute the residuals
        std::vector<float> residuals(group_size * dimension);
        compute_residuals(group_size, data, residuals.data(), subcentroids.data(), subcentroid_idxs.data());

        //Encode the base residuals
        std::vector<uint8_t> residual_codes(group_size * code_size);
        this->pq->compute_codes(residuals.data(), residual_codes.data(), group_size);

        //Decode the codes
        std::vector<float> decoded_residuals(group_size * dimension);
        this->pq->decode(residual_codes.data(), decoded_residuals.data(), group_size);

        //Reconstruct base vectors
        std::vector<float> reconstructed_x(group_size * dimension);
        reconstruct(group_size, reconstructed_x.data(),decoded_residuals.data(), subcentroids.data(), subcentroid_idxs.data());
        
        //Compute norms
        std::vector<float> norms(group_size);
        faiss::fvec_norms_L2sqr(norms.data(), reconstructed_x.data(), dimension, group_size);

        if (use_quantized_dis){
            //Compute norm codes
            std::vector<uint8_t> xnorm_codes(group_size);
            this->norm_pq->compute_codes(norms.data(), xnorm_codes.data(), group_size);


            std::vector<std::vector<uint8_t>> construction_norm_codes(nsubc);
            for (size_t i = 0; i < group_size; i++){
                idx_t subcentroid_idx = subcentroid_idxs[i];
                construction_norm_codes[subcentroid_idx].push_back(xnorm_codes[i]);
            }
            for (size_t subc = 0; subc < nsubc; subc++){
                for(size_t i = 0; i < group_size; i++){
                    base_norm_codes[centroid_idx].push_back(construction_norm_codes[subc][i]);
                }
            }
        }
        else{
            std::vector<std::vector<float>> construction_norm(nsubc);
            for (size_t i = 0; i < group_size; i++){
                idx_t subcentroid_idx = subcentroid_idxs[i];
                construction_norm[subcentroid_idx].push_back(norms[i]);
            }
            for (size_t subc = 0; subc < nsubc; subc++){
                for(size_t i = 0; i < group_size; i++){
                    base_norms[centroid_idx].push_back(construction_norm[subc][i]);
                }
            }
        }

        //Assign the code to the index
        std::vector<std::vector<idx_t>> construction_ids(nsubc);
        std::vector<std::vector<uint8_t>> construction_codes(nsubc);
        for (size_t i = 0; i < group_size; i++){
            idx_t idx = idxs[i];
            idx_t subcentroid_idx = subcentroid_idxs[i];
            construction_ids[subcentroid_idx].push_back[idx];
            for (size_t j = 0 ; j < code_size; j++)
                construction_codes[subcentroid_idx].push_back(residual_codes[i *code_size + j]);
            
        }

        //Add codes to the index
        for (size_t subc =0; subc < nsubc; subc++){
            idx_t subgroup_size = construction_ids[subc].size();
            subgroup_sizes[centroid_idx].push_back(subgroup_size);

            for (size_t i = 0; i < subgroup_size; i++){
                ids[centroid_idx].push_back(construction_ids[subc][i]);
                for (size_t j = 0; j < code_size; j++)
                    base_codes[centroid_idx].push_back(construction_codes[subc][i * code_size + j]);
            }
        }
    }

    void BS_LIB_VQ_LQ::compute_centroid_norms(){
    for (size_t i = 0; i < nc; i++){
        const float * centroid = this->quantizer->xb.data()+ i * dimension;
        centroid_norms[i] = faiss::fvec_norm_L2sqr(centroid, dimension);
    }
    }

    void BS_LIB_VQ_LQ::compute_inter_centroid_dists(){
        for (size_t i = 0; i < nc; i++){
            const float * centroid = this->quantizer->xb.data() + i * dimension;
            inter_centroid_dists[i].resize(nsubc);
            for (size_t subc = 0; subc < nsubc; subc++){
                const idx_t nn_centroid_idx = nn_centroid_idxs[i][subc];
                const float * nn_centroid = this->quantizer->xb.data() + nn_centroid_idx * dimension;
                inter_centroid_dists[i][subc] = faiss::fvec_L2sqr(nn_centroid, centroid, dimension);
            }
        }
    }

    void BS_LIB_VQ_LQ::read(const char * path_index){
        std::ifstream input(path_index, std::ios::binary);

        read_variable(input, dimension);
        read_variable(input, nc);
        read_variable(input, nsubc);

        for (size_t i = 0; i < nc; i++)
            read_vector(input, ids[i]);
        
        for (size_t i = 0; i < nc; i++)
            read_vector(input, base_codes[i]);
        
        for (size_t i = 0; i < nc; i++)
            read_vector(input, base_norm_codes[i]);

        for (size_t i = 0; i < nc; i++)
            read_vector(input, nn_centroid_idxs[i]);

        for (size_t i = 0; i < nc; i++)
            read_vector(input, subgroup_sizes[i]);

        read_vector(input, alphas);

        read_vector(input, centroid_norms);

        for (size_t i = 0; i < nc; i++)
            read_vector(input, inter_centroid_dists[i]);
    }

    void BS_LIB_VQ_LQ::write(const char * path_index){
        std::ofstream output(path_index, std::ios::binary);

        write_variable(output, dimension);
        write_variable(output, nc);
        write_variable(output, nsubc);

        for (size_t i = 0; i < nc; i++)
            write_vector(output, ids[i]);
        
        for (size_t i = 0; i < nc; i++)
            write_vector(output, base_codes[i]);

        for (size_t i = 0; i < nc; i++)
            write_vector(output, base_norm_codes[i]);
        
        for (size_t i = 0; i < nc; i++)
            write_vector(output, nn_centroid_idxs[i]);

        for(size_t i = 0; i < nc; i++)
            write_vector(output, subgroup_sizes[i]);

        write_vector(output, alphas);

        write_vector(output, centroid_norms);

        for (size_t i = 0; i < nc; i++)
            write_vector(output, inter_centroid_dists[i]);
    }
    
}
