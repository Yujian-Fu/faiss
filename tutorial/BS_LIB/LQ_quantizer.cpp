#include "LQ_quantizer.h"

namespace bslib{
    LQ_quantizer::LQ_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group, const float * upper_centroids,
                               const idx_t * upper_centroid_idxs, const float * upper_centroid_dists):
        Base_quantizer(dimension, nc_upper, nc_per_group){
            this->alphas.resize(nc_upper);
            this->upper_centroids.resize(dimension * nc_upper);
            this->nn_centroid_idxs.resize(nc_upper);
            this->nn_centroid_dists.resize(nc_upper);

            for (size_t i = 0; i < dimension * nc_upper; i++){
                this->upper_centroids[i] = upper_centroids[i];
            }

            for (size_t i = 0; i < nc_upper; i++){
                for (size_t j = 0; j < nc_per_group; j++){
                    this->nn_centroid_idxs[i].push_back(upper_centroid_idxs[i * nc_per_group + j]);
                    this->nn_centroid_dists[i].push_back(upper_centroid_dists[i * nc_per_group + j]);
                }
            }
        }


    float LQ_quantizer::compute_alpha(const float * centroid_vectors, const float * points, const float * centroid,
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

            for (size_t subc = 0; subc < nc_per_group; subc++){
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


    void LQ_quantizer::build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_idxs, bool update_idxs){
        std::vector<std::vector<float>> train_set;
        train_set.resize(nc_upper);

        for(size_t i = 0; i < train_set_size; i++){
            idx_t idx = train_data_idxs[i];
            for (size_t j = 0; j < this->dimension; j++)
            train_set[idx].push_back(train_data[i * dimension + j]);
        }

        std::cout << "Computing alphas for lq_quantizer with upper centroids: " << this->upper_centroids.size() << " nc_per_group: " << this->nc_per_group << std::endl;
        std::cout << "The size of upper_centroids: " << this->upper_centroids.size() / this->dimension << std::endl;
        std::cout << "The size if nn_centroid_idxs: " << nn_centroid_idxs.size() << std::endl;

        for (size_t i = 0; i < nc_upper; i++){
            std::vector<float> centroid_vectors(nc_per_group * dimension);
            const float * centroid = this->upper_centroids.data() + i * dimension;
            for (size_t j = 0; j < nc_per_group; j++){
                const float * nn_centroid = this->upper_centroids.data() + this->nn_centroid_idxs[i][j] * dimension;
                faiss::fvec_madd(dimension, nn_centroid, -1.0, centroid, centroid_vectors.data()+ j * dimension);
            }
            size_t group_size = train_set[i].size() / this->dimension;
            if ( i % 100 == 0)
            std::cout << "Computing alpha for [ " << i << " / " << nc_upper << " ]" << std::endl;
            this->alphas[i] = compute_alpha(centroid_vectors.data(), train_set[i].data(), centroid, nn_centroid_dists[i].data(), group_size);
        }
        
        std::cout << "finished computing centoids" <<std::endl;

        if (update_idxs){
            std::cout << "Initializing all_quantizer for lq-quantizer" <<std::endl;
            this->all_quantizer = faiss::IndexFlatL2(dimension);
            std::vector<float> final_centroids(this->nc * dimension);
            for (size_t i = 0; i < this->nc; i++){
                compute_final_centroid(i, final_centroids.data() + i * dimension);
            }
            all_quantizer.add(this->nc, final_centroids.data());

            std::cout << "searching the idxs for train vectors " << std::endl;
            //Find the centroid idxs for train vectors
            std::vector<float> centroid_distances(train_set_size);
            std::vector<faiss::Index::idx_t> centroid_idxs(train_set_size);
            all_quantizer.search(train_set_size, train_data, 1, centroid_distances.data(), centroid_idxs.data());
            for (size_t i = 0; i < train_set_size; i++){
                train_data_idxs[i] = centroid_idxs[i];
            }
        }
    }

    void LQ_quantizer::compute_final_centroid(idx_t label, float * sub_centroid){
            size_t j = size_t(label / nc_per_group);
            /*
            for (j = 0; j < nc_upper; j++){
                if (CentroidDistributionMap[j + 1] >= label){
                    break;
                }
            }
            */
            size_t group_label = label - CentroidDistributionMap[j];
            size_t nn_centroid_idx = nn_centroid_idxs[j][group_label];
            float alpha = alphas[j];
            std::vector<float> centroid_vector(dimension);
            const float * nn_centroid = this->upper_centroids.data() + nn_centroid_idx * dimension;
            const float * centroid = this->upper_centroids.data() + j * dimension;
            faiss::fvec_madd(dimension, nn_centroid, -1.0, centroid, centroid_vector.data());
            faiss::fvec_madd(dimension, centroid, alpha, centroid_vector.data(), sub_centroid);
    }


    void LQ_quantizer::compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals){
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            if (i == 999999){
                std::cout << labels[i] << " " << i << " " << std::endl;;
                for (size_t j = 0; j < dimension; j++){
                    std::cout << x[i * dimension + j] << " ";
                }
                std::cout << std::endl;
            }

            compute_final_centroid(labels[i], final_centroid.data());
            
            if (i == 999999){
                std::cout << "Finished compute one final centroid " << std::endl;
                std::cout << labels[i] << " " << i << " " << std::endl;;
                for (size_t j = 0; j < dimension; j++){
                    std::cout << x[i * dimension + j] << " ";
                }
                std::cout << std::endl;
            }
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, final_centroid.data(), residuals + i * dimension);
        }
        std::cout << "Finished computing lq residuals" << std::endl;
    }

    void LQ_quantizer::recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x){
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            compute_final_centroid(labels[i], final_centroid.data());
            faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, final_centroid.data(), x + i * dimension);
        }
    }

    void LQ_quantizer::search_in_group(size_t n, size_t k, float * dists, idx_t * labels, const idx_t * group_id, const float * query_centroid_dists){
        std::vector<float> query_dists(k);
        std::vector<faiss::Index::idx_t> query_labels(k);
        faiss::maxheap_heapify(k, query_dists.data(), query_labels.data());

        for (size_t i = 0; i < n; i++){
            float alpha = this->alphas[group_id[i]];
            // The distance equation is dis = sqrt((1-alpha)* dis_q_c^2 + alpha(1-alpha)*dis_c_n^2+alpha*dis_q_n^2)
            for (size_t j = 0; j < nc_per_group; j++){
                idx_t nn_centroid_idx = nn_centroid_idxs[group_id[i]][j];
                const float term1 = (1 - alpha) * query_centroid_dists[group_id[i]] * query_centroid_dists[group_id[i]];
                const float term2 = alpha * (1 - alpha) * nn_centroid_dists[group_id[i]][nn_centroid_idx] * nn_centroid_dists[group_id[i]][nn_centroid_idx];
                const float term3 = alpha * query_centroid_dists[nn_centroid_idx] * query_centroid_dists[nn_centroid_idx];
                float dist = sqrt(term1 + term2 + term3);
                if (dist < query_dists[0]){
                    faiss::maxheap_pop(k, query_dists.data(), query_labels.data());
                    faiss::maxheap_push(k, query_dists.data(), query_labels.data(), dist, j);
                }
            }
            size_t base_idx = CentroidDistributionMap[group_id[i]];
            for (size_t j = 0; j < k; j++){
                dists[i * k + j] = query_dists[k-1-j];
                labels[i * k + j] = base_idx + query_labels[k-1-j];
            }
        }
        for (size_t i = 0; i < 100; i++){
            std::cout << labels[i] << " ";
        }
        std::cout << std::endl;
    }

    void LQ_quantizer::compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * labels){
        std::vector<faiss::Index::idx_t> search_nn_idxs((k + 1) * this->nc);
        std::vector<float> search_nn_dists((k + 1) * this->nc);

        for (size_t i = 0; i < this->nc * this->dimension; i ++){
            nn_centroids[i] = this->all_quantizer.xb[i];
        }

        this->all_quantizer.search(this->nc, all_quantizer.xb.data(), k+1, search_nn_dists.data(), search_nn_idxs.data());
        for (size_t i = 0; i < this->nc; i++){
            for (size_t j = 0; j < k; j++){
                labels[i * k + j] = search_nn_idxs[i * (k + 1) + j + 1 ];
                nn_dists[i * k + j] = search_nn_dists[i * (k + 1) + j + 1];
            }
        }

    }

    






}