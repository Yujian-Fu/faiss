#include "LQ_quantizer.h"
#define Not_Found -1.0

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
        std::vector<std::vector<float>> train_set(this->nc_upper);

        for(size_t i = 0; i < train_set_size; i++){
            idx_t idx = train_data_idxs[i];
            assert(idx <= this->nc_upper);
            for (size_t j = 0; j < this->dimension; j++){

                train_set[idx].push_back(train_data[i * dimension + j]);
            }

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
            std::cout << "Finding the centroid idxs for train vectors for futher quantizer construction " << std::endl;

            for (size_t i = 0; i < this->nc_upper; i++){
                std::vector<float> final_centroids(this->nc_per_group * dimension);
                idx_t base_idx = CentroidDistributionMap[i];
                faiss::IndexFlatL2 group_quantizer(dimension);
                for (size_t j = 0; j < this->nc_per_group; j ++){
                    compute_final_centroid(j + base_idx, final_centroids.data() + j * dimension);
                }
                group_quantizer.add(this->nc_per_group, final_centroids.data());
                size_t group_size = train_set[i].size() / dimension;
                std::vector<float> centroid_distances(group_size);
                std::vector<faiss::Index::idx_t> centroid_idxs(group_size);
                group_quantizer.search(group_size, train_set[i].data(), 1, centroid_distances.data(), centroid_idxs.data());
                for (size_t j = 0; j < group_size; j++){
                    train_data_idxs[i * nc_per_group + j] = centroid_idxs[j] + base_idx;
                }
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
            std::cout << "The centroid idx and nn centroid idx is " << j << " " << group_label << " " << nn_centroid_idx << std::endl;
            const float * nn_centroid = this->upper_centroids.data() + nn_centroid_idx * dimension;
            const float * centroid = this->upper_centroids.data() + j * dimension;
            faiss::fvec_madd(dimension, nn_centroid, -1.0, centroid, centroid_vector.data());
            faiss::fvec_madd(dimension, centroid, alpha, centroid_vector.data(), sub_centroid);
    }


    void LQ_quantizer::compute_residual_group_id(size_t n, const idx_t * labels, const float * x, float * residuals){
#pragma omp parallel for        
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            compute_final_centroid(labels[i], final_centroid.data());
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, final_centroid.data(), residuals + i * dimension);
        }
    }

    void LQ_quantizer::recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            compute_final_centroid(labels[i], final_centroid.data());
            faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, final_centroid.data(), x + i * dimension);
        }
    }

    float LQ_quantizer::search_in_map(std::map<idx_t, float> dist_map, idx_t key){
        std::map<idx_t, float>::iterator iter;
        iter = dist_map.find(key);
        if(iter!=dist_map.end())
        {
            return iter->second;
        }
        return Not_Found;
    }

    void LQ_quantizer::search_in_group(size_t n, const float * queries, const std::vector<std::map<idx_t, float>> queries_upper_centroid_dists, const idx_t * group_idxs, float * result_dists){
        std::vector<std::vector<idx_t>> query_sequence_set(this->nc_upper);

        for (size_t i = 0; i < n; i++){
            idx_t idx = group_idxs[i];
            query_sequence_set[idx].push_back(i);
        }

        for (size_t i = 0; i < this->nc_upper; i++){
            if (query_sequence_set[i].size() == 0)
                continue;
            else{

                std::vector<std::vector<float>> sub_centroids(this->nc_per_group);
                idx_t base_idx = CentroidDistributionMap[i];
                float alpha = this->alphas[i];

                for (size_t j = 0; j < query_sequence_set[i].size(); j++){
                    idx_t sequence_id = query_sequence_set[i][j];

                    std::vector<float> query_sub_centroids_dists(this->nc_per_group);
                    std::cout << "Computing distance to all sub centroids " << std::endl;
                    for (size_t m = 0; m < this->nc_per_group; m++){
                        idx_t nn_idx = this->nn_centroid_idxs[i][m];
                        float query_nn_dist = search_in_map(queries_upper_centroid_dists[sequence_id], nn_idx);
                        float easy_dist = 0;
                        if (query_nn_dist != Not_Found){
                            std::cout << "Computing easy distance" << std::endl;
                            idx_t group_idx = group_idxs[sequence_id];
                            float query_group_dist = search_in_map(queries_upper_centroid_dists[sequence_id], group_idx);
                            assert (query_group_dist != Not_Found);
                            float group_nn_dist = this->nn_centroid_dists[group_idx][nn_idx];

                            easy_dist = sqrt(alpha*(alpha-1)*group_nn_dist*group_nn_dist+(1-alpha)*query_group_dist*query_group_dist+alpha*query_nn_dist);
                        }
                        //else{
                        std::cout << "Computing normal distance from sequence id to label " << sequence_id << " " << base_idx+m << std::endl;
                        if (sub_centroids[m].size() == 0){
                            idx_t label = base_idx + m;
                            compute_final_centroid(label, sub_centroids[m].data());
                        }
                        
                        const float * query = queries + sequence_id * dimension;
                        std::vector<float> query_sub_centroid_vector(dimension);
                        faiss::fvec_madd(dimension, sub_centroids[m].data(), -1.0, query, query_sub_centroid_vector.data());
                        query_sub_centroids_dists[m] = faiss::fvec_norm_L2sqr(query_sub_centroid_vector.data(), dimension);

                        if (query_nn_dist != Not_Found){
                            std::cout << easy_dist << "_" << query_sub_centroids_dists[m] << std::endl;
                        }
                        //}
                    }

                    std::cout << "Adding distance to the result distances " << std::endl;
                    for (size_t m = 0; m < this->nc_per_group; m++){
                        result_dists[sequence_id * this->nc_per_group + m] = query_sub_centroids_dists[m];
                    }
                }
            }
        }
    }

    void LQ_quantizer::compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * labels){
        faiss::IndexFlatL2 all_quantizer(dimension);
        std::vector<float> final_centroids(this->nc * dimension);

        for (size_t i = 0; i < this->nc; i++){
            compute_final_centroid(i, final_centroids.data() + i * dimension);
        }
        all_quantizer.add(this->nc, final_centroids.data());

        std::cout << "searching the idxs for centroids nearest neighbors " << std::endl;
        //Find the centroid idxs for train vectors
        std::vector<float> search_nn_dists(this->nc * (k+1));
        std::vector<faiss::Index::idx_t> search_nn_idxs(this->nc * (k+1));

        for (size_t i = 0; i < this->nc * this->dimension; i ++){
            nn_centroids[i] = all_quantizer.xb[i];
        }
        all_quantizer.search(this->nc, all_quantizer.xb.data(), k+1, search_nn_dists.data(), search_nn_idxs.data());
        for (size_t i = 0; i < this->nc; i++){
            for (size_t j = 0; j < k; j++){
                labels[i * k + j] = search_nn_idxs[i * (k + 1) + j + 1 ];
                nn_dists[i * k + j] = search_nn_dists[i * (k + 1) + j + 1];
            }
        }
    }









}