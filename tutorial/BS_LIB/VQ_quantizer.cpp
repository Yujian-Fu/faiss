#include "VQ_quantizer.h"

namespace bslib{
    VQ_quantizer::VQ_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group):
        Base_quantizer(dimension, nc_upper, nc_per_group){}


    void VQ_quantizer::build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_idxs, bool update_idxs){
        std::cout << "Adding " << train_set_size << " train set data into " << nc_upper << " group " << std::endl;
        std::vector<std::vector<float>> train_set(this->nc_upper);

        for (size_t i = 0; i < train_set_size; i++){
            idx_t idx = train_data_idxs[i];
            for (size_t j = 0; j < dimension; j++)
                train_set[idx].push_back(train_data[i*dimension + j]);
        }

        this->quantizers.resize(this->nc_upper);
        std::cout << "Building group quantizers for vq_quantizer " << std::endl;
#pragma omp parallel for
        for (size_t i = 0; i < nc_upper; i++){
            std::vector<float> centroids(dimension * nc_per_group);
            size_t nt_sub = train_set[i].size() / this->dimension;
            faiss::kmeans_clustering(dimension, nt_sub, nc_per_group, train_set[i].data(), centroids.data());
            faiss::IndexFlatL2 centroid_quantizer(dimension);
            centroid_quantizer.add(nc_per_group, centroids.data());
            this->quantizers[i] = centroid_quantizer;
        }

        std::cout << "finished computing centoids" <<std::endl;
        if (update_idxs){
            std::cout << "Finding the centroid idxs for train vectors for futher quantizer construction " << std::endl;
            //Find the centroid idxs for train vectors
#pragma omp parallel for
            for (size_t i = 0; i < this->nc_upper; i++){
                size_t base_idx = CentroidDistributionMap[i];
                size_t group_size = train_set[i].size() / dimension;
                std::vector<float> centroid_distances(group_size);
                std::vector<faiss::Index::idx_t> centroid_idxs(group_size);
                this->quantizers[i].search(group_size, train_set[i].data(), 1, centroid_distances.data(), centroid_idxs.data());
                for (size_t j = 0; j < group_size; j++){
                    train_data_idxs[i * nc_per_group + j] = centroid_idxs[j] + base_idx;
                }
            }
        }
    }

    void VQ_quantizer::search_in_group(size_t n, const float * queries, const idx_t * group_idxs, float * result_dists){
        //clock_t starttime = clock();
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            idx_t idx = group_idxs[i];
            for (size_t j = 0; j < this->nc_per_group; j++){
                std::vector<float> query_sub_centroid_vector(dimension);
                const float * sub_centroid = this->quantizers[idx].xb.data() + j * this->dimension;
                const float * query = queries + i * dimension;
                faiss::fvec_madd(dimension, sub_centroid, -1.0, query, query_sub_centroid_vector.data());
                result_dists[i * this->nc_per_group + j] = faiss::fvec_norm_L2sqr(query_sub_centroid_vector.data(), dimension);
            }
        }
        //clock_t endtime = clock();
        //std::cout << "Search time in VQ " << float(endtime - starttime) / CLOCKS_PER_SEC << std::endl;
    }

    void VQ_quantizer::compute_final_centroid(idx_t label, float * final_centroid){
        size_t j = size_t (label / nc_per_group);
        size_t group_label = label - CentroidDistributionMap[j];
        for (size_t i = 0; i < dimension; i++){
            std::cout << i << " ";
            final_centroid[i] = this->quantizers[group_label].xb[group_label * this->dimension + i];
        }
    }


    void VQ_quantizer::compute_residual_group_id(size_t n,  const idx_t * labels, const float * x, float * residuals){
        std::cout << "Computing VQ residual for train data with " << n << std::endl;

//#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::cout << "now computing " << i << " ";
        }
        exit(0);
    }

    void VQ_quantizer::recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x){
        std::cout << "Computing LQ residual for train data " << std::endl;
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            compute_final_centroid(labels[i], final_centroid.data());
            faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, final_centroid.data(), x + i * dimension);
        }
    }

    void VQ_quantizer::compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * labels){
        faiss::IndexFlatL2 all_quantizer(dimension);

        for (size_t i = 0; i < this->nc_upper; i++){
            all_quantizer.add(this->nc_per_group, this->quantizers[i].xb.data());
        }

        for (size_t i = 0; i < this->nc * this->dimension; i++){
            nn_centroids[i] = all_quantizer.xb[i];
        }

        std::vector<faiss::Index::idx_t>  search_nn_idxs(this->nc * (k + 1));
        std::vector<float> search_nn_dists ( this->nc * (k+1));
        all_quantizer.search(this->nc, all_quantizer.xb.data(), k+1, search_nn_dists.data(), search_nn_idxs.data());

        for (size_t i = 0; i < this->nc; i++){
            for (size_t j = 0; j < k; j++){
                nn_dists[i * k + j] = search_nn_dists[i * (k + 1) + j + 1];
                labels[i * k + j] = search_nn_idxs[i * (k + 1) + j + 1];
            }
        }
    }



}