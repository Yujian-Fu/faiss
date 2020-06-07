#include "VQ_quantizer.h"

namespace bslib{
    VQ_quantizer::VQ_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group):
        Base_quantizer(dimension, nc_upper, nc_per_group){}


    void VQ_quantizer::build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_idxs){
        std::cout << "Adding " << train_set_size << " train set data into " << nc_upper << " group " << std::endl;
        std::vector<std::vector<float>> centroid_train_set;
        centroid_train_set.resize(nc_upper);

        for (size_t i = 0; i < train_set_size; i++){
            idx_t idx = train_data_idxs[i];
            for (size_t j = 0; j < dimension; j++)
                centroid_train_set[idx].push_back(train_data[i*dimension + j]);
        }

        std::cout << "Building all_quantizer and group quantizers for vq_quantizer " << std::endl;
        this->all_quantizer = faiss::IndexFlatL2(dimension);
        for (size_t i = 0; i < nc_upper; i++){
            std::vector<float> centroids(dimension * nc_per_group);
            size_t nt_sub = centroid_train_set[i].size();
            std::cout << "Running kmeans for " << i << " th group with " << nt_sub / dimension << " points to generate " << nc_per_group << " centroids " << std::endl;
            faiss::kmeans_clustering(dimension, nt_sub, nc_per_group, centroid_train_set[i].data(), centroids.data());
            std::cout << "Finished kmeans, add centroids to quantizers" << std::endl;
            faiss::IndexFlatL2 centroid_quantizer(dimension);
            centroid_quantizer.add(nc_per_group, centroids.data());
            this->all_quantizer.add(nc_per_group, centroids.data());
            this->quantizers.push_back(centroid_quantizer);
        }

        std::cout << "Finding the centroid idxs for train vectors for futher quantizer construction " << std::endl;
        //Find the centroid idxs for train vectors
        std::vector<float> centroid_distances(train_set_size);
        std::vector<faiss::Index::idx_t> centroid_idxs(train_set_size);
        all_quantizer.search(train_set_size, train_data, 1, centroid_distances.data(), centroid_idxs.data());
        for (size_t i = 0; i < train_set_size; i++){
            train_data_idxs[i] = centroid_idxs[i];
        }
    }

    void VQ_quantizer::search_in_group(size_t n, const float * instances, size_t k, float * dists, idx_t * labels, const idx_t * group_id){
        std::vector<float> query_dists(k);
        std::vector<faiss::Index::idx_t> query_labels(k);
        for (size_t i = 0; i < n; i++){
            this->quantizers[group_id[i]].search(1, instances + i * dimension, k, query_dists.data() + i * k, query_labels.data() + i * k);
        }
        
        for (size_t i = 0; i < n; i++){
            size_t base_idx = CentroidDistributionMap[group_id[i]];
            for (size_t j = 0; j < k; j++){
                labels[i] = base_idx + query_labels[i * k + j];
                dists[i] = query_dists[i * k + j];
            }
        }
    }

    void VQ_quantizer::compute_final_centroid(idx_t label, float * final_centroid){
        size_t j = size_t (label / nc_per_group);
        /*
        for (j = 0; j < nc_upper; j++){
            if (CentroidDistributionMap[j+1] >= label){
                break;
            }
        }
        */
        size_t group_label = label - CentroidDistributionMap[j];
        for (size_t i = 0; i < dimension; i++)
            final_centroid[i] = this->quantizers[group_label].xb[group_label * this->dimension + i];
    }


    void VQ_quantizer::compute_residual_group_id(size_t n,  const idx_t * labels, const float * x, float * residuals){
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid;
            compute_final_centroid(labels[i], final_centroid.data());
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, final_centroid.data(), residuals + i * dimension);
        }
    }

    void VQ_quantizer::recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x){
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid;
            compute_final_centroid(labels[i], final_centroid.data());
            faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, final_centroid.data(), x + i * dimension);
        }
    }

    void VQ_quantizer::compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * labels){
        std::vector<faiss::Index::idx_t>  search_nn_idxs(this->nc * (k + 1));
        std::vector<float> search_nn_dists ( this->nc * (k+1));
        this->all_quantizer.search(this->nc, this->all_quantizer.xb.data(), k+1, search_nn_dists.data(), search_nn_idxs.data());
        
        for (size_t i = 0; i < this->nc; i++){
            for (size_t j = 0; j < k; j++){
                nn_dists[i * k + j] = search_nn_dists[i * (k + 1) + j + 1];
                labels[i * k + j] = search_nn_idxs[i * (k + 1) + j + 1];
            }
            for (size_t j = 0; j < dimension; j++){
                nn_centroids[i * dimension + j] = all_quantizer.xb[i * dimension + j];
            }
        }
    }




}