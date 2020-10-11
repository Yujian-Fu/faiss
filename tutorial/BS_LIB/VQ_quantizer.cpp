#include "VQ_quantizer.h"

namespace bslib{

    /**
     * This is the initialize function for VQ function:
     * 
     * Input: 
     * nc_upper:        number of groups = number of centroids in upper layer
     * nc_per_group:    number of subcentroids per group
     * use_HNSW:        use HNSW for searching or L2 exhaustive search
     * M:               for HNSW construction, 16 in default
     * efConstruction:  for HNSW construction
     * efSearch:        for efSearch: keep_space < efSearch < nc_per_group
     * 
     **/
    VQ_quantizer::VQ_quantizer(size_t dimension, size_t nc_upper, size_t nc_per_group, size_t M, size_t efConstruction, size_t efSearch, bool use_HNSW):
        Base_quantizer(dimension, nc_upper, nc_per_group), use_HNSW(use_HNSW){
            if (use_HNSW){
                this->M = M;
                this->efConstruction = efConstruction;
                this->efSearch = efSearch;
                this->HNSW_quantizers.resize(nc_upper);
            }
            else{
                this->L2_quantizers.resize(nc_upper);
            }
        }

    /**
     * Construct centroids for index in VQ layer
     * 
     * Input: 
     * train_data:      the train data for construct centroids        size: train_set_size * dimension
     * train_set_size:  the train data set size                       size_t
     * train_data_ids: the group_id for train data                    size: train_set_size
     * 
     * Output:
     * train_data_ids: updated train group id
     * 
     **/
    void VQ_quantizer::build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_ids){
        std::cout << "Adding " << train_set_size << " train set data into " << nc_upper << " groups " << std::endl;
        std::vector<std::vector<float>> train_set(this->nc_upper);

        for (size_t i = 0; i < train_set_size; i++){
            idx_t group_id = train_data_ids[i];
            assert(group_id <= this->nc_upper);
            for (size_t j = 0; j < dimension; j++)
                train_set[group_id].push_back(train_data[i*dimension + j]);
        }

        std::cout << "Building group quantizers for vq_quantizer " << std::endl;
        size_t min_train_size = train_set[0].size() / dimension; 
        for (size_t i = 0; i < nc_upper; i++){if (min_train_size > train_set[i].size() / dimension) min_train_size = train_set[i].size() / dimension;std::cout << train_set[i].size()<<" ";}

        std::cout <<  std::endl << "The min size for sub train set is: " << min_train_size << std::endl;

#pragma omp parallel for
        for (size_t i = 0; i < nc_upper; i++){
            std::vector<float> centroids(dimension * nc_per_group);
            size_t nt_sub = train_set[i].size() / this->dimension;
            //std::cout << "Clustering " << nt_sub << " train vectors into " << nc_per_group << " groups " << std::endl;
            faiss::kmeans_clustering(dimension, nt_sub, nc_per_group, train_set[i].data(), centroids.data());

            //Adding centroids into quantizers
            if (use_HNSW){
                hnswlib::HierarchicalNSW * centroid_quantizer = new hnswlib::HierarchicalNSW(dimension, nc_per_group, M, 2 * M, efConstruction);
                for (size_t j = 0; j < nc_per_group; j++){
                    centroid_quantizer->addPoint(centroids.data() + j * dimension);
                }
                this->HNSW_quantizers[i] = centroid_quantizer;
            }
            else
            {
                faiss::IndexFlatL2 * centroid_quantizer = new faiss::IndexFlatL2(dimension);
                centroid_quantizer->add(nc_per_group, centroids.data());
                this->L2_quantizers[i] = centroid_quantizer;
            }
        }
        std::cout << "finished computing centoids" <<std::endl;
    }


    /**
     * 
     * This is the function for updating the ids for train set data
     * 
     * Input:
     * train_data: the new train data for next layer     size: n * dimension
     * n: the size for query_set
     * 
     * Output:
     * train_data_ids: the ids for train vectors        size: n
     * 
     **/
    void VQ_quantizer::search_all(const size_t n, const size_t k, const float * query_data, idx_t * query_data_ids){
        faiss::IndexFlatL2 centroid_index(dimension * k);
        std::vector<float> one_centroid(dimension * k);

        for (size_t group_id = 0; group_id < nc_upper; group_id++){
            for (size_t inner_group_id = 0; inner_group_id < nc_per_group; inner_group_id++){
                compute_final_centroid(group_id, inner_group_id, one_centroid.data());
                centroid_index.add(1, one_centroid.data());
            }
        }

        std::vector<float> result_dists(n);
        centroid_index.search(n, query_data, k, result_dists.data(), query_data_ids);
    }
    

    /**
     * For a query sequence queries, search in their group and return the k cloest centroids
     * 
     * Notice: 
     * nc_per_group should be larger than efSearch and k should be smaller than efSearch
     * Use parallel when n is large, in searching, ignore parallel
     * 
     * Input:
     * Queries: the query data points                                               size: n * dimension
     * group_ids: the group id for relative query: which group should we search in  size: n
     * 
     * Output:
     * result_labels in L2 search: the search result labels in L2 search mode       size: n * nc_per_group
     * result_labels in HNSW search: the search result labels in HNSW search mode   size: n * k (n * efSearch?)
     * result_dists in L2 search:                                                   size: n * nc_per_group
     * result_dists in HNSW search:                                                 size: n * k (n * efSearch?)
     * The output result for HNSW search and L2 search is different
     **/
    void VQ_quantizer::search_in_group(const float * query, const idx_t group_id, float * result_dists, idx_t * result_labels, size_t k){
        if (use_HNSW){
//#pragma omp parallel for
            //The distance result for search kNN is in reverse 
            size_t search_para = efSearch > k ? efSearch : k;
            auto result_queue = this->HNSW_quantizers[group_id]->searchBaseLayer(query, search_para);
            size_t result_length = result_queue.size();
            // The result of search result 
            assert (result_length >= k && result_length <= nc_per_group);
            for (size_t j = 0; j < this->nc_per_group; j++){
                if (j < result_length){
                    result_dists[result_length - j - 1] = result_queue.top().first;
                    result_labels[result_length - j -1] = CentroidDistributionMap[group_id] + result_queue.top().second;
                    result_queue.pop();
                }
                else{
                    result_dists[j] = MAX_DIST;
                    result_labels[j] = INVALID_ID;
                }
            }
        }
        else{
//#pragma omp parallel for
            for (size_t j = 0; j < this->nc_per_group; j++){
                const float * target_centroid = this->L2_quantizers[group_id]->xb.data() + j * this->dimension;
                result_dists[j] = faiss::fvec_L2sqr(target_centroid, query, dimension);
                result_labels[j] = CentroidDistributionMap[group_id] + j;
            }
        }
    }


    /**
     * The size for final centroid: dimension
     * The label is the idx for sub centroid in every layer: nc_upper * nc_per_group
     * 
     * Input: 
     * group_id:        the group of the target centroid   
     * inner_group_id:  the inner id of centroid in group
     * 
     * Output:
     * final_centroid:  the target centroid          size: dimension
     **/
    void VQ_quantizer::compute_final_centroid(const idx_t group_id, const idx_t inner_group_id, float * final_centroid){
        if (use_HNSW){
            const float * target_centroid = this->HNSW_quantizers[group_id]->getDataByInternalId(inner_group_id);
            for (size_t i = 0; i < dimension; i++){
                final_centroid[i] = target_centroid[i];
            }
        }
        else{
            for (size_t i = 0; i < dimension; i++){
                final_centroid[i] = this->L2_quantizers[group_id]->xb[inner_group_id * this->dimension + i];
            }
        }
    }


    /**
     * This is for computing residual between data point and the centroid
     * 
     * Input:
     * n:             number of data points.               size: umlimited
     * labels:        id of the centroid, [0, nc].         size: n
     * x:             data point.                          size: n * dimension
     * 
     * Output:
     * residuals:     residual of data point and centroid  size: n * dimension
     * 
     **/
    void VQ_quantizer::compute_residual_group_id(size_t n,  const idx_t * labels, const float * x, float * residuals){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            size_t group_id = labels[i] / nc_per_group;
            size_t inner_group_id = labels[i] - CentroidDistributionMap[group_id];

            compute_final_centroid(group_id, inner_group_id, final_centroid.data());
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, final_centroid.data(), residuals + i * dimension);
        }
    }


    /**
     * This is for recover data point with residual and the centroid
     * 
     * Input:
     * n:             number of residual points.           size: umlimited
     * labels:        id of the centroid, [0, nc].         size: n
     * residuals:     precomputed residual of data point.  size: n * dimension
     * 
     * Output:
     * x:             reconstructed data point             size: n * dimension
     **/
    void VQ_quantizer::recover_residual_group_id(size_t n, const idx_t * labels, const float * residuals, float * x){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            size_t group_id = labels[i] / nc_per_group;
            size_t inner_group_id = labels[i] - CentroidDistributionMap[group_id];

            compute_final_centroid(group_id, inner_group_id, final_centroid.data());
            faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, final_centroid.data(), x + i * dimension);
        }
    }


    /**
     * This is for building VQ layer, to compute the nearest neighbor for every centroid
     * The neighbor is global: should consider neighbors in different group
     * 
     * Input:
     * k: is the number of NN neighbors to be computed    
     * 
     * Output:
     * nn_centroids: data of the upper layer (this layer) centroids       size: nc * dimension
     * nn_dists: distance of the upper layer nearest neighbor centroids   size: nc * k
     * nn_labels: idx of the upper layer nearest neighbor centroids       size: nc * k
     **/

    void VQ_quantizer::compute_nn_centroids(size_t k, float * nn_centroids, float * nn_dists, idx_t * nn_ids){
        faiss::IndexFlatL2 all_quantizer(dimension);

        //Add all centroids to the all_quantizer
        for (size_t group_id = 0; group_id < this->nc_upper; group_id++){
            for (size_t inner_group_id = 0; inner_group_id < this->nc_per_group; inner_group_id++){
                std::vector<float> target_centroid(dimension);
                compute_final_centroid(group_id, inner_group_id, target_centroid.data());
                all_quantizer.add(1, target_centroid.data());
            }
        }

        std::cout << "searching the idx for centroids' nearest neighbors " << std::endl;
        for (size_t i = 0; i < this->nc * this->dimension; i++){
            nn_centroids[i] = all_quantizer.xb[i];
        }

        std::vector<idx_t> search_nn_ids(this->nc * (k+1));
        std::vector<float> search_nn_dis (this->nc * (k+1));
        all_quantizer.search(this->nc, all_quantizer.xb.data(), k+1, search_nn_dis.data(), search_nn_ids.data());

        for (size_t i = 0; i < this->nc; i++){
            for (size_t j = 0; j < k; j++){
                nn_dists[i * k + j] = search_nn_dis[i * (k + 1) + j + 1];
                nn_ids [i * k + j] = search_nn_ids[i * (k + 1) + j + 1];
            }
        }
    }

    void VQ_quantizer::write_HNSW(std::ofstream & output){
        std::cout << "Saving HNSW Index" << std::endl;
        assert(HNSW_quantizers.size() == nc_upper);

        for (size_t HNSW_id = 0; HNSW_id < nc_upper; HNSW_id++){
            //Saving Info
            hnswlib::HierarchicalNSW * HNSW = HNSW_quantizers[HNSW_id];
            writeBinaryPOD(output, HNSW->maxelements_);
            writeBinaryPOD(output, HNSW->enterpoint_node);
            writeBinaryPOD(output, HNSW->data_size_);
            writeBinaryPOD(output, HNSW->offset_data);
            writeBinaryPOD(output, HNSW->size_data_per_element);
            writeBinaryPOD(output, HNSW->M_);
            writeBinaryPOD(output, HNSW->maxM_);
            writeBinaryPOD(output, HNSW->size_links_level0);

            //Saving Edges
            for (size_t i = 0; i < HNSW->maxelements_; i++) {
                uint8_t *ll_cur = HNSW->get_linklist0(i);
                uint32_t size = *ll_cur;

                output.write((char *) &size, sizeof(uint32_t));
                hnswlib::idx_t *data = (hnswlib::idx_t *)(ll_cur + 1);
                output.write((char *) data, sizeof(hnswlib::idx_t) * size);
            }

            //Saving Data
            for (size_t i = 0; i < nc_per_group; i++){
                output.write((char *) & dimension, sizeof(uint32_t));
                output.write((char *) HNSW->getDataByInternalId(i), dimension * sizeof(float));
            }
        }
    }

    void VQ_quantizer::read_HNSW(std::ifstream & input){
        std::cout << "Reading HNSW index" << std::endl;
        assert(HNSW_quantizers.size() == nc_upper);

        for (size_t HNSW_id = 0; HNSW_id < nc_upper; HNSW_id++){
            hnswlib::HierarchicalNSW * HNSW = new hnswlib::HierarchicalNSW();

            //Load Info
            readBinaryPOD(input, HNSW->maxelements_);
            readBinaryPOD(input, HNSW->enterpoint_node);
            readBinaryPOD(input, HNSW->data_size_);
            readBinaryPOD(input, HNSW->offset_data);
            readBinaryPOD(input, HNSW->size_data_per_element);
            readBinaryPOD(input, HNSW->M_);
            readBinaryPOD(input, HNSW->maxM_);
            readBinaryPOD(input, HNSW->size_links_level0);

            HNSW->d_ = HNSW->data_size_ / sizeof(float);
            HNSW->data_level0_memory_ = (char *) malloc(HNSW->maxelements_ * HNSW->size_data_per_element);

            HNSW->efConstruction_ = 0;
            HNSW->cur_element_count = HNSW->maxelements_;

            HNSW->visitedlistpool = new hnswlib::VisitedListPool(1, HNSW->maxelements_);

            //Load Edges
            uint32_t size;
            for (size_t i = 0; i < HNSW->maxelements_; i++) {
                input.read((char *) &size, sizeof(uint32_t));

                uint8_t *ll_cur = HNSW->get_linklist0(i);
                *ll_cur = size;
                hnswlib::idx_t *data = (hnswlib::idx_t *)(ll_cur + 1);

                input.read((char *) data, size * sizeof(hnswlib::idx_t));
            }

            //Load data
            uint32_t dim;
            float mass[HNSW->d_];
            for (size_t i = 0; i < HNSW->maxelements_; i++) {
                input.read((char *) &dim, sizeof(uint32_t));
                if (dim != HNSW->d_) {
                    std::cout << "Wront data dim" << std::endl;
                    exit(1);
                }
                input.read((char *) mass, dim * sizeof(float));
                memcpy(HNSW->getDataByInternalId(i), mass, HNSW->data_size_);
            }
            HNSW_quantizers[HNSW_id] = HNSW;
        }
    }
}