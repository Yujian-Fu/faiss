#include "bslib_index.h"
#include "time.h"

namespace bslib{

    Bslib_Index::Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type, const bool use_subset, const bool pq_use_subset):
        dimension(dimension), layers(layers){
            this->pq_use_subset = pq_use_subset;
            this->use_subset = use_subset;
            this->index_type.resize(layers);

            for (size_t i = 0; i < layers; i++){
                this->index_type[i] = index_type[i];
            }
            this->nt = 0;
            this->subnt = 0;
        }
    
    void Bslib_Index::add_vq_quantizer(size_t nc_upper, size_t nc_per_group, bool update_idxs){
        
        VQ_quantizer vq_quantizer (this->dimension, nc_upper, nc_per_group);
        ShowMessage("Building centroids for vq quantizer with train data idxs");
        vq_quantizer.build_centroids(this->train_data.data(), this->nt, this->train_data_idxs.data(), update_idxs);
        ShowMessage("Checking whether all vq centroids are added correctly");
        for (size_t i = 0; i < nc_upper; i++){
            assert(vq_quantizer.quantizers[i].xb.size() == nc_per_group * dimension);
        }
        this->vq_quantizer_index.push_back(vq_quantizer);
    }

    void Bslib_Index::add_lq_quantizer(size_t nc_upper, size_t nc_per_group, const float * upper_centroids, const idx_t * upper_nn_centroid_idxs, const float * upper_nn_centroid_dists, bool update_idxs){
        LQ_quantizer lq_quantizer (dimension, nc_upper, nc_per_group, upper_centroids, upper_nn_centroid_idxs, upper_nn_centroid_dists);
        ShowMessage("Building centroids for lq quantizer");
        lq_quantizer.build_centroids(this->train_data.data(), this->nt, this->train_data_idxs.data(), update_idxs);
        //Check whether all centroids are added in all_quantizer correctly 
        ShowMessage("Showing samples of alpha");
        this->lq_quantizer_index.push_back(lq_quantizer);
    }



    void Bslib_Index::encode(size_t n, const float * encode_data, const idx_t * encoded_ids, float * encoded_data){
        if (index_type[layers-1] == "VQ"){
            std::cout << "Encoding in VQ layer " << std::endl;
            vq_quantizer_index[vq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else if(index_type[layers-1] == "LQ"){
            std::cout << "Encoding in LQ layer " << std::endl;
            lq_quantizer_index[lq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else{
            std::cout << "The type name is wrong with " << index_type[layers - 1] << "!" << std::endl;
            exit(0);
        }
    }

    void Bslib_Index::decode(size_t n, const float * encoded_data, const idx_t * encoded_ids, float * decoded_data){
        if (index_type[layers-1] == "VQ"){
            vq_quantizer_index[vq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data);
        }
        else if (index_type[layers-1] == "LQ"){
            lq_quantizer_index[lq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data);
        }
        else{
            std::cout << "The type name is wrong with " << index_type[layers - 1] << "!" << std::endl;
            exit(0);
        }
    }

    void Bslib_Index::build_quantizers(const uint32_t * ncentroids, const char * path_quantizers, const char * path_learn){
        if (exists(path_quantizers)){
            read_quantizers(path_quantizers);
            std::cout << "Checking the quantizers read from file " << std::endl;
            std::cout << "The number of quantizers: " << this->vq_quantizer_index.size() << " " << this->lq_quantizer_index.size() << std::endl;
            std::cout  << "The number of centroids in each quantizer: ";
            for (size_t i = 0 ; i < this->vq_quantizer_index[0].quantizers.size(); i++){
                std::cout << vq_quantizer_index[0].quantizers[i].xb.size() / dimension << " ";
            }
        }
        else
        {
        ShowMessage("No preconstructed quantizers, constructing quantizers");
        //Load the train set into the index
        this->train_data.resize(this->nt * this->dimension);
        std::cout << "Loading learn set from " << path_learn << std::endl;
        std::ifstream learn_input(path_learn, std::ios::binary);
        readXvecFvec<learn_data_type>(learn_input, this->train_data.data(), this->dimension, this->nt, true);
        this->train_data_idxs.resize(this->nt);
        for (size_t i = 0; i < this->nt; i++){
            this->train_data_idxs[i] = 0;
        }
        if (this->use_subset){
            ShowMessage("Using subset of the train set, saving sub train set");
            std::vector<float> train_subset(this->subnt * this->dimension);
            RandomSubset(this->train_data.data(), train_subset.data(), this->dimension, this->nt, this->subnt);
            train_data.resize(this->subnt * this->dimension);
            for (size_t i = 0; i < subnt * dimension; i++){
                this->train_data[i] = train_subset[i];
            }
            this->nt = this->subnt;
            CheckResult<float>(this->train_data.data(), this->dimension);
        }

        assert(index_type.size() == layers && index_type[0] != "LQ");
        std::cout << "adding layers to the index structure " << std::endl;
        // The number of centroids in the upper layer
        uint32_t nc_upper = 1; 
        // The number of centroids in each group (one upper centroid) 
        uint32_t nc_per_group;
        std::vector<float> upper_centroids;
        std::vector<idx_t> nn_centroids_idxs;
        std::vector<float> nn_centroids_dists;

        for (size_t i = 0; i < layers; i++){
            nc_per_group = ncentroids[i];
            if (index_type[i] == "VQ"){
                ShowMessage("Adding VQ quantizer");
                bool update_idxs = (i == layers-1) ? false:true;
                add_vq_quantizer(nc_upper, nc_per_group, update_idxs);
                std::cout << i << "th VQ quantizer added, check it " << std::endl;
                std::cout << "The vq quantizer size is: " <<  vq_quantizer_index.size() << " the num of quantizers (groups): " << vq_quantizer_index[vq_quantizer_index.size()-1].quantizers.size() << std::endl;
            }
            else if(index_type[i] == "LQ"){
                assert (i >= 1);
                bool update_idxs = (i == layers-1) ? false:true;

                upper_centroids.resize(nc_upper * dimension);
                nn_centroids_idxs.resize(nc_upper * nc_per_group);
                nn_centroids_dists.resize(nc_upper * nc_per_group);
                
                if (index_type[i-1] == "VQ"){
                    ShowMessage("Adding VQ quantizer with VQ front layer");
                    size_t last_vq = vq_quantizer_index.size() - 1;
                    ShowMessage("VQ computing nn centroids");
                    assert(vq_quantizer_index[last_vq].nc > nc_per_group);
                    vq_quantizer_index[last_vq].compute_nn_centroids(nc_per_group, upper_centroids.data(), nn_centroids_dists.data(), nn_centroids_idxs.data());
                    add_lq_quantizer(nc_upper, nc_per_group, upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data(), update_idxs);
                }
                else if (index_type[i-1] == "LQ"){
                    ShowMessage("Adding LQ quantizer with LQ front layer");
                    size_t last_lq = lq_quantizer_index.size() - 1;
                    ShowMessage("LQ computing nn centroids");
                    assert(lq_quantizer_index[last_lq].nc > nc_per_group);
                    lq_quantizer_index[last_lq].compute_nn_centroids(nc_per_group, upper_centroids.data(), nn_centroids_dists.data(), nn_centroids_idxs.data());
                    add_lq_quantizer(nc_upper, nc_per_group, upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data(), update_idxs);
                }

                std::cout << "lq quantizer added, check it " << std::endl;
                std::cout << "The lq quantizer size is: " <<  lq_quantizer_index.size() << " the num of alphas: " << lq_quantizer_index[lq_quantizer_index.size()-1].alphas.size();
            }
            nc_upper  = nc_upper * nc_per_group;
        }
        write_quantizers(path_quantizers);
        }  
    }

    void Bslib_Index::train_pq(const char * path_pq, const char * path_norm_pq, const char * path_learn){
        // If the train dataset is not loaded
        if (this->train_data.size() != this->nt * dimension){
            std::cout << "Load the train set for PQ training" << std::endl;
            this->train_data.resize(this-> nt * dimension);
            std::ifstream learn_input(path_learn, std::ios::binary);
            readXvecFvec<learn_data_type>(learn_input, this->train_data.data(), this->dimension, this->nt, true);
        }

        if (this->pq_use_subset && this->nt != this->subnt){
            std::cout << "Using subset for pq training " << std::endl;
            assert (this->nt != this->subnt);
            std::vector<float> train_subset(subnt * dimension);
            RandomSubset(this->train_data.data(), train_subset.data(), dimension, this->nt, this->subnt);
            train_data.resize(subnt * dimension);
            train_data_idxs.resize(subnt);
            for (size_t i = 0; i < subnt * dimension; i++){
                this->train_data[i] = train_subset[i];
            }
            this->nt = this->subnt;
        }

        std::cout << "Initilizing index " << std::endl;
        this->pq = faiss::ProductQuantizer(this->dimension, this->M, this->nbits);
        this->norm_pq = faiss::ProductQuantizer(1, this->norm_M, this->nbits);
        this->code_size = this->pq.code_size;
        this->norm_code_size = this->norm_pq.code_size;

        std::cout << "Assigning the train dataset to compute residual" << std::endl;
        std::vector<float> residuals(this->dimension * this->nt);
        std::vector<idx_t> group_ids(this->nt);
        assign(this->nt, this->train_data.data(), group_ids.data());

        std::cout << std::endl;
        std::cout << "Encoding the train dataset " << std::endl;
        encode(this->nt, this->train_data.data(), group_ids.data(), residuals.data());
        /*
        std::cout << "Checking the residual data " << std::endl;
        for (size_t i = 0; i < 10; i++){
            std::cout << group_ids[i] << " ";
        }
        std::cout << std::endl;
        std::vector<float> centroid(dimension);
        this->lq_quantizer_index[0].compute_final_centroid(group_ids[0], centroid.data());
        for (size_t i = 0; i < dimension; i++){
            std::cout << train_data[i] << " " << residuals[i] << " " << centroid[i] << "          ";
        }
        std::cout << std::endl;
        */

        std::cout << "Training the pq " << std::endl;
        this->pq.verbose = true;
        this->pq.train(nt, residuals.data());
        faiss::write_ProductQuantizer(& this->pq, path_pq);

        std::cout << "decoding the residual data " << std::endl;
        std::vector<float> reconstructed_x(dimension * this->nt);
        decode(this->nt, residuals.data(), group_ids.data(), reconstructed_x.data());
        /*
        std::cout << "Checking the reconstructed data " << std::endl;
        for (size_t i = 0; i < dimension; i++){
            std::cout << train_data[i] << " "  << reconstructed_x[i] << "          ";
        }
        std::cout << std::endl;
        */

        std::vector<float> xnorm(this->nt);
        for (size_t i = 0; i < this->nt; i++){
            xnorm[i] = faiss::fvec_norm_L2sqr(reconstructed_x.data() + i * dimension, dimension);
        }
        std::cout << "Training the norm pq" << std::endl;
        this->norm_pq.verbose = true;
        this->norm_pq.train(this->nt, xnorm.data());
        faiss::write_ProductQuantizer(& this->norm_pq, path_norm_pq);
    }
    
    void Bslib_Index::get_final_nc(){
        if (this->index_type[layers -1] == "VQ"){
            this->final_nc =  vq_quantizer_index[vq_quantizer_index.size() -1].nc;
        }
        else{
            this->final_nc =  lq_quantizer_index[lq_quantizer_index.size() -1].nc;
        }
    }

    void Bslib_Index::add_batch(size_t n, const float * data, const idx_t * ids, idx_t * encoded_ids){
        std::vector<float> residuals(n * dimension);
        encode(n, data, encoded_ids, residuals.data());

        //The code is for the residual between base vectors and their final neighbor centroids
        std::vector<uint8_t> batch_codes(n * this->code_size);
        this->pq.compute_codes(residuals.data(), batch_codes.data(), n);
        std::cout << "The sample codes " << std::endl;
        for (size_t i = 0; i < 10 ; i++){
            for (size_t j = 0; j < this->code_size; j++){
                std::cout << (float)batch_codes[i*code_size + j] << " ";
            }
            std::cout << std::endl;
        }
        

        std::vector<float> reconstructed_x(n * dimension);

        /*
        Todo: should we use the decoded reconstructed_x for exp? actually we may use
        the distance in full precision for exp?
        */
       
        decode(n, residuals.data(), encoded_ids, reconstructed_x.data());

        std::cout << "Compare origin vector and reconstructed codes " << std::endl;
        for (size_t i = 0; i < 10; i++){
            for (size_t j = 0; j < dimension; j++){
                std::cout << data[i * dimension + j] << " " << reconstructed_x[i * dimension + j] << "   ";
            }
            std::cout << std::endl;
        }
        /*
        Use the origin distance can save time?
        */
        std::vector<float> xnorms (n);
        for (size_t i = 0; i < n; i++){
            xnorms[i] =  faiss::fvec_norm_L2sqr(data + i * dimension, dimension);
        }

        std::vector<uint8_t> xnorm_codes (n * norm_code_size);

        this->norm_pq.compute_codes(xnorms.data(), xnorm_codes.data(), n);

        std::cout << "The sample codes " << std::endl;
        for (size_t i = 0; i < 10 ; i++){
            for (size_t j = 0; j < this->norm_code_size; j++){
                std::cout << (float)xnorm_codes[i*code_size + j] << " ";
            }
            std::cout << std::endl;
        }

        for (size_t i = 0 ; i < n; i++){
            for (size_t j = 0; j < this->code_size; j++){
                this->base_codes[encoded_ids[i]].push_back(batch_codes[i * this->code_size + j]);
            }

            for (size_t j =0; j < this->norm_code_size; j++){
                this->base_norm_codes[encoded_ids[i]].push_back(xnorm_codes[i * this->norm_code_size +j]);
            }

            this->origin_ids[encoded_ids[i]].push_back(ids[i]);
        }
    }

    void Bslib_Index::compute_centroid_norm(){
        if (this->index_type[layers -1] == "VQ"){
            size_t n_vq = vq_quantizer_index.size();
            
            for (size_t i = 0; i < final_nc; i++){
                assert(final_nc == vq_quantizer_index[n_vq-1].nc);
                std::vector<float> each_centroid(dimension);
                vq_quantizer_index[n_vq-1].compute_final_centroid(i, each_centroid.data());
                this->centroid_norms[i] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
            }
            this->norm_pq.compute_codes(centroid_norms.data(), this->centroid_norm_codes.data(), this->final_nc);
        }
        else{
            size_t n_lq = lq_quantizer_index.size();
            std::vector<float> centroid_norms(final_nc);

            for (size_t i = 0; i < final_nc; i++){
                assert (final_nc == lq_quantizer_index[n_lq-1].nc);
                std::vector<float> each_centroid(dimension);
                lq_quantizer_index[n_lq-1].compute_final_centroid(i, each_centroid.data());
                this->centroid_norms[i] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
            }
            this->norm_pq.compute_codes(centroid_norms.data(), this->centroid_norm_codes.data(), this->final_nc);
        }
    }

    void Bslib_Index::assign(const size_t n, const float * assign_data, idx_t * assigned_ids){
        
        //std::cout << "assigning vectors " << std::endl;
        std::vector<idx_t> group_idxs (n);
        std::vector<float> group_dists(n);
        for (size_t i = 0; i < n; i++){
            group_idxs[i] = 0;
            group_dists[i] = 0.0;
        }

        size_t n_vq = 0;
        size_t n_lq = 0;

        std::vector<float> result_dists;
        std::vector<idx_t> result_labels;
        std::vector<std::map<idx_t, float>>  queries_upper_centroid_dists(n);

        //clock_t starttime1 = clock();
        for (size_t i = 0; i < this->layers; i++){
            assert(n_lq + n_vq == i);
            size_t group_size;
            if (index_type[i] == "VQ"){
                //std::cout << "searching in VQ layer" << std::endl;
                group_size = vq_quantizer_index[n_vq].nc_per_group;
                result_dists.resize(group_size * n);
                vq_quantizer_index[n_vq].search_in_group(n, assign_data, group_idxs.data(), result_dists.data());
                result_labels.resize(group_size * n);
                for (size_t j = 0; j < n; j++){
                    for (size_t m = 0; m < group_size; m++){
                        result_labels[j * group_size + m] = group_idxs[j] * group_size + m;
                    }
                }
                n_vq ++;
            }

            else if(index_type[i] == "LQ"){
                //std::cout << "searching in LQ layer" << std::endl;
                group_size = lq_quantizer_index[n_lq].nc_per_group;
                result_dists.resize(group_size * n);
                lq_quantizer_index[n_lq].search_in_group(n, assign_data, queries_upper_centroid_dists, group_idxs.data(), result_dists.data());
                result_labels.resize(group_size * n);
                for (size_t j = 0; j < n; j++){
                    queries_upper_centroid_dists[j].clear();
                    for (size_t m = 0; m < group_size; m++){
                        result_labels[j * group_size + m] = group_idxs[j] * group_size + m;
                    }
                }
                n_lq ++;
            }
            else{
                std::cout << "The type name is wrong with " << index_type[i] << "!" << std::endl;
                exit(0); 
            }

            if (i < this->layers-1 && index_type[i+1] == "LQ"){
                //std::cout << "The next layer is LQ, load the query centroid distsnaces" << std::endl;
                for (size_t j = 0; j < n; j++){
                    for (size_t m = 0; m < group_size; m++)
                        queries_upper_centroid_dists[j].insert(std::pair<idx_t, float>(result_labels[j*group_size+m], result_dists[j*group_size+m]));
                }
            }

            //std::cout << "Choosing k instances with smallest distances " << std::endl;

            //clock_t starttime = clock();
            for (size_t j = 0; j < n; j++){
                keep_k_min(group_size, 1, result_dists.data()+j*group_size, result_labels.data()+j*group_size, group_dists.data()+j, group_idxs.data()+j);
            }
            //clock_t endtime = clock();
            //std::cout << "Time in selct k min: " << float(endtime - starttime) / CLOCKS_PER_SEC << std::endl;
        }
        //clock_t endtime1 = clock();

        assert((n_vq + n_lq) == this->layers);
        for (size_t i = 0; i < n; i++){
            assigned_ids[i] = group_idxs[i];
        }

        /*
        std::cout << "Checking whether the ids are correctly assigned " << std::endl;
        faiss::IndexFlatL2 final_quantizer(dimension);
        for (size_t i = 0; i < this->final_nc; i++){
            std::vector<float> final_centroid(dimension);
            this->lq_quantizer_index[0].compute_final_centroid(i, final_centroid.data());
            final_quantizer.add(1, final_centroid.data());
        }

        clock_t starttime2 = clock();
        std::vector<faiss::Index::idx_t> final_idx(n);
        std::vector<float> final_dist(n);
        final_quantizer.search(n, assign_data, 1, final_dist.data(), final_idx.data());
        clock_t endtime2 = clock();
        size_t correct = 0;
        float dist_proportion = 0.0;
        for (size_t i = 0; i < n; i++){
            if ((group_idxs[i] - final_idx[i]) == 0){
                correct ++;
            }
            dist_proportion += group_dists[i] / final_dist[i];
            //std::cout << group_idxs[i] - final_idx[i] << "_" << group_idxs[i] << "_" << final_idx[i] << " " << group_dists[i] << "_" << final_dist[i] << " ";
        }
        std::cout << "Checking finished" << std::endl;
        std::cout << "The correct number is " << correct << " The dist proportion is: " << dist_proportion / n << std::endl;
        std::cout << "Origin search time: " << float(endtime1 - starttime1) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Test search time: " << " " << float(endtime2 - starttime2)/ CLOCKS_PER_SEC << std::endl;
        */
        
        
    }


    void Bslib_Index::keep_k_min(const size_t m, const size_t k, const float * all_dists, const idx_t * all_ids, float * sub_dists, idx_t * sub_ids){
        assert(m >= k);
        if (m == k){
            for (size_t i = 0; i < m; i++){
                sub_dists[i] = all_dists[i];
                sub_ids[i] = all_ids[i];
            }
        }
        else{
            std::vector<float> dists(k);
            std::vector<faiss::Index::idx_t> ids(k);
            faiss::maxheap_heapify(k, dists.data(), ids.data());
            for (size_t i = 0; i < m; i++){
                if (all_dists[i] < dists[0]){
                    faiss::maxheap_pop(k, dists.data(), ids.data());
                    faiss::maxheap_push(k, dists.data(), ids.data(), all_dists[i], all_ids[i]);
                }
            }

            for (size_t i = 0; i < k; i++){
                sub_ids[i] = ids[i];
                sub_dists[i] = dists[i];
            }
        }
    }


    float Bslib_Index::pq_L2sqr(const uint8_t *code)
    {
        float result = 0.;
        const size_t dim = code_size >> 2;
        size_t m = 0;
        for (size_t i = 0; i < dim; i++) {
            result += precomputed_table[this->pq.ksub * m + code[m]]; m++;
            result += precomputed_table[this->pq.ksub * m + code[m]]; m++;
            result += precomputed_table[this->pq.ksub * m + code[m]]; m++;
            result += precomputed_table[this->pq.ksub * m + code[m]]; m++;
        }
        return result;
    }

    void Bslib_Index::get_next_group_idx(size_t keep_result_space, idx_t * group_ids, float * query_group_dists, float * result_idx_dist){
        idx_t min_label = INVALID_ID;
        float min_dist = MAX_DIST;
        size_t min_i = -1;
        for (size_t i = 0; i < keep_result_space; i++){
            if (group_ids[i] != INVALID_ID){
                if (query_group_dists[i] < min_dist){
                    min_label = group_ids[i];
                    min_dist = query_group_dists[i];
                    min_i = i;
                }
            }
        }
        group_ids[min_i] = INVALID_ID;
        if (min_label == INVALID_ID){
            std::cout << std::endl <<  "No enough group ids for: " << keep_result_space << std::endl;
            exit(0);
        }
        result_idx_dist[0] = min_label;
        result_idx_dist[1] = min_dist;
    }

     /*
      *    d = || x - y_C - y_R ||^2
      *    d = || x - y_C ||^2 - || y_C ||^2 + || y_C + y_R ||^2 - 2 * (x|y_R)
      *        -----------------------------   -----------------   -----------
      *                     term 1                   term 2           term 3
      */

    void Bslib_Index::search(size_t n, size_t result_k, float * queries, float * query_dists, faiss::Index::idx_t * query_ids, size_t * keep_space, uint32_t * groundtruth){
        float overall_proportion = 0;
        float avg_visited_vectors = 0;
        std::vector<float> avg_time_consumption(layers+1);
        for (size_t i = 0; i < layers+1; i++){avg_time_consumption[i] = 0;}
        float avg_query_centroid_dist = 0;

//#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> time_consumption(layers+1);
            time_recorder Trecorder = time_recorder();

            std::ifstream base_input("/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_base.fvecs", std::ios::binary);
            
            std::unordered_set<uint32_t> grountruth_set;
            for (size_t j = 0; j < result_k; j++)
                grountruth_set.insert(groundtruth[i * 100 + j]);

            const float * query = queries + i * dimension;

            std::vector<idx_t> group_idxs(1);
            std::vector<float> group_dists(1);

            group_idxs[0] = 0;
            group_dists[0] = 0;

            std::vector<float> result_dists; 
            std::vector<idx_t> result_labels;
            std::vector<std::map<idx_t, float>> query_upper_centroid_dists(1);
            
            size_t n_vq = 0;
            size_t n_lq = 0;

            size_t keep_result_space = 1;

            for (size_t j = 0; j < layers; j++){

                keep_result_space = group_idxs.size() * keep_space[j];
                assert(n_vq+ n_lq == j);
                size_t group_size;
                
                if (index_type[j] == "VQ"){
                    //std::cout << "searching in VQ layer" << std::endl;
                    group_size = vq_quantizer_index[n_vq].nc_per_group;
                    result_dists.resize(group_idxs.size()*group_size);
#pragma omp parallel for
                    for (size_t m = 0; m < group_idxs.size(); m++){
                        vq_quantizer_index[n_vq].search_in_group(1, query, group_idxs.data()+m, result_dists.data()+m*group_size);
                    }
                    result_labels.resize(group_idxs.size()*group_size);
                    for(size_t m = 0; m < group_idxs.size(); m++){
                        for (size_t n = 0; n < group_size; n++){
                            result_labels[m * group_size + n] = group_idxs[m] * group_size + n;
                        }
                    }
                    n_vq ++;
                }

                else if(index_type[j] == "LQ") {
                    //std::cout << "searching in LQ layer" << std::endl;
                    group_size = lq_quantizer_index[n_lq].nc_per_group;
                    result_dists.resize(group_size * n);
                    assert(query_upper_centroid_dists[0].size() > 0);
#pragma omp parallel for
                    for (size_t m = 0; m < group_idxs.size(); m++){
                        lq_quantizer_index[n_lq].search_in_group(1, query, query_upper_centroid_dists, group_idxs.data()+m, result_dists.data()+m*group_size);
                    }
                    result_labels.resize(group_idxs.size()*group_size);
                    for (size_t m = 0; m < group_idxs.size(); m++){
                        for (size_t n = 0; n < group_size; n++){
                            result_labels[m * group_size + n] = group_idxs[m] * group_size + n;
                        }
                    }
                    query_upper_centroid_dists[0].clear();
                    n_lq ++;
                }
                else{
                    std::cout << "The type name is wrong with " << index_type[j] << "!" << std::endl;
                    exit(0);
                }
                
                if (j < this->layers-1 && index_type[j+1] == "LQ"){
                    //std::cout << "The next layer is LQ, load the query centroid distsnaces" << std::endl;
                    for (size_t m = 0; m < group_idxs.size()*group_size; m++){
                        query_upper_centroid_dists[0].insert(std::pair<idx_t, float>(result_labels[m], result_dists[m]));
                    }
                }

                //std::cout << "Choosing k instances with smallest distances " << std::endl;
                size_t search_space = group_size * group_idxs.size();
                group_idxs.resize(keep_result_space);
                group_dists.resize(keep_result_space);

                keep_k_min(search_space, keep_result_space, result_dists.data(), result_labels.data(), group_dists.data(), group_idxs.data());
                time_consumption[j] = Trecorder.getTimeConsumption() / (search_space / group_size);
                Trecorder.reset();
            }

            assert((n_vq + n_lq) == this->layers);
            this->pq.compute_inner_prod_table(query, this->precomputed_table.data());
            size_t visited_vectors = 0;
            size_t visited_gt = 0;
            float query_centroid_dists = 0;


            for (size_t j = 0; j < keep_result_space; j++){
                query_centroid_dists += group_dists[j];
            }
            query_centroid_dists = query_centroid_dists / keep_result_space;
            avg_query_centroid_dist += query_centroid_dists;

            
            size_t total_size = 0;
            for (size_t j = 0; j < keep_result_space; j++){
                total_size += this->origin_ids[group_idxs[j]].size();
            }
            std::vector<float> query_search_dists(total_size);
            std::vector<idx_t> query_search_labels(total_size);
            std::vector<float> query_actual_dists(total_size);


            for (size_t j = 0; j < keep_result_space; j++){

                std::vector<float> result_idx_dist(2);
                get_next_group_idx(keep_result_space, group_idxs.data(), group_dists.data(), result_idx_dist.data());
                uint32_t group_id = result_idx_dist[0];
                float q_c_dist = result_idx_dist[1];

                size_t group_size = this->origin_ids[group_id].size();
                assert(group_size == this->base_codes[group_id].size() / this->code_size);


                float term1 = q_c_dist - centroid_norms[group_id];
                std::vector<float> base_norms(group_size);
                assert(group_size == base_norm_codes[group_id].size() / this->norm_code_size);
                this->norm_pq.decode(base_norm_codes[group_id].data(), base_norms.data(), group_size);
                const uint8_t * code = base_codes[group_id].data();

                for (size_t m = 0; m < group_size; m++){
                    float term2 = base_norms[m];
                    float term3 = 2 * pq_L2sqr(code + m * code_size);
                    float dist = term1 + term2 - term3;
                    query_search_dists[visited_vectors] = dist;


                    //Compute the actual distance
                    /************************************************/
                    std::vector<float> base_vector(dimension);
                    uint32_t dim;
                    base_input.seekg(origin_ids[group_id][m] * dimension * sizeof (float) + origin_ids[group_id][m] * sizeof(uint32_t), std::ios::beg);
                    base_input.read((char *) & dim, sizeof(uint32_t));
                    assert(dim == this->dimension);
                    base_input.read((char *) base_vector.data(), sizeof(float)*dimension);

                    std::vector<float> distance_vector(dimension);
                    faiss::fvec_madd(dimension, base_vector.data(), -1, query, distance_vector.data());
                    float actual_dist =  faiss::fvec_norm_L2sqr(distance_vector.data(), dimension);
                    query_actual_dists[visited_vectors] = actual_dist;
                    /**********************************************/


                    if ((i == 2 && origin_ids[group_id][m] == 80543)){


                        std::cout << term1 << " " << term2 << " " << term3 << " " << std::endl;
                        std::vector<uint8_t> vector_codes(this->code_size);
                        std::vector<float> group_centroid(dimension);
                        this->lq_quantizer_index[0].compute_final_centroid(group_id, group_centroid.data());
                        std::vector<float> distance_vector(dimension);
                        faiss::fvec_madd(dimension, base_vector.data(), -1, group_centroid.data(), distance_vector.data());
                        this->pq.compute_code(distance_vector.data(), vector_codes.data());
                        for (size_t temp = 0; temp < code_size; temp++){std::cout << float(vector_codes[temp]) << " ";} 
                        std::cout << std::endl;
                        for (size_t temp = 0; temp < code_size; temp++){std::cout << float(code[m * code_size + temp]) << " ";}
                        std::cout << std::endl;
                        float product_sum = 0;
                        std::vector<float> reconstructed_x(dimension);
                        this->pq.decode(vector_codes.data(), reconstructed_x.data());
                        for (size_t temp = 0; temp < dimension; temp++){product_sum += reconstructed_x[temp]*query[temp];}
                        std::cout << term3 / 2 << " " << product_sum << std::endl;
                    }


                    query_search_labels[visited_vectors] = origin_ids[group_id][m];
                    visited_vectors ++;

                    if (grountruth_set.count(this->origin_ids[group_id][m]) != 0)
                        visited_gt ++;
                }
                if (visited_vectors > this->max_visited_vectors)
                    break;
            }

            //Compute the distance sort for computed distance
            std::vector<idx_t> search_dist_index(visited_vectors);
            uint32_t x=0;
            std::iota(search_dist_index.begin(),search_dist_index.end(),x++);
            std::sort( search_dist_index.begin(),search_dist_index.end(), [&](int i,int j){return query_search_dists[i]<query_search_dists[j];} );

            //Compute the distance sort for actual distance
            std::vector<idx_t> actual_dist_index(visited_vectors);
            x = 0;
            std::iota(actual_dist_index.begin(), actual_dist_index.end(), x++);
            std::sort( actual_dist_index.begin(),actual_dist_index.end(), [&](int i,int j){return query_actual_dists[i]<query_actual_dists[j];} );

            
            size_t correct = 0;
            if (use_reranking){
                size_t re_ranking_range = this->reraking_space * result_k;
                std::vector<float> reranking_dists(re_ranking_range);
                std::vector<float> reranking_labels(re_ranking_range);
                for (size_t j = 0; j < re_ranking_range; j++){
                    reranking_dists[j] = query_actual_dists[search_dist_index[j]];
                    reranking_labels[j] = query_search_labels[search_dist_index[j]];
                }

                std::vector<idx_t> reranking_dist_index(re_ranking_range);
                x = 0;
                std::iota(reranking_dist_index.begin(), reranking_dist_index.end(), x++);
                std::sort(reranking_dist_index.begin(), reranking_dist_index.end(), [&](int i,int j){return reranking_dists[i] < reranking_dists[j];});

                for (size_t j = 0; j < result_k; j++){
                    query_dists[i * result_k + j] = reranking_dists[reranking_dist_index[j]];
                    query_ids[i * result_k + j] = reranking_labels[reranking_dist_index[j]];
                    if (grountruth_set.count(query_ids[i * result_k + j]) != 0)
                        correct ++;
                }
            }

            else{
                for (size_t j = 0; j < result_k; j++){
                    query_dists[i * result_k + j] = query_search_dists[search_dist_index[j]];
                    query_ids[i * result_k + j] = query_search_labels[search_dist_index[j]];
                    if (grountruth_set.count(query_ids[i * result_k + j]) != 0)
                        correct ++;
                }
            }


            float recall = float(correct) / result_k;
            std::cout << i << " th recall: " << recall << std::endl;
            if (recall < 0){
                for (size_t temp = 0; temp < 300; temp++){
                    std::cout << query_search_labels[search_dist_index[temp]] << " " << query_search_dists[search_dist_index[temp]] << " ";
                }
                std::cout << std::endl;
                for (size_t temp = 0; temp < 300; temp ++){
                    std::cout << query_search_labels[actual_dist_index[temp]] << " " << query_actual_dists[actual_dist_index[temp]] << " ";
                }
                std::cout << std::endl;
                for (size_t temp = 0; temp < 100; temp++){
                    std::cout << groundtruth[i * 100 + temp] << " ";
                }
                std::cout << std::endl;
            }
            

           overall_proportion += float(visited_gt) / result_k;
            time_consumption[this->layers]  = Trecorder.getTimeConsumption();

            avg_visited_vectors += visited_vectors;
            for (size_t j = 0; j < layers + 1; j++){
                avg_time_consumption[j] += time_consumption[j];    
            }
        }

        std::cout << "The time consumption: ";
        for (size_t i = 0; i < layers+1; i++){
            std::cout << avg_time_consumption[i] / n << " ";
        }
        std::cout << std::endl;

        std::cout << "The average visited vectors: " << avg_visited_vectors / n << std::endl;
        std::cout << "The average query centroid distance: " << avg_query_centroid_dist / n << std::endl;
        std::cout << "The avarage groundtruth proportion is: " << overall_proportion / n << std::endl;
    }


    void Bslib_Index::write_quantizers(const char * path_quantizer){
        ShowMessage("Writing quantizers");
        std::ofstream quantizers_output(path_quantizer, std::ios::binary);
        size_t n_vq = 0;
        size_t n_lq = 0;
        for (size_t i = 0; i < this->layers; i++){
            if (index_type[i] == "VQ"){
                ShowMessage("Writing VQ quantizer layer");
                const size_t nc = vq_quantizer_index[n_vq].nc;
                const size_t nc_upper = vq_quantizer_index[n_vq].nc_upper;
                const size_t nc_per_group = vq_quantizer_index[n_vq].nc_per_group;
                quantizers_output.write((char *) & nc, sizeof(size_t));
                quantizers_output.write((char *) & nc_upper, sizeof(size_t));
                quantizers_output.write((char *) & nc_per_group, sizeof(size_t));

                for (size_t j = 0; j < nc_upper; j++){
                    size_t group_quantizer_data_size = nc_per_group * this->dimension;
                    assert(vq_quantizer_index[n_vq].quantizers[j].xb.size() == group_quantizer_data_size);
                    quantizers_output.write((char * ) vq_quantizer_index[n_vq].quantizers[j].xb.data(), group_quantizer_data_size * sizeof(float));
                }
                assert(n_vq + n_lq == i);
                n_vq ++;
            }
            else if (index_type[i] == "LQ"){
                ShowMessage("Writing LQ quantizer layer");
                const size_t nc = lq_quantizer_index[n_lq].nc;
                const size_t nc_upper = lq_quantizer_index[n_lq].nc_upper;
                const size_t nc_per_group = lq_quantizer_index[n_lq].nc_per_group;
                quantizers_output.write((char *) & nc, sizeof(size_t));
                quantizers_output.write((char *) & nc_upper, sizeof(size_t));
                quantizers_output.write((char *) & nc_per_group, sizeof(size_t));

                assert(lq_quantizer_index[n_lq].alphas.size() == nc_upper);
                quantizers_output.write((char *) lq_quantizer_index[n_lq].alphas.data(), nc_upper * sizeof(float));
                assert(lq_quantizer_index[n_lq].upper_centroids.size() == nc_upper * dimension);
                quantizers_output.write((char *) lq_quantizer_index[n_lq].upper_centroids.data(), nc_upper * this->dimension*sizeof(float));
                for (size_t j = 0; j < nc_upper; j++){
                    assert(lq_quantizer_index[n_lq].nn_centroid_idxs[j].size() == nc_per_group);
                    quantizers_output.write((char *)lq_quantizer_index[n_lq].nn_centroid_idxs[j].data(), nc_per_group * sizeof(idx_t));
                }
                for (size_t j = 0; j < nc_upper; j++){
                    assert(lq_quantizer_index[n_lq].nn_centroid_dists[j].size() == nc_per_group);
                    quantizers_output.write((char * )lq_quantizer_index[n_lq].nn_centroid_dists[j].data(), nc_per_group * sizeof(float));
                }
                assert(n_vq + n_lq == i);
                n_lq ++;
            }
            else{
                std::cout << "Index type error: " << index_type[i] << std::endl;
                exit(0);
            }
        }
        quantizers_output.close();
    }

    void Bslib_Index::read_quantizers(const char * path_quantizer){
        std::cout << "Reading quantizers " << std::endl;
        std::ifstream quantizer_input(path_quantizer, std::ios::binary);

        //For each layer, there is nc, nc_upper and nc_per_group
        size_t nc;
        size_t nc_upper;
        size_t nc_per_group;

        for(size_t i = 0; i < this->layers; i++){
            if (index_type[i] == "VQ"){
                std::cout << "Reading VQ layer" << std::endl;
                quantizer_input.read((char *) & nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & nc_per_group, sizeof(size_t));
                std::cout << nc << " " << nc_upper << " " << nc_per_group << " " << std::endl;
                assert(nc_per_group * nc_upper == nc);
                VQ_quantizer vq_quantizer = VQ_quantizer(this->dimension, nc_upper, nc_per_group);
                std::vector<float> centroids(nc_per_group * this->dimension);
                for (size_t j = 0; j < nc_upper; j++){
                    quantizer_input.read((char * ) centroids.data(), nc_per_group * this->dimension * sizeof(float));
                    faiss::IndexFlatL2 centroid_quantizer(dimension);
                    centroid_quantizer.add(nc_per_group, centroids.data());
                    vq_quantizer.quantizers.push_back(centroid_quantizer);
                }
                this->vq_quantizer_index.push_back(vq_quantizer);
            }

            else if (index_type[i] == "LQ"){
                std::cout << "Reading LQ layer " << std::endl;
                quantizer_input.read((char *) & nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & nc_per_group, sizeof(size_t));

                assert(nc_per_group * nc_upper == nc);
                std::cout << nc << " " << nc_upper << " " << nc_per_group << " " << std::endl;
                std::vector<float> alphas(nc_upper);
                std::vector<float> upper_centroids(nc_upper * dimension);
                std::vector<idx_t> nn_centroid_idxs(nc_upper * nc_per_group);
                std::vector<float> nn_centroid_dists(nc_upper * nc_per_group);

                quantizer_input.read((char *) alphas.data(), nc_upper * sizeof(float));
                quantizer_input.read((char *) upper_centroids.data(), nc_upper * this->dimension * sizeof(float));
                quantizer_input.read((char *) nn_centroid_idxs.data(), nc_upper * nc_per_group * sizeof(idx_t));
                quantizer_input.read((char *) nn_centroid_dists.data(), nc_upper * nc_per_group * sizeof(float));

                LQ_quantizer lq_quantizer = LQ_quantizer(dimension, nc_upper, nc_per_group, upper_centroids.data(), nn_centroid_idxs.data(), nn_centroid_dists.data());
                for (size_t j = 0; j < nc_upper; j++){
                    lq_quantizer.alphas[j] = alphas[j];
                }
                this->lq_quantizer_index.push_back(lq_quantizer);
            }
        }
        quantizer_input.close();
    }

    void Bslib_Index::write_index(const char * path_index){
        std::ofstream output(path_index, std::ios::binary);
        output.write((char *) & this->final_nc, sizeof(size_t));
        assert((base_norm_codes.size() == base_codes.size()) && (base_codes.size() == origin_ids.size()) && (origin_ids.size() == final_nc )  );
        for (size_t i = 0; i < this->final_nc; i++){
            assert((base_norm_codes[i].size() == base_codes[i].size() / this->code_size) && (base_norm_codes[i].size() == origin_ids[i].size()));
            size_t group_size = base_norm_codes[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) base_norm_codes[i].data(), group_size * sizeof(uint8_t));
        }

        output.write((char *) & this->final_nc, sizeof(size_t));
        for (size_t i = 0; i < this->final_nc; i++){
            size_t group_size = base_codes[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) base_codes[i].data(), group_size * sizeof(uint8_t));
        }

        output.write((char *) & this->final_nc, sizeof(size_t));
        for (size_t i = 0; i < this->final_nc; i++){
            size_t group_size = origin_ids[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) origin_ids[i].data(), group_size * sizeof(idx_t));
        }
        output.write((char *) & this->final_nc, sizeof(size_t));
        assert(centroid_norms.size() == this->final_nc);
        output.write((char *) centroid_norms.data(), this->final_nc * sizeof(float));
        assert(centroid_norm_codes.size() == this->final_nc * this->norm_code_size);
        output.write((char *) centroid_norm_codes.data(), this->final_nc * this->norm_code_size * sizeof(uint8_t));
        output.close();
    }


    void Bslib_Index::read_index(const char * path_index){
        std::ifstream input(path_index, std::ios::binary);
        size_t final_nc_input;
        size_t group_size_input;

        this->base_codes.resize(this->final_nc);
        this->base_norm_codes.resize(this->final_nc);
        this->origin_ids.resize(this->final_nc);

        this->centroid_norms.resize(this->final_nc);
        this->centroid_norm_codes.resize(this->final_nc * this->norm_code_size);

        input.read((char *) & final_nc_input, sizeof(size_t));
        assert(final_nc_input == this->final_nc);
        for (size_t i = 0; i < this->final_nc; i++){
            input.read((char *) & group_size_input, sizeof(size_t));
            this->base_norm_codes[i].resize(group_size_input);
            input.read((char *) base_norm_codes[i].data(), group_size_input * sizeof(uint8_t));
        }
        std::cout << std::endl;


        input.read((char *) & final_nc_input, sizeof(size_t));
        assert(final_nc_input == this->final_nc);
        for (size_t i = 0; i < this->final_nc; i++){
            input.read((char *) & group_size_input, sizeof(size_t));
            base_codes[i].resize(group_size_input);
            input.read((char *) base_codes[i].data(), group_size_input * sizeof(uint8_t));
        }

        input.read((char *) & final_nc_input, sizeof(size_t));
        assert(final_nc_input == this->final_nc);
        for (size_t i = 0; i < this->final_nc; i++){
            input.read((char *) & group_size_input, sizeof(size_t));
            origin_ids[i].resize(group_size_input);
            input.read((char *) origin_ids[i].data(), group_size_input * sizeof(idx_t));
        }

        input.read((char *) & final_nc_input, sizeof(size_t));
        assert(final_nc_input == this->final_nc);
        input.read((char *) centroid_norms.data(), this->final_nc * sizeof(float));
        input.read((char *) centroid_norm_codes.data(), this->final_nc * this->norm_code_size * sizeof(uint8_t));
        input.close();
    }
}



/***********************************/
/*
size_t nb = 1000000;
std::vector<float> base_dataset(dimension * nb);
std::ifstream base_input("/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_base.fvecs", std::ios::binary);
readXvecFvec<float>(base_input, base_dataset.data(), dimension, nb);

std::cout << "Checking query centroid distance: " << std::endl;
std::vector<float> group_centroid(dimension);
this->lq_quantizer_index[0].compute_final_centroid(group_id, group_centroid.data());
std::vector<float> distance_vector(dimension);
faiss::fvec_madd(dimension, group_centroid.data(), -1 ,query, distance_vector.data());
std::cout << "Using: " << q_c_dist << " Groundtruth: " << faiss::fvec_norm_L2sqr(distance_vector.data(), dimension) << std::endl;

std::cout << "Checking centroid norm distance: " << std::endl;
std::cout << "Using: " << centroid_norms[group_id] << " Groundtruth: " << faiss::fvec_norm_L2sqr(group_centroid.data(), dimension) << std::endl;

std::cout << "Checking base norm distance: " << std::endl;
float * base_vector = base_dataset.data() + origin_ids[group_id][m] * dimension;
std::cout << "Using: " << base_norms[m] << " Groundtruth: " << faiss::fvec_norm_L2sqr(base_vector, dimension) << std::endl;

std::cout << "Checking final distance: " << std::endl;
faiss::fvec_madd(dimension, base_vector, -1, query, distance_vector.data());
std::cout << "Using: " << dist << " Groundtruth: " << faiss::fvec_norm_L2sqr(distance_vector.data(), dimension) << std::endl;
std::cout << "The proportion of error on final distance: " << (faiss::fvec_norm_L2sqr(distance_vector.data(), dimension) - dist) / dist << std::endl;

if (m == 10)
    exit(0);
    */
/***********************************/


/*
            std::cout << "The assigned cluster and the centroids are: " << std::endl;
            for (size_t temp = 0; temp < keep_result_space; temp++){
                std::cout << group_idxs[temp] << " " << group_dists[temp] << " ";
            }
            std::cout << std::endl << std::endl;

            faiss::IndexFlatL2 final_cluster_quantizer(dimension);
            std::vector<float> centroid(dimension * this->final_nc);
            for (size_t temp = 0; temp < this->final_nc; temp++){
                this->lq_quantizer_index[0].compute_final_centroid(temp, centroid.data() + temp * dimension);
            }
            final_cluster_quantizer.add(this->final_nc, centroid.data());
            std::vector<float> final_query_distance(keep_result_space);
            std::vector<faiss::Index::idx_t> final_query_group_id(keep_result_space);
            final_cluster_quantizer.search(1, query, keep_result_space, final_query_distance.data(), final_query_group_id.data());

            for (size_t temp = 0; temp < keep_result_space; temp++){
                std::cout << final_query_group_id[temp] << " " << final_query_distance[temp] << " ";
            }
            std::cout << std::endl << std::endl;
            
            size_t ngt = 100;
            std::cout << "Find where are the top 10 groundtruth: " << std::endl;
            for (size_t temp = 0; temp < 10; temp ++){
                uint32_t groundtruth_label = groundtruth[i * ngt + temp];
                for (size_t j = 0; j < this->final_nc; j++){
                    for (size_t m = 0; m < this->origin_ids[j].size(); m++){
                        if (this->origin_ids[j][m] == groundtruth_label)
                            std::cout << j << " ";
                    }
                }
            }
            exit(0);
*/


// The original search function with maxheap
/*
    void Bslib_Index::search(size_t n, size_t result_k, float * queries, float * query_dists, faiss::Index::idx_t * query_ids, size_t * keep_space, uint32_t * groundtruth){
        float overall_proportion = 0;
        float avg_visited_vectors = 0;
        std::vector<float> avg_time_consumption(layers+1);
        for (size_t i = 0; i < layers+1; i++){avg_time_consumption[i] = 0;}
        float avg_query_centroid_dist = 0;

//#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> time_consumption(layers+1);
            time_recorder Trecorder = time_recorder();

            std::ifstream base_input("/home/y/yujianfu/ivf-hnsw/data/SIFT1M/sift_base.fvecs", std::ios::binary);
            
            std::unordered_set<uint32_t> grountruth_set;
            for (size_t j = 0; j < result_k; j++)
                grountruth_set.insert(groundtruth[i * 100 + j]);

            const float * query = queries + i * dimension;

            std::vector<idx_t> group_idxs(1);
            std::vector<float> group_dists(1);

            group_idxs[0] = 0;
            group_dists[0] = 0;

            std::vector<float> result_dists; 
            std::vector<idx_t> result_labels;
            std::vector<std::map<idx_t, float>> query_upper_centroid_dists(1);
            
            size_t n_vq = 0;
            size_t n_lq = 0;

            size_t keep_result_space = 1;

            for (size_t j = 0; j < layers; j++){

                keep_result_space = group_idxs.size() * keep_space[j];
                assert(n_vq+ n_lq == j);
                size_t group_size;
                
                if (index_type[j] == "VQ"){
                    //std::cout << "searching in VQ layer" << std::endl;
                    group_size = vq_quantizer_index[n_vq].nc_per_group;
                    result_dists.resize(group_idxs.size()*group_size);
//#pragma omp parallel for
                    for (size_t m = 0; m < group_idxs.size(); m++){
                        vq_quantizer_index[n_vq].search_in_group(1, query, group_idxs.data()+m, result_dists.data()+m*group_size);
                    }
                    result_labels.resize(group_idxs.size()*group_size);
                    for(size_t m = 0; m < group_idxs.size(); m++){
                        for (size_t n = 0; n < group_size; n++){
                            result_labels[m * group_size + n] = group_idxs[m] * group_size + n;
                        }
                    }
                    n_vq ++;
                }

                else if(index_type[j] == "LQ") {
                    //std::cout << "searching in LQ layer" << std::endl;
                    group_size = lq_quantizer_index[n_lq].nc_per_group;
                    result_dists.resize(group_size * n);
                    assert(query_upper_centroid_dists[0].size() > 0);
//#pragma omp parallel for
                    for (size_t m = 0; m < group_idxs.size(); m++){
                        lq_quantizer_index[n_lq].search_in_group(1, query, query_upper_centroid_dists, group_idxs.data()+m, result_dists.data()+m*group_size);
                    }
                    result_labels.resize(group_idxs.size()*group_size);
                    for (size_t m = 0; m < group_idxs.size(); m++){
                        for (size_t n = 0; n < group_size; n++){
                            result_labels[m * group_size + n] = group_idxs[m] * group_size + n;
                        }
                    }
                    query_upper_centroid_dists[0].clear();
                    n_lq ++;
                }
                else{
                    std::cout << "The type name is wrong with " << index_type[j] << "!" << std::endl;
                    exit(0);
                }
                
                if (j < this->layers-1 && index_type[j+1] == "LQ"){
                    //std::cout << "The next layer is LQ, load the query centroid distsnaces" << std::endl;
                    for (size_t m = 0; m < group_idxs.size()*group_size; m++){
                        query_upper_centroid_dists[0].insert(std::pair<idx_t, float>(result_labels[m], result_dists[m]));
                    }
                }

                //std::cout << "Choosing k instances with smallest distances " << std::endl;
                size_t search_space = group_size * group_idxs.size();
                group_idxs.resize(keep_result_space);
                group_dists.resize(keep_result_space);

                keep_k_min(search_space, keep_result_space, result_dists.data(), result_labels.data(), group_dists.data(), group_idxs.data());
                time_consumption[j] = Trecorder.getTimeConsumption() / (search_space / group_size);
                Trecorder.reset();
            }

            //std::cout << "Finished assigned query data, start computing the distance to base vectors" << std::endl;
            assert((n_vq + n_lq) == this->layers);
            this->pq.compute_inner_prod_table(query, this->precomputed_table.data());
            size_t visited_vectors = 0;
            std::vector<float> query_search_dists(result_k);
            std::vector<faiss::Index::idx_t> query_search_labels(result_k);
            faiss::maxheap_heapify(result_k, query_search_dists.data(), query_search_labels.data());

            std::vector<float> actual_search_dists(result_k);
            std::vector<faiss::Index::idx_t> actual_search_labels(result_k);
            faiss::maxheap_heapify(result_k, actual_search_dists.data(), actual_search_labels.data());
            
            float query_centroid_dists = 0;
            for (size_t j = 0; j < keep_result_space; j++){
                query_centroid_dists += group_dists[j];
            }
            query_centroid_dists = query_centroid_dists / keep_result_space;
            avg_query_centroid_dist += query_centroid_dists;

            size_t visited_gt = 0;
            for (size_t j = 0; j < keep_result_space; j++){

                std::vector<float> result_idx_dist(2);
                get_next_group_idx(keep_result_space, group_idxs.data(), group_dists.data(), result_idx_dist.data());
                
                uint32_t group_id = result_idx_dist[0];
                float q_c_dist = result_idx_dist[1];

                size_t group_size = this->origin_ids[group_id].size();
                assert(group_size == this->base_codes[group_id].size() / this->code_size);

                float term1 = q_c_dist - centroid_norms[group_id];


                std::vector<float> base_norms(group_size);
                assert(group_size == base_norm_codes[group_id].size() / this->norm_code_size);
                this->norm_pq.decode(base_norm_codes[group_id].data(), base_norms.data(), group_size);
                const uint8_t * code = base_codes[group_id].data();


                for (size_t m = 0; m < group_size; m++){
                    float term2 = base_norms[m];
                    float term3 = 2 * pq_L2sqr(code + m * code_size);
                    float dist = term1 + term2 - term3;


                    std::vector<float> base_vector(dimension);
                    uint32_t dim;
                    base_input.seekg(origin_ids[group_id][m] * dimension * sizeof (float) + origin_ids[group_id][m] * sizeof(uint32_t), std::ios::beg);
                    base_input.read((char *) & dim, sizeof(uint32_t));
                    assert(dim == this->dimension);
                    base_input.read((char *) base_vector.data(), sizeof(float)*dimension);

                    std::vector<float> distance_vector(dimension);
                    faiss::fvec_madd(dimension, base_vector.data(), -1, query, distance_vector.data());
                    float actual_dist =  faiss::fvec_norm_L2sqr(distance_vector.data(), dimension);
                    if (actual_dist < actual_search_dists[0]){
                        faiss::maxheap_pop(result_k, actual_search_dists.data(), actual_search_labels.data());
                        faiss::maxheap_push(result_k, actual_search_dists.data(), actual_search_labels.data(), actual_dist, this->origin_ids[group_id][m]);
                    }


                    if (grountruth_set.count(this->origin_ids[group_id][m]) != 0)
                        visited_gt ++;
                    if (dist < query_search_dists[0]){
                        faiss::maxheap_pop(result_k, query_search_dists.data(), query_search_labels.data());
                        faiss::maxheap_push(result_k, query_search_dists.data(), query_search_labels.data(), dist, this->origin_ids[group_id][m]);
                    }
                    std::cout << this->origin_ids[group_id][m] << " " << dist << " " << actual_dist << " ";
                }
                visited_vectors += group_size;
                if (visited_vectors > this->max_visited_vectors)
                    break;
            }

            
            std::cout << "The searching results: " << std::endl;
            for (size_t j = 0; j < result_k; j ++){
                std::cout << actual_search_labels[j] << " " <<  actual_search_dists[j] << ",";
            }
            std::cout << std::endl;
            exit(0);
            
            
            for (size_t j = 0; j < result_k; j++){
                query_dists[i * result_k + j] = query_search_dists[j];
                query_ids[i * result_k + j] = query_search_labels[j];
            }
            
            
           overall_proportion += float(visited_gt) / result_k;
            time_consumption[this->layers]  = Trecorder.getTimeConsumption();

            avg_visited_vectors += visited_vectors;
            for (size_t j = 0; j < layers + 1; j++){
                avg_time_consumption[j] += time_consumption[j];    
            }
        }

        std::cout << "The time consumption: ";
        for (size_t i = 0; i < layers+1; i++){
            std::cout << avg_time_consumption[i] / n << " ";
        }
        std::cout << std::endl;

        std::cout << "The average visited vectors: " << avg_visited_vectors / n << std::endl;
        std::cout << "The average query centroid distance: " << avg_query_centroid_dist / n << std::endl;
        std::cout << "The avarage groundtruth proportion is: " << overall_proportion / n << std::endl;
    }
*/