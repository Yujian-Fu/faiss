#include "bslib_index.h"

namespace bslib{

    Bslib_Index::Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type):
        dimension(dimension), layers(layers){
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
        std::cout << this->train_data_idxs[0] << " " << this->train_data_idxs[1] << " " << this->train_data_idxs[2] << std::endl;;
        vq_quantizer.build_centroids(this->train_data.data(), this->nt, this->train_data_idxs.data(), update_idxs);
        ShowMessage("Checking whether all vq centroids are added correctly");
        for (size_t i = 0; i < nc_upper; i++){
            assert(vq_quantizer.quantizers[i].xb.size() == nc_per_group * dimension);
        }
        if (update_idxs){
            assert(vq_quantizer.all_quantizer.xb.size() == vq_quantizer.nc * dimension);
        }
        this->vq_quantizer_index.push_back(vq_quantizer);
    }

    void Bslib_Index::add_lq_quantizer(size_t nc_upper, size_t nc_per_group, const float * upper_centroids, const idx_t * upper_nn_centroid_idxs, const float * upper_nn_centroid_dists, bool update_idxs){
        LQ_quantizer lq_quantizer (dimension, nc_upper, nc_per_group, upper_centroids, upper_nn_centroid_idxs, upper_nn_centroid_dists);
        ShowMessage("Building centroids for lq quantizer");
        lq_quantizer.build_centroids(this->train_data.data(), this->nt, this->train_data_idxs.data(), update_idxs);
        //Check whether all centroids are added in all_quantizer correctly 
        if (update_idxs){
            ShowMessage("Checking whether all lq centroids are added correctly");
            assert(lq_quantizer.all_quantizer.xb.size() == nc_upper * nc_per_group * dimension);
        }
        ShowMessage("Showing samples of alpha");
        for(size_t i = 0; i < 20; i++){
            std::cout << lq_quantizer.alphas[i] << " " << std::endl;
        }
        std::cout << std::endl;
        this->lq_quantizer_index.push_back(lq_quantizer);
    }



    void Bslib_Index::encode(size_t n, const float * encode_data, const idx_t * encoded_ids, float * encoded_data){
        if (index_type[layers-1] == "VQ"){
            vq_quantizer_index[vq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else if(index_type[layers-1] == "LQ"){
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

    void Bslib_Index::train_pq(const char * path_pq, const char * path_norm_pq, const char * path_learn){
        // If the train dataset is not loaded
        if (this->train_data.size() != this->nt * dimension){
            std::cout << "Load the train set for PQ training" << std::endl;
            bool use_subset = true;
            this->train_data.resize(this-> nt * dimension);
            std::ifstream learn_input(path_learn, std::ios::binary);
            readXvecFvec<uint8_t>(learn_input, this->train_data.data(), this->dimension, this->nt, true);
            this->train_data_idxs.resize(nt);
            for (size_t i = 0; i < nt; i++){
                this->train_data_idxs[i] = 0;
            }
            if (use_subset){
                assert (this->nt != this->subnt);
                std::vector<float> train_subset(subnt * dimension);
                RandomSubset(this->train_data.data(), train_subset.data(), dimension, this->nt, this->subnt);
                train_data.resize(subnt * dimension);
                train_data_idxs.resize(subnt);
                for (size_t i = 0; i < subnt * dimension; i++){
                    this->train_data[i] = train_subset[i];
                }
                for (size_t i = 0; i < subnt; i++){
                    this->train_data_idxs[i] = 0;
                }
                this->nt = this->subnt;
            }
        }
        std::cout << "Initilizing index " << std::endl;
        this->pq = faiss::ProductQuantizer(this->dimension, this->M, this->nbits);
        this->norm_pq = faiss::ProductQuantizer(1, this->norm_M, this->nbits);
        this->code_size = this->pq.code_size;
        this->norm_code_size = this->norm_pq.code_size;

        std::cout << "Encoding the train dataset to compute residual" << std::endl;
        std::vector<float> residuals(dimension * this->nt);
        std::vector<idx_t> group_ids(this->nt);
        assign(this->nt, this->train_data.data(), group_ids.data());

        encode(this->nt, this->train_data.data(), group_ids.data(), residuals.data());
        this->pq.verbose = true;
        this->pq.train(nt, residuals.data());
        faiss::write_ProductQuantizer(& this->pq, path_pq);

        std::vector<float> reconstructed_x(dimension * this->nt);
        decode(this->nt, residuals.data(), group_ids.data(), reconstructed_x.data());
        std::vector<float> xnorm(this->nt);
        for (size_t i = 0; i < this->nt; i++){
            xnorm[i] = faiss::fvec_norm_L2sqr(reconstructed_x.data() + i * dimension, dimension);
        }
        this->norm_pq.verbose = true;
        this->norm_pq.train(this->nt, xnorm.data());
        faiss::write_ProductQuantizer(& this->norm_pq, path_norm_pq);
    }

    void Bslib_Index::build_quantizers(const uint32_t * ncentroids, const char * path_quantizers, const char * path_learn){
        if (exists(path_quantizers)){
            read_quantizers(path_quantizers);
            std::cout << "Checking the quantizers read from file " << std::endl;
            std::cout << "The number of quantizers: " << this->vq_quantizer_index.size() << " " << this->lq_quantizer_index.size();
            std::cout  << "The number of centroids in each quantizer: ";
            for (size_t i = 0 ; i < this->vq_quantizer_index[0].quantizers.size(); i++){
                std::cout << vq_quantizer_index[0].quantizers[i].xb.size() << " ";
            }
            std::cout << std::endl << "The alpha in lq quantizer: ";
            for (size_t i = 0; i < this->lq_quantizer_index[0].alphas.size(); i++){
                std::cout << this->lq_quantizer_index[0].alphas[i] << " ";
            }
            std::cout << std::endl;

        }
        else{
        ShowMessage("No preconstructed quantizers, constructing quantizers");
        //Load the train set into the index
        bool use_subset = true;
        this->train_data.resize(this->nt * this->dimension);
        std::cout << "Loading learn set from " << path_learn << std::endl;
        std::ifstream learn_input(path_learn, std::ios::binary);
        readXvecFvec<uint8_t>(learn_input, this->train_data.data(), this->dimension, this->nt, true);
        this->train_data_idxs.resize(this->nt);
        for (size_t i = 0; i < this->nt; i++){
            this->train_data_idxs[i] = 0;
        }
        if (use_subset){
            ShowMessage("Using subset of the train set, saving sub train set");
            std::vector<float> train_subset(this->subnt * this->dimension);
            RandomSubset(this->train_data.data(), train_subset.data(), this->dimension, this->nt, this->subnt);
            train_data.resize(this->subnt * this->dimension);
            for (size_t i = 0; i < subnt * dimension; i++){
                this->train_data[i] = train_subset[i];
            }
            this->nt = subnt;
            CheckResult<float>(this->train_data.data(), this->dimension);
        }

        assert(index_type.size() == layers && index_type[0] != "LQ");
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
                std::cout << "VQ quantizer added, check it " << std::endl;
                std::cout << "The vq quantizer size is: " <<  vq_quantizer_index.size() << " the num of quantizers: " << vq_quantizer_index[vq_quantizer_index.size()-1].quantizers.size();
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

        std::vector<uint8_t> batch_codes(n * this->code_size);
        this->pq.compute_codes(residuals.data(), batch_codes.data(), n);

        std::vector<float> reconstructed_x(n * dimension);

        /*
        Todo: should we use the decoded reconstructed_x for exp? actually we may use
        the distance in full precision for exp?
        */
       
        decode(n, residuals.data(), encoded_ids, reconstructed_x.data());

        /*
        Use the origin distance can save time?
        */
        std::vector<float> xnorms (n);
        for (size_t i = 0; i < n; i++){
            xnorms[i] =  faiss::fvec_norm_L2sqr(data + i * dimension, dimension);
        }

        std::vector<uint8_t> xnorm_codes (n * norm_code_size);

        this->norm_pq.compute_codes(xnorms.data(), xnorm_codes.data(), n);
    
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
                centroid_norms[i] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
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
                centroid_norms[i] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
            }
            this->norm_pq.compute_codes(centroid_norms.data(), this->centroid_norm_codes.data(), this->final_nc);
        }
    }

    void Bslib_Index::assign(size_t n, const float * assign_data, idx_t * assigned_ids){

        std::vector<idx_t> group_id (n);
        for (size_t i = 0; i < n; i++){
            group_id[i] = 0;
        }

        size_t n_vq = 0;
        size_t n_lq = 0;

        std::vector<float> dists(n);
        std::vector<idx_t> labels(n);
        for (size_t i = 0; i < this->layers; i++){
            assert(n_lq + n_vq == i);
            if (index_type[i] == "VQ"){
                std::cout << "Searching in VQ layer for " << n << " data vectors " << std::endl;
                vq_quantizer_index[n_vq].search_in_group(n, assign_data, 1, dists.data(), labels.data(), group_id.data());
                n_vq ++;
            }

            else if(index_type[i] == "LQ"){
                std::cout << "Searching in LQ layer " << std::endl;
                lq_quantizer_index[n_lq].search_in_group(n, assign_data, 1, dists.data(), labels.data(), group_id.data());
                n_lq ++;
            }
            else{
                std::cout << "The type name is wrong with " << index_type[i] << "!" << std::endl;
                exit(0); 
            }
            for (size_t j = 0; j < n; j++){
                group_id[j] = labels[j];
                labels[j] = 0;
                dists[j] = 0;
            }
        }
        assert((n_vq + n_lq) == this->layers);
        for (size_t i = 0; i < n; i++){
            assigned_ids[i] = group_id[i];
        }
    }


    void Bslib_Index::keep_k_min(size_t m, size_t k, float * all_dists, idx_t * all_ids, float * sub_dists, idx_t * sub_ids){
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

            std::cout << "The kept choices " << std::endl;
            for (size_t i = 0; i < k; i++){
                sub_ids[i] = ids[i];
                sub_dists[i] = dists[i];
                std::cout << ids[i] << "_" << dists[i] << " ";
            }
            std::cout << std::endl;
        }
    }


    float Bslib_Index::pq_L2sqr(const uint8_t *code)
    {
        float result = 0.;
        for (size_t i = 0; i < this->code_size; i++) {
            result += precomputed_table[pq.ksub * i + code[i]];
        }
        return result;
    }

     /*
      *    d = || x - y_C - y_R ||^2
      *    d = || x - y_C ||^2 - || y_C ||^2 + || y_C + y_R ||^2 - 2 * (x|y_R)
      *        -----------------------------   -----------------   -----------
      *                     term 1                   term 2           term 3
      */

    void Bslib_Index::search(size_t n, size_t result_k, float * queries, float * query_dists, faiss::Index::idx_t * query_ids, size_t * search_space, size_t * keep_space){

        for (size_t i = 0; i < n; i++){
            faiss::maxheap_heapify(result_k, query_dists + i * result_k, query_ids + i * result_k);
            const float * query = queries + i * dimension;
            std::vector<idx_t> search_ids(1);
            std::vector<float> search_q_c_dists(1);
            search_ids[0] = 0;
            search_q_c_dists[0] = 0;

            std::vector<idx_t> result_ids;
            std::vector<float> resuld_q_c_dists; 

            size_t n_vq = 0;
            size_t n_lq = 0;

            size_t search_result_space;
            size_t keep_result_space;


            for (size_t j = 0; j < layers; j++){

                search_result_space = search_ids.size() * search_space[j];
                keep_result_space = search_ids.size() * keep_space[j];
                result_ids.resize(search_result_space);
                resuld_q_c_dists.resize(search_result_space);
                std::cout << "Parameters: " << search_result_space << " " << keep_result_space << std::endl;;

                if (index_type[j] == "VQ"){
                    for (size_t m = 0; m < search_ids.size(); m++){
                        vq_quantizer_index[n_vq].search_in_group(1, query, search_space[j], resuld_q_c_dists.data()+m * search_space[j], result_ids.data()+ m * search_space[j], search_ids.data()+m);
                    }
                    n_vq ++;
                }

                else{
                    for (size_t m = 0; m < search_ids.size(); m++){
                        lq_quantizer_index[n_lq].search_in_group(1, query, search_space[j], resuld_q_c_dists.data() + m*search_space[j], result_ids.data() + m * search_space[j], search_ids.data() + m);
                    }
                    n_lq ++;
                }
                
                search_q_c_dists.resize(keep_result_space);
                search_ids.resize(keep_result_space);
                keep_k_min(search_result_space, keep_result_space, resuld_q_c_dists.data(), result_ids.data(), search_q_c_dists.data(), search_ids.data());
            }
            
            assert((n_vq + n_lq) == this->layers);
            pq.compute_inner_prod_table(query, precomputed_table.data());
            
            for (size_t j = 0; j < keep_result_space; j++){
                size_t group_id = search_ids[j];
                float q_c_dist = search_q_c_dists[j];
                size_t group_size = this->origin_ids[group_id].size();

                float term1 = q_c_dist - centroid_norms[group_id];

                std::vector<float> base_norms(group_size);
                this->norm_pq.decode(base_norm_codes[group_id].data(), base_norms.data());
                const uint8_t * code = base_codes[group_id].data();

                for (size_t m = 0; m < group_size; m++){
                    float term2 = base_norms[m];
                    float term3 = 2 * pq_L2sqr(code + j * code_size);
                    float dist = term1 + term2 - term3;
                    if (dist < query_dists[i * result_k]){
                        faiss::maxheap_heapify(result_k, query_dists + i * result_k, query_ids + i * result_k);
                        faiss::maxheap_push(result_k, query_dists + i * result_k, query_ids + i * result_k, dist, this->origin_ids[group_id][m]);
                    }
                }
            }
        }
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
                    for (size_t id = 0; id < 10; id++){
                        for (size_t temp = 0; temp < dimension; temp++){
                            std::cout << vq_quantizer.quantizers[0].xb[id * dimension + temp] << " ";
                        }
                        std::cout << std::endl;
                    }
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