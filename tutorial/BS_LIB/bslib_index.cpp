#include "bslib_index.h"
#include "time.h"

namespace bslib{

    /**
     * The initialize function for BSLIB struct 
     **/
    Bslib_Index::Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type, const bool use_HNSW_VQ,  const bool use_norm_quantization):
        dimension(dimension), layers(layers){

            this->use_HNSW_VQ = use_HNSW_VQ;
            this->use_HNSW_group = use_HNSW_group;
            this->use_norm_quantization = use_norm_quantization;
            this->use_OPQ = use_OPQ;

            this->index_type.resize(layers);
            this->ncentroids.resize(layers);

            for (size_t i = 0; i < layers; i++){
                this->index_type[i] = index_type[i];
            }
            this->train_size = 0;
            this->use_hash = true ? index_type[layers - 1] == "PQ" : false;
        }
    
    /**
     * The function for adding a VQ layer in the whole structure
     * 
     * Parameters required for building a VQ layer: 
     * If use L2 quantizer: nc_upper, nc_group 
     * Use HNSW quantizer: nc_upper, nc_group, M, efConstruction, efSearch
     * 
     **/
    void Bslib_Index::add_vq_quantizer(size_t nc_upper, size_t nc_per_group, size_t M = 16, size_t efConstruction = 500, size_t efSearch = 100){
        
        VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, nc_per_group, M, efConstruction, efSearch, use_HNSW_VQ);
        PrintMessage("Building centroids for vq quantizer");
        vq_quantizer.build_centroids(this->train_data.data(), this->train_data.size() / dimension, this->train_data_ids.data());
        PrintMessage("Finished construct the VQ layer");
        
        this->vq_quantizer_index.push_back(vq_quantizer);
    }

    /**
     * The function for adding a LQ layer in the whole structure
     * 
     * Parameters required for building a LQ layer: 
     * nc_upper, nc_per_group, upper_centroids, upper_nn_centroid_idxs, upper_nn_centroid_dists
     * 
     * upper_centroids: the upper centroids data   size: nc_upper * dimension
     * upper_nn_centroid_idxs: the upper centroid neighbor idxs      size: nc_upper * nc_per_group 
     * upper_nn_centroid_dists: the upper centroid neighbor dists    size: nc_upper * nc_per_group
     * 
     **/
    void Bslib_Index::add_lq_quantizer(size_t nc_upper, size_t nc_per_group, const float * upper_centroids, const idx_t * upper_nn_centroid_idxs, const float * upper_nn_centroid_dists){
        LQ_quantizer lq_quantizer = LQ_quantizer(dimension, nc_upper, nc_per_group, upper_centroids, upper_nn_centroid_idxs, upper_nn_centroid_dists);
        PrintMessage("Building centroids for lq quantizer");
        lq_quantizer.build_centroids(this->train_data.data(), this->train_data.size() / dimension, this->train_data_ids.data());
        PrintMessage("Finished construct the LQ layer");
        this->lq_quantizer_index.push_back(lq_quantizer);
    }

    /**
     * The function for adding a PQ layer in the whole structure
     * 
     * Parameter required for building a PQ layer
     * nc_upper, M, nbits
     * 
     **/
    void Bslib_Index::add_pq_quantizer(size_t nc_upper, size_t M, size_t nbits){
        PQ_quantizer pq_quantizer = PQ_quantizer(dimension, nc_upper, M, nbits);
        PrintMessage("Building centroids for pq quantizer");
        pq_quantizer.build_centroids(this->train_data.data(), this->train_data_ids.size(), this->train_data_ids.data());
        this->pq_quantizer_index.push_back(pq_quantizer);
    }

    /**
     * This is the function for encoding the origin base vectors into residuals to the centroids
     * 
     * Input:
     * encode_data: the vector to be encoded.    size: n * dimension
     * encoded_ids: the group id of the vectors  size: n
     * 
     * Output:
     * encoded_data: the encoded data            size: n *  dimension
     * 
     **/
    void Bslib_Index::encode(size_t n, const float * encode_data, const idx_t * encoded_ids, float * encoded_data){
        if (index_type[layers-1] == "VQ"){
            vq_quantizer_index[vq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else if(index_type[layers-1] == "LQ"){
            lq_quantizer_index[lq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else if(index_type[layers - 1] == "PQ"){
            pq_quantizer_index[pq_quantizer_index.size() -1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else{
            std::cout << "The type name is wrong with " << index_type[layers - 1] << "!" << std::endl;
            exit(0);
        }
    }

    /**
     * This is the function for decode the vector data
     * 
     * Input:
     * encoded_data: the encoded data                  size: n * dimension
     * encoded_ids:  the group id of the encoded data  size: n
     * 
     * Output:
     * decoded_data: the reconstruction data            size: n * dimension
     * 
     **/
    void Bslib_Index::decode(size_t n, const float * encoded_data, const idx_t * encoded_ids, float * decoded_data){
        if (index_type[layers-1] == "VQ"){
            vq_quantizer_index[vq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data);
        }
        else if (index_type[layers-1] == "LQ"){
            lq_quantizer_index[lq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data);
        }
        else if (index_type[layers-1] == "PQ"){
            pq_quantizer_index[pq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data);
        }
        else{
            std::cout << "The type name is wrong with " << index_type[layers - 1] << "!" << std::endl;
            exit(0);
        }
    }


    /**
     * 
     * This is the function for building centroids for train set and then select sub train set evenly
     * 
     * Input:
     * path_learn: the path for learn set
     * path_groups: the path for generated centroids
     * path_labels: the path for assigned labels
     * 
     * total_train_size: the total size of learn set
     * sub_train_size: the size of subset size
     * group_size: the size of centroids
     * 
     **/
    void Bslib_Index::build_train_selector(const std::string path_learn, const std::string path_groups, const std::string path_labels, size_t total_train_size, size_t sub_train_size, size_t group_size){
        PrintMessage("Building train dataset selector for further train tasks");
        if (exists(path_labels)){
            PrintMessage("Selector already constructed, load it");
            std::ifstream labels_input(path_labels, std::ios::binary);
            assert(this->train_set_ids.size() == 0);
            this->train_set_ids.resize(group_size);
            size_t num_per_group;
            for (size_t i = 0; i < group_size; i++){
                labels_input.read((char *) & num_per_group, sizeof(size_t));
                this->train_set_ids[i].resize(num_per_group);
                labels_input.read((char *) train_set_ids[i].data(), num_per_group * sizeof(idx_t));
            }
            PrintMessage("Finished reading the selector");
        }
        else{
            std::vector<float> origin_train_set(total_train_size * dimension);
            std::ifstream learn_input(path_learn, std::ios::binary);
            readXvecFvec<learn_data_type>(learn_input, origin_train_set.data(), dimension, total_train_size, true);
            std::vector<float> train_set_centroids(group_size * dimension);

            if (sub_train_size < total_train_size){
                std::vector<float> sub_train_set(sub_train_size * dimension);
                RandomSubset(origin_train_set.data(), sub_train_set.data(), dimension, total_train_size, sub_train_size);
                faiss::kmeans_clustering(dimension, sub_train_size, group_size, sub_train_set.data(), train_set_centroids.data());
            }
            else{
                faiss::kmeans_clustering(dimension, total_train_size, group_size, origin_train_set.data(), train_set_centroids.data());
            }

            
            faiss::IndexFlatL2 centroid_index(dimension);
            centroid_index.add(group_size, train_set_centroids.data());

            std::vector<idx_t> result_ids(total_train_size);
            std::vector<float> result_dists(total_train_size);
            centroid_index.search(total_train_size, origin_train_set.data(), 1, result_dists.data(), result_ids.data());

            this->train_set_ids.resize(group_size);
            for (size_t i = 0; i < total_train_size; i++){this->train_set_ids[result_ids[i]].push_back(i);}

            std::cout << "Saving group centroids to " << path_groups << std::endl;
            std::cout << "Saving IDs to " << path_labels << std::endl;
            std::ofstream groups_output(path_groups, std::ios::binary);
            std::ofstream labels_output(path_labels, std::ios::binary);
            for (size_t i = 0; i < group_size; i++){
                groups_output.write((char *) & dimension, sizeof(uint32_t)); 
                groups_output.write((char *) train_set_centroids.data() + i * dimension, dimension * sizeof(float));

                size_t num_per_group = this->train_set_ids[i].size();
                std::cout << num_per_group << " ";
                labels_output.write((char *) & num_per_group, sizeof(size_t));
                labels_output.write((char *) train_set_ids[i].data(), num_per_group * sizeof(idx_t));
            }
            std::cout << std::endl;
        }
    }


    /**
     * 
     * This is the function for selecting subset of the origin set
     * 
     * Input:
     * path_learn: this is the path for learn set
     * total_size: the size of learn set size 
     * train_set_size: the size to be read 
     * 
     **/
    void Bslib_Index::read_train_set(const std::string path_learn, size_t total_size, size_t train_set_size){
        std::cout << "Reading " << train_set_size << " from " << total_size << " for training" << std::endl;
        this->train_data.resize(train_set_size * dimension);
        this->train_data_ids.resize(train_set_size, 0);

        if (total_size == train_set_size){
            std::ifstream learn_input(path_learn, std::ios::binary);
            readXvecFvec<float>(learn_input, this->train_data.data(), dimension, train_set_size, true, false);
        }
        else{
            assert(this->train_set_ids.size() > 0);
            size_t group_size = train_set_ids.size();
            srand((unsigned)time(NULL));
            std::ifstream learn_input(path_learn, std::ios::binary);
            uint32_t dim;

            for (size_t i = 0; i < train_set_size; i++){
                size_t group_id = i % group_size;
                size_t inner_group_id = rand() % this->train_set_ids[group_id].size();
                idx_t sequence_id = this->train_set_ids[group_id][inner_group_id];

                learn_input.seekg(sequence_id * dimension * sizeof (float) + sequence_id * sizeof(uint32_t), std::ios::beg);
                learn_input.read((char *) & dim, sizeof(uint32_t));
                assert(dim == dimension);
                learn_input.read((char *)this->train_data.data() + i * dimension, dimension * sizeof(float));
                
            }
        }
    }

    /**
     * The function for building quantizers in the whole structure
     * 
     * Input: 
     * 
     * ncentroids: number of centroids in all layers  size: layers
     * Note that the last ncentroids para for PQ layer is not used, it should be placed in PQ_paras 
     * path_quantizer: path for saving or loading quantizer
     * path_learn: path for learning dataset
     * n_train: the number of train vectors to be use in all layers     size: layers
     * 
     * HNSW_paras: the parameters for constructing HNSW graph
     * PQ_paras: the parameters for constructing PQ layer
     * 
     * 
     **/
    void Bslib_Index::build_quantizers(const uint32_t * ncentroids, const std::string path_quantizer, const std::string path_learn, const size_t * num_train, const std::vector<HNSW_para> HNSW_paras, const std::vector<PQ_para> PQ_paras){
        if (exists(path_quantizer)){
            read_quantizers(path_quantizer);
            std::cout << "Checking the quantizers read from file " << std::endl;
            std::cout << "The number of quantizers: " << this->vq_quantizer_index.size() << " " << this->lq_quantizer_index.size() << " " << this->pq_quantizer_index.size() << std::endl;

            /*
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

            

            this->max_group_size = 10000;
           std::cout << "reading the VQ centroids " << std::endl;
           std::ifstream quantizer_input(path_quantizer, std::ios::binary);
            VQ_quantizer vq_quantizer = VQ_quantizer(this->dimension, 1, 10000);
            std::vector<float> centroids(10000 * this->dimension);
            readXvec<float>(quantizer_input, centroids.data(), dimension, 10000, true);
            faiss::IndexFlatL2 * centroid_quantizer = new faiss::IndexFlatL2(dimension);
            centroid_quantizer->add(10000, centroids.data());
            vq_quantizer.L2_quantizers.push_back(centroid_quantizer);
            this->vq_quantizer_index.push_back(vq_quantizer);
            
            std::cout << this->vq_quantizer_index[0].L2_quantizers.size() << " " << this->vq_quantizer_index[0].L2_quantizers[0]->xb.size() << " " << std::endl;
            
            size_t group_size = this->train_data.size() / dimension;
            std::vector<float> centroid_distances(group_size);
            std::vector<idx_t> centroid_idxs(group_size);

            this->vq_quantizer_index[0].L2_quantizers[0]->search(group_size, this->train_data.data(), 1, centroid_distances.data(), centroid_idxs.data());
            for (size_t j = 0; j < group_size; j++){
                this->train_data_idxs[j] = centroid_idxs[j];
            }


            uint32_t nc_per_group = ncentroids[1];
            std::vector<float> upper_centroids (ncentroids[0]*dimension);
            std::vector<idx_t> nn_centroids_idxs(ncentroids[0]*ncentroids[1]);
            std::vector<float> nn_centroids_dists(ncentroids[0]*ncentroids[1]);
            this->vq_quantizer_index[0].compute_nn_centroids(nc_per_group, upper_centroids.data(), nn_centroids_dists.data(), nn_centroids_idxs.data());
            std::cout << "Adding lq quantizer" << std::endl;
            add_lq_quantizer(ncentroids[0], ncentroids[1], upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data(), false);
            for (size_t i = 0; i < this->lq_quantizer_index[0].alphas.size(); i++){
                std::cout << lq_quantizer_index[0].alphas[i] << " ";
            }
            std::cout << std::endl;
            */
        }

        else
        {
        PrintMessage("No preconstructed quantizers, constructing quantizers");
        //Load the train set into the index
        assert(index_type.size() == layers && index_type[0] != "LQ");
        std::cout << "adding layers to the index structure " << std::endl;
        this->max_group_size = 0;
        uint32_t nc_upper = 1; 
        uint32_t nc_per_group;
        std::vector<float> upper_centroids;
        std::vector<idx_t> nn_centroids_idxs;
        std::vector<float> nn_centroids_dists;

        //Prepare train set for initialization
        read_train_set(path_learn, this->train_size, num_train[0]);

        for (size_t i = 0; i < layers; i++){
            
            bool update_ids = (i == layers-1) ? false:true;
            nc_per_group = index_type[i] == "PQ" ? 0 : ncentroids[i];
            this->ncentroids[i] = nc_per_group;
            if (nc_per_group > this->max_group_size){this->max_group_size = nc_per_group;}

            if (index_type[i] == "VQ"){
                
                std::cout << "Adding VQ quantizer with parameters: " << nc_upper << " " << nc_per_group << std::endl;
                if (use_HNSW_VQ){
                    size_t existed_VQ_layers = this->vq_quantizer_index.size();
                    HNSW_para para = HNSW_paras[existed_VQ_layers];
                    add_vq_quantizer(nc_upper, nc_per_group, para.first.first, para.first.second, para.second);
                }
                else{
                    add_vq_quantizer(nc_upper, nc_per_group);
                }
                
                //Prepare train set for the next layer
                if (update_ids){
                    read_train_set(path_learn, this->train_size, num_train[i+1]);
                    vq_quantizer_index[vq_quantizer_index.size() - 1].update_train_ids(train_data.data(), train_data_ids.data(), train_data_ids.size());
                }

                std::cout << i << "th VQ quantizer added, check it " << std::endl;
                std::cout << "The vq quantizer size is: " <<  vq_quantizer_index.size() << " the num of L2 quantizers (groups): " << vq_quantizer_index[vq_quantizer_index.size()-1].L2_quantizers.size() << std::endl;
            }
            else if(index_type[i] == "LQ"){
                assert (i >= 1);

                upper_centroids.resize(nc_upper * dimension);
                nn_centroids_idxs.resize(nc_upper * nc_per_group);
                nn_centroids_dists.resize(nc_upper * nc_per_group);
                
                if (index_type[i-1] == "VQ"){
                    PrintMessage("Adding VQ quantizer with VQ upper layer");
                    size_t last_vq = vq_quantizer_index.size() - 1;
                    PrintMessage("VQ computing nn centroids");
                    assert(vq_quantizer_index[last_vq].nc > nc_per_group);
                    vq_quantizer_index[last_vq].compute_nn_centroids(nc_per_group, upper_centroids.data(), nn_centroids_dists.data(), nn_centroids_idxs.data());
                    add_lq_quantizer(nc_upper, nc_per_group, upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data());
                }
                else if (index_type[i-1] == "LQ"){
                    PrintMessage("Adding LQ quantizer with LQ upper layer");
                    size_t last_lq = lq_quantizer_index.size() - 1;
                    PrintMessage("LQ computing nn centroids");
                    assert(lq_quantizer_index[last_lq].nc > nc_per_group);
                    lq_quantizer_index[last_lq].compute_nn_centroids(nc_per_group, upper_centroids.data(), nn_centroids_dists.data(), nn_centroids_idxs.data());
                    add_lq_quantizer(nc_upper, nc_per_group, upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data());
                }

                //Prepare train set for the next layer
                if (update_ids){
                    read_train_set(path_learn, this->train_size, num_train[i+1]);
                    lq_quantizer_index[lq_quantizer_index.size() - 1].update_train_ids(train_data.data(), train_data_ids.data(), train_data_ids.size());
                }

                std::cout << i << "th LQ quantizer added, check it " << std::endl;
                std::cout << "The LQ quantizer size is: " <<  lq_quantizer_index.size() << " the num of alphas: " << lq_quantizer_index[lq_quantizer_index.size()-1].alphas.size() << std::endl;;
            }
            else if (index_type[i] == "PQ"){
                //The PQ layer should be placed in the last layer
                assert(i == layers-1);
                PrintMessage("Adding PQ quantizer");
                add_pq_quantizer(nc_upper, PQ_paras[0].first, PQ_paras[0].second);
                std::cout << i << "th PQ quantizer added, check it " << std::endl;
            }
            nc_upper  = nc_upper * nc_per_group;
        }
        write_quantizers(path_quantizer);
        }  
    }


    /**
     * 
     * This is the function for training PQ compresser 
     * 
     * Input:
     * 
     * path_pq: the path for saving and loading PQ quantizer for origin vector
     * path_norm_pq: the path for saving and loading PQ quantizer for norm value
     * path_learn: the path for learning dataset
     * 
     **/
    void Bslib_Index::train_pq(const std::string path_pq, const std::string path_norm_pq, const std::string path_learn, const size_t train_set_size){

        // Load the train set fot training
        read_train_set(path_learn, this->train_size, train_set_size);

        std::cout << "Initilizing index PQ quantizer " << std::endl;
        this->pq = faiss::ProductQuantizer(this->dimension, this->M, this->nbits);
        this->code_size = this->pq.code_size;

        std::cout << "Assigning the train dataset to compute residual" << std::endl;
        std::vector<float> residuals(train_set_size * dimension);

        assign(train_set_size, this->train_data.data(), train_data_ids.data());

        for (size_t i = 0; i < 100; i++){std::cout << train_data_ids[i] << " ";} std::cout << std::endl;
        
        std::cout << "Encoding the train dataset with " << train_set_size<< " data points " << std::endl;
        encode(train_set_size, this->train_data.data(), train_data_ids.data(), residuals.data());

        std::cout << "Training the pq " << std::endl;
        this->pq.verbose = true;
        this->pq.train(train_set_size, residuals.data());
        std::cout << "Writing PQ codebook to " << path_pq << std::endl;
        faiss::write_ProductQuantizer(& this->pq, path_pq.c_str());

        if (use_norm_quantization){
            std::cout << "Decoding the residual data and train norm product quantizer" << std::endl;
            std::vector<float> reconstructed_x(dimension * train_set_size);
            decode(train_set_size, residuals.data(), this->train_data_ids.data(), reconstructed_x.data());

            std::vector<float> xnorm(train_set_size);
            for (size_t i = 0; i < train_set_size; i++){
                xnorm[i] = faiss::fvec_norm_L2sqr(reconstructed_x.data() + i * dimension, dimension);
            }
            this->norm_pq = faiss::ProductQuantizer(1, this->norm_M, this->nbits);
            this->norm_code_size = this->norm_pq.code_size;

            std::cout << "Training the norm pq" << std::endl;
            this->norm_pq.verbose = true;
            this->norm_pq.train(train_set_size, xnorm.data());
            std::cout << "Writing norm_PQ codebook to " << path_norm_pq << std::endl;
            faiss::write_ProductQuantizer(& this->norm_pq, path_norm_pq.c_str());
        }
    }
    
    /**
     * 
     * This is the function for computing final centroids
     * 
     * The return value of this function is the total number of groups
     * 
     **/
    void Bslib_Index::get_final_group_num(){
        if (this->index_type[layers -1] == "VQ"){
            this->final_group_num =  vq_quantizer_index[vq_quantizer_index.size() -1].nc;
        }
        else if (this->index_type[layers -1] == "LQ"){
            this->final_group_num =  lq_quantizer_index[lq_quantizer_index.size() -1].nc;
        }
        else if (this->index_type[layers - 1] == "PQ"){
            this->final_group_num = pq_quantizer_index[pq_quantizer_index.size() -1].hash_size;
        }
    }


    /**
     * 
     * This is the function for adding a base batch 
     * 
     * Input:
     * n: the batch size of the batch data     size: size_t
     * data: the base data                     size: n * dimension
     * ids: the origin sequence id of data     size: n
     * encoded_ids: the group id of the data   size: n
     * 
     **/
    void Bslib_Index::add_batch(size_t n, const float * data, const idx_t * ids, idx_t * encoded_ids){
        std::vector<float> residuals(n * dimension);
        //Compute residuals
        encode(n, data, encoded_ids, residuals.data());


        //Compute code for residuals
        std::vector<uint8_t> batch_codes(n * this->code_size);
        this->pq.compute_codes(residuals.data(), batch_codes.data(), n);
        
        std::vector<idx_t> hash_ids;
        if (use_hash){hash_ids.resize(n, 0);HashMapping(n, encoded_ids, hash_ids.data(), final_group_num);}

        assert(this->base_codes.size() == this->final_group_num);

        //Add codes into index
        for (size_t i = 0 ; i < n; i++){
            idx_t group_id = use_hash ? hash_ids[i] : encoded_ids[i];
            for (size_t j = 0; j < this->code_size; j++){this->base_codes[group_id].push_back(batch_codes[i * this->code_size + j]);}
            this->base_sequence_ids[group_id].push_back(ids[i]);
            if (use_hash){this->base_pre_hash_ids[group_id].push_back(encoded_ids[i]);}
        }

        std::vector<float> decoded_residuals(n * dimension);
        this->pq.decode(batch_codes.data(), decoded_residuals.data(), n);
        
        std::vector<float> reconstructed_x(n * dimension);
        decode(n, decoded_residuals.data(), encoded_ids, reconstructed_x.data());

       //This is the norm for reconstructed vectors
        std::vector<float> xnorms (n);
        for (size_t i = 0; i < n; i++){
            xnorms[i] =  faiss::fvec_norm_L2sqr(reconstructed_x.data() + i * dimension, dimension);
        }

        //The size of base_norm_code or base_norm should be initialized in main function
        if (use_norm_quantization){
            assert(this->base_norm_codes.size() == this->final_group_num);
            std::vector<uint8_t> xnorm_codes (n * norm_code_size);
            this->norm_pq.compute_codes(xnorms.data(), xnorm_codes.data(), n);
            for (size_t i = 0 ; i < n; i++){
                idx_t group_id = use_hash ? hash_ids[i] : encoded_ids[i];
                for (size_t j =0; j < this->norm_code_size; j++){
                    this->base_norm_codes[group_id].push_back(xnorm_codes[i * this->norm_code_size +j]);
                }
            }
        }
        else{
            assert(this->base_norm.size() == this->final_group_num);
            for (size_t i = 0; i < n; i++){
                idx_t group_id = use_hash ? hash_ids[i] : encoded_ids[i];
                this->base_norm[group_id].push_back(xnorms[i]);
            }
        }
    }

    /**
     * 
     * This is the function for get a final centroid data
     * 
     * Input: 
     * group id: the id of the final group in the last layer
     * 
     * Output:
     * final centroid: the centroid data of the group id
     * 
     **/
    void Bslib_Index::get_final_centroid(size_t group_id, float * final_centroid){
        if (index_type[layers - 1] == "VQ"){
            size_t n_vq = vq_quantizer_index.size();
            size_t group_num = vq_quantizer_index[n_vq-1].nc_upper;
            size_t group_size = vq_quantizer_index[n_vq-1].nc_per_group;
            size_t vq_group_id = size_t(group_id / group_size);
            assert(vq_group_id < group_num);
            size_t inner_vq_group_id = group_id - vq_group_id * group_size;
            vq_quantizer_index[n_vq - 1].compute_final_centroid(vq_group_id, inner_vq_group_id, final_centroid);
        }
        else if(index_type[layers -1] == "LQ"){
            size_t n_lq = lq_quantizer_index.size();
            size_t group_num = lq_quantizer_index[n_lq - 1].nc_upper;
            size_t group_size = lq_quantizer_index[n_lq - 1].nc_per_group;
            size_t lq_group_id = size_t(group_id / group_size);
            assert(lq_group_id < group_num);
            size_t inner_lq_group_id = group_id - lq_group_id * group_size;
            lq_quantizer_index[n_lq - 1].compute_final_centroid(lq_group_id, inner_lq_group_id, final_centroid);
        }
        else{
            size_t n_pq = pq_quantizer_index.size();
            pq_quantizer_index[n_pq - 1].compute_final_centroid(group_id, final_centroid);
        }
    }


    /**
     * 
     * This is the function for computing norms of final centroids
     * For LQ and VQ layer, we compute the norm directly, but for PQ, we compute it indirectedly
     * The number of centroids in VQ and LQ are: final_nc
     * The size of centroid norm is 0 for PQ layer
     * 
     **/
    void Bslib_Index::compute_centroid_norm(){
        if (this->index_type[layers -1] == "VQ"){
            this->centroid_norms.resize(final_group_num);
            size_t n_vq = vq_quantizer_index.size();
            size_t group_num = vq_quantizer_index[n_vq-1].nc_upper;
            size_t group_size = vq_quantizer_index[n_vq-1].nc_per_group;
            assert(final_group_num == group_num * group_size);

            std::cout << "Computing centroid norm for " << final_group_num << " centroids in VQ layer" << std::endl;
            for (size_t i = 0; i < group_num; i++){
                for (size_t j = 0; j < group_size; j++){
                    std::vector<float> each_centroid(dimension);
                    vq_quantizer_index[n_vq-1].compute_final_centroid(i, j, each_centroid.data());
                    this->centroid_norms[i * group_size + j] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
                }
            }
        }
        else if (this->index_type[layers -1] == "LQ"){
            this->centroid_norms.resize(final_group_num);
            size_t n_lq = lq_quantizer_index.size();
            size_t group_num = lq_quantizer_index[n_lq - 1].nc_upper;
            size_t group_size = lq_quantizer_index[n_lq - 1].nc_per_group;
            assert(final_group_num == group_num * group_size);

            std::cout << "Computing centroid norm for " << final_group_num << " centroids in LQ layer" << std::endl;
            for (size_t i = 0; i < group_num; i++){
                for (size_t j = 0; j < group_size; j++){
                    std::vector<float> each_centroid(dimension);
                    lq_quantizer_index[n_lq-1].compute_final_centroid(i, j, each_centroid.data());
                    this->centroid_norms[i * group_size + j] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
                }
            }
        }
        else{
            PrintMessage("The centroid norms of PQ layer should be computed indirectedly");
        }
    }

    /**
     * 
     * This is the function for assigning the vectors into group
     * 
     * Input: 
     * assign_data: the vectors to be assigned:                 size: n * dimension
     *  
     * Output:
     * assigned_ids: the result id result for assigned vectors  size: n
     * 
     **/
    void Bslib_Index::assign(const size_t n, const float * assign_data, idx_t * assigned_ids){
        bool report_time = false;
        std::vector<idx_t> group_ids (n, 0);
        std::vector<float> group_dists(n, 0.0);

        size_t n_vq = 0;
        size_t n_lq = 0;
        size_t n_pq = 0;
        size_t group_size = 0;
        clock_t starttime, endtime;

        //The size for results should be: n * group_size * search_space
        std::vector<float> result_dists(this->max_group_size * n * 1, 0);
        std::vector<idx_t> result_labels (this->max_group_size * n * 1, 0);
        //search_space is num_group * group_size, num_group should always be 1 in assigning
        size_t search_space = 1;
        //The keep_space is always 1 for assign
        const size_t keep_space = 1;

        for (size_t i = 0; i < this->layers; i++){
            assert(n_lq + n_vq + n_pq == i);

            if (index_type[i] == "VQ"){
                starttime = clock();
                group_size = vq_quantizer_index[n_vq].nc_per_group;
                vq_quantizer_index[n_vq].search_in_group(n, assign_data, group_ids.data(), result_dists.data(), result_labels.data(), 1);
                //Update the group_idxs for VQ layer
                if (use_HNSW_VQ){
                    for (size_t j = 0; j < n; j++){group_ids[j] = result_labels[j * group_size]; group_dists[j] = result_dists[j * group_size];}
                }    
                else{
#pragma omp parallel for
                    for (size_t j = 0; j < n; j++){keep_k_min(group_size, 1, result_dists.data()+j*group_size, result_labels.data()+j*group_size, group_dists.data()+j, group_ids.data()+j);}
                }
                n_vq ++;
                endtime = clock();
                if(report_time) std::cout << "Time for searching in VQ layer: " << float(endtime - starttime) / CLOCKS_PER_SEC << std::endl;
            }

            else if(index_type[i] == "LQ"){
                clock_t starttime = clock();
                group_size = lq_quantizer_index[n_lq].nc_per_group;
                //Copying the upper layer result for LQ layer usage
                //This search space is the search space for upper search space = upper group size * 1
                //The group size is the nc_per_group in this layer
                std::vector<idx_t> upper_result_labels(n * search_space);
                std::vector<float> upper_result_dists(n * search_space);
                memcpy(upper_result_labels.data(), result_labels.data(), n * search_space * sizeof(idx_t));
                memcpy(upper_result_dists.data(), result_dists.data(), n * search_space * sizeof(float));
                //for (size_t j = 0; j < n * search_space; j++){upper_result_labels[j] = result_labels[j]; upper_result_dists[j] = result_dists[j];}

                lq_quantizer_index[n_lq].search_in_group(n, assign_data, upper_result_labels.data(), upper_result_dists.data(), search_space, group_ids.data(), result_dists.data(), result_labels.data());
                //Update the group_ids for LQ layer
#pragma omp parallel for
                for (size_t j = 0; j < n; j++){keep_k_min(group_size, 1, result_dists.data()+j*group_size, result_labels.data()+j*group_size, group_dists.data()+j, group_ids.data()+j);}
                n_lq ++;
                endtime = clock();
                if(report_time) std::cout << "Time in LQ layer: " << float(endtime - starttime) / CLOCKS_PER_SEC << std::endl;
            }
            else if (index_type[i] == "PQ"){
                starttime = clock();
                std::vector<float> pq_result_dists(n, 0);
                std::vector<idx_t> pq_result_labels(n, 0);
                pq_quantizer_index[n_pq].search_in_group(n, assign_data, group_ids.data(), pq_result_dists.data(), pq_result_labels.data(), keep_space);
                //for (size_t j = 0; j < n; j++){group_ids[j] = result_labels[j]; group_dists[j] = result_dists[j];}
                memcpy(group_ids.data(), pq_result_labels.data(), n * sizeof(idx_t));
                memcpy(group_dists.data(),pq_result_dists.data(), n * sizeof(float));
                n_pq ++;
                endtime = clock();
                if(report_time) std::cout << "Time in PQ layer" << float(endtime - starttime) / CLOCKS_PER_SEC << std::endl;
            }
            else{
                std::cout << "The type name is wrong with " << index_type[i] << "!" << std::endl;
                exit(0); 
            }
            search_space = group_size * keep_space;
        }

        assert((n_vq + n_lq + n_pq) == this->layers);
        memcpy(assigned_ids, group_ids.data(), n * sizeof(idx_t));
        //for (size_t i = 0; i < n; i++){assigned_ids[i] = group_ids[i];}
    }


    /**
     * 
     * This is the function for keeping k results in m result value
     * 
     * Input:
     * m: the total number of result pairs
     * k: the number of result pairs that to be kept
     * all_dists: the origin results of dists         size: m 
     * all_labels: the origin label results           size: m
     * sub_dists: the kept result of dists            size: k
     * sub_labels: the kept result of labels          size: k   
     * 
     **/
    void Bslib_Index::keep_k_min(const size_t m, const size_t k, const float * all_dists, const idx_t * all_labels, float * sub_dists, idx_t * sub_labels){
        
        if (k < m){
            faiss::maxheap_heapify(k, sub_dists, sub_labels);
            for (size_t i = 0; i < m; i++){
                if (all_dists[i] < sub_dists[0]){
                    faiss::maxheap_pop(k, sub_dists, sub_labels);
                    faiss::maxheap_push(k, sub_dists, sub_labels, all_dists[i], all_labels[i]);
                }
            }
        }
        else if (k == m){
            memcpy(sub_dists, all_dists, k * sizeof(float));
            memcpy(sub_labels, all_labels, k * sizeof(idx_t));
            //for (size_t i = 0; i < m; i++){sub_dists[i] = all_dists[i];sub_labels[i] = all_labels[i];}
        }
        else{
            std::cout << "k should be at least as large as m in keep_k_min function " << std::endl;
            exit(0);
        }
    }

    /**
     * 
     * This is the function for getting the product between query and base vector
     * 
     * Input:
     * code: the code of base vectors
     * 
     * Output:
     * the product value of query and quantized base vectors
     * 
     **/
    float Bslib_Index::pq_L2sqr(const uint8_t *code, const float * precomputed_table)
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

    /**
     * 
     * This is the function for selecting the next group for searching
     * 
     * Input:
     * keep_result_space: the size of group_ids and query_group_dists
     * group_ids: the list of cloest group_ids               size: keep_result_space
     * result_idx_dist: the list of corresponding dists      size: keep_result_space
     * 
     **/
    size_t Bslib_Index::get_next_group_idx(size_t keep_result_space, idx_t * group_ids, float * query_group_dists, std::pair<idx_t, float> & result_idx_dist){
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
        result_idx_dist.first = min_label;
        result_idx_dist.second = min_dist;
        return min_i;
    }

     /**
      * d = || x - y_C - y_R ||^2
      * d = || x - y_C ||^2 - || y_C ||^2 + || y_C + y_R ||^2 - 2 * (x|y_R)
      *        -----------------------------   -----------------   -----------
      *                     term 1                   term 2           term 3
      * Term1 can be computed in assigning the queries
      * Term2 is the norm of base vectors
      * Term3 is the inner product between x and y_R
      * 
      * Input:
      * n: the number of query vectors
      * result_k: the kNN neighbor that we want to search
      * queries: the query vectors                                   size: n * dimension
      * keep_space: the kept centroids in each layer           
      * ground_truth: the groundtruth labels for testing             size: n * result_k
      * 
      * Output:
      * query_ids: the result origin label                           size: n
      * query_dists: the result dists to origin base vectors         size: n * result_k
      * 
      * 
      **/

    void Bslib_Index::search(size_t n, size_t result_k, float * queries, float * query_dists, idx_t * query_ids, size_t * keep_space, uint32_t * groundtruth, std::string path_base){
        
        // Variables for testing and validation and printing
        // Notice: they should only be activated when parallel is not used
        bool validation = false; size_t validation_print_space = 50;
        bool analysis = false;
        bool showmessage = false;

        std::vector<float>  visited_gt_proportion;
        std::vector<size_t> actual_visited_vectors;
        std::vector<std::vector<float>> time_consumption;
        std::vector<float> q_c_dists;
        std::vector<float> recall;
        if (analysis){
            visited_gt_proportion.resize(n, 0); actual_visited_vectors.resize(n, 0);
            time_consumption.resize(n); for (size_t i=0;i<n;i++){time_consumption[i].resize(layers+1, 0);}
            q_c_dists.resize(n, 0); recall.resize(n, 0);
        }

        std::ifstream base_input;
        if (validation){base_input = std::ifstream(path_base, std::ios::binary);}

//Use parallel in real use
//#pragma omp parallel for
        for (size_t i = 0; i < n; i++){

            //Variables for analysis
            time_recorder Trecorder = time_recorder();
            std::unordered_set<uint32_t> grountruth_set;
            if (analysis){for (size_t j = 0; j < result_k; j++){grountruth_set.insert(groundtruth[i * 100 + j]);}}

            const float * query = queries + i * dimension;
            size_t n_vq = 0;
            size_t n_lq = 0;
            size_t n_pq = 0;

            size_t keep_result_space = 1;
            size_t search_space = 1;
            size_t group_size = 1;

            // The final keep space is the number of groups that we kept for searching at last, the product of all keep space
            size_t final_keep_space = 1;
            // The max search space we meet: for one query in on layer, it is the keep space upper layer * ncentroids this layer
            size_t max_search_space = 1;

            for (size_t j = 0; j < layers; j++){

                if (final_keep_space * ncentroids[j] > max_search_space || final_keep_space * keep_space[j] > max_search_space){
                    max_search_space = final_keep_space * (ncentroids[j] > keep_space[j]) ? ncentroids[j] : keep_space[j];
                }
                final_keep_space *= keep_space[j];
            }

            std::vector<float> query_result_dists(1 * max_search_space, 0);
            std::vector<idx_t> query_result_labels(1 * max_search_space, 0);

            std::vector<idx_t> query_group_ids(1 * final_keep_space, 0);
            std::vector<float> query_group_dists(1 * final_keep_space, 0);

//#pragma omp critical
            for (size_t j = 0; j < layers; j++){
                assert(n_vq+ n_lq + n_pq== j);
                
                if (index_type[j] == "VQ"){
                    if (showmessage) PrintMessage("Searching in VQ Layer");
                    group_size = vq_quantizer_index[n_vq].nc_per_group;
//#pragma omp parallel for
                    for (size_t m = 0; m < keep_result_space; m++){
                        vq_quantizer_index[n_vq].search_in_group(1, query, query_group_ids.data()+m, query_result_dists.data()+m*group_size, query_result_labels.data() + m*group_size, keep_space[j]);
                    }
                    if (use_HNSW_VQ){
                        for (size_t m = 0; m < keep_result_space; m++){
                            for (size_t k = 0; k < keep_space[j]; k++){
                                query_group_ids[m * keep_space[j] + k] = query_result_labels[m * group_size + k];
                                query_group_dists[m * keep_space[j] + k] = query_result_dists[m * group_size + k];
                            }
                        }
                    }
                    else{
                        for (size_t m = 0; m < keep_result_space; m++){
                            keep_k_min(group_size, keep_space[j], query_result_dists.data()+m*group_size, query_result_labels.data()+m*group_size, query_group_dists.data()+m*keep_space[j], query_group_ids.data()+m*keep_space[j]); 
                        }
                    }
                    n_vq ++;
                }
                else if(index_type[j] == "LQ") {
                    if (showmessage) PrintMessage("Searching in LQ layer");
                    group_size = lq_quantizer_index[n_lq].nc_per_group;
                    // Copy the upper search result for LQ layer 
                    std::vector<idx_t> upper_result_labels(search_space);
                    std::vector<float> upper_result_dists(search_space);
                    memcpy(upper_result_labels.data(), query_result_labels.data(), search_space * sizeof(idx_t));
                    memcpy(upper_result_dists.data(), query_result_dists.data(), search_space * sizeof(float));

                    //for (size_t m = 0; m < search_space; m++){upper_result_labels[m] = result_labels[m]; upper_result_dists[m] = result_dists[m];}
//#pragma omp parallel for
                    for (size_t m = 0; m < keep_result_space; m++){
                        lq_quantizer_index[n_lq].search_in_group(1, query, upper_result_labels.data(), upper_result_dists.data(), search_space, query_group_ids.data()+m*1, query_result_dists.data()+m*group_size, query_result_labels.data()+m*group_size);
                    }
                    for (size_t m = 0; m < keep_result_space; m++){
                        keep_k_min(group_size, keep_space[j], query_result_dists.data()+m*group_size, query_result_labels.data()+m*group_size, query_group_dists.data()+m*keep_space[j], query_group_ids.data()+m*keep_space[j]);
                    }
                    n_lq ++;
                }

                else if(index_type[j] == "PQ"){
                    if (showmessage) PrintMessage("Searching in PQ layer");
                    assert(j == this->layers-1);
//#pragma omp parallel for
                    for (size_t m = 0; m < keep_result_space; m++){
                        pq_quantizer_index[n_pq].search_in_group(1, query, query_group_ids.data()+m, query_result_dists.data()+m*keep_space[j], query_result_labels.data()+m*keep_space[j], keep_space[j]);
                    }
                    assert(keep_result_space * keep_space[j] == final_keep_space);
                    for (size_t m = 0; m < keep_result_space; m++){
                        for (size_t k = 0; k < keep_space[j]; k++){
                            query_group_ids[m * keep_result_space + k] = query_result_labels[m * keep_space[j] + k];
                            query_group_dists[m * keep_result_space + k] = query_result_dists[m * keep_space[j] + k];
                        }
                    }
                    n_pq ++;
                }
                else{
                    std::cout << "The type name is wrong with " << index_type[j] << "!" << std::endl;
                    exit(0);
                }
                search_space = group_size * keep_result_space;
                keep_result_space = keep_result_space * keep_space[j];

                if (analysis){time_consumption[i][j] = Trecorder.getTimeConsumption();Trecorder.reset();}
            }

            assert((n_vq + n_lq + n_pq) == this->layers);
            
            std::vector<float> precomputed_table(pq.M * pq.ksub);
            this->pq.compute_inner_prod_table(query, precomputed_table.data());

            //The analysis variables
            size_t visited_vectors = 0;
            size_t visited_gt = 0;
            if (analysis){
                float query_centroid_dists = 0;
                for (size_t j = 0; j < final_keep_space; j++){query_centroid_dists += query_group_dists[j];}
                query_centroid_dists = query_centroid_dists / final_keep_space;
                q_c_dists[i] = query_centroid_dists;
            }

            std::vector<idx_t> query_pre_hash_ids; 
            if (use_hash){query_pre_hash_ids.resize(final_keep_space, 0); memcpy(query_pre_hash_ids.data(), query_group_ids.data(), final_keep_space * sizeof(idx_t)); 
            HashMapping(final_keep_space, query_pre_hash_ids.data(), query_group_ids.data(), final_group_num);}
            size_t max_size = 0;for (size_t j = 0; j < final_keep_space; j++){if (base_sequence_ids[query_group_ids[j]].size() > max_size) max_size = base_sequence_ids[query_group_ids[j]].size();}
            
            std::vector<float> query_search_dists(max_visited_vectors + max_size, 0);std::vector<idx_t> query_search_labels(max_visited_vectors + max_size, 0);
            // The dists for actual data
            std::vector<float> query_actual_dists; if (validation){(query_actual_dists.resize(max_visited_vectors + max_size));}

            for (size_t j = 0; j < final_keep_space; j++){

                std::pair<idx_t, float> result_idx_dist;
                size_t id_position = get_next_group_idx(final_keep_space, query_group_ids.data(), query_group_dists.data(), result_idx_dist);
                
                idx_t group_id = result_idx_dist.first;
                float q_c_dist = result_idx_dist.second;
                if (showmessage) std::cout << "Searching in " << group_id << " th group with distance " << q_c_dist << std::endl;

                size_t group_size = this->base_sequence_ids[group_id].size();
                assert(group_size == this->base_codes[group_id].size() / this->code_size);

                float centroid_norm;
                idx_t centroid_id = (use_hash) ? query_pre_hash_ids[id_position] : group_id;
                if (use_hash)
                    centroid_norm = this->pq_quantizer_index[0].get_centroid_norms(centroid_id);
                else
                    centroid_norm = centroid_norms[centroid_id];
                
                assert(centroid_norm > 0);
                float term1 = q_c_dist - centroid_norm;

                // Validating the computation of q_c_dist and centroid_norm
                if (validation){
                    float actual_centroid_norm, actual_q_c_dist;
                    std::vector<float> centroid(dimension);
                    std::vector<float> q_c_dis_vector(dimension);
                    get_final_centroid(centroid_id, centroid.data());
                    actual_centroid_norm = faiss::fvec_norm_L2sqr(centroid.data(), dimension);
                    faiss::fvec_madd(dimension, query, -1.0, centroid.data(), q_c_dis_vector.data());
                    actual_q_c_dist = faiss::fvec_norm_L2sqr(q_c_dis_vector.data(), dimension);
                    assert(abs(q_c_dist - actual_q_c_dist) < VALIDATION_EPSILON && abs(centroid_norm - actual_centroid_norm) < VALIDATION_EPSILON);
                }

                const uint8_t * code = base_codes[group_id].data();

                for (size_t m = 0; m < group_size; m++){
                    if (base_pre_hash_ids[j][m] != query_pre_hash_ids[j]){
                        continue;
                    }
                    else{
                        std::vector<float> base_reconstructed_norm(1);
                        if (use_norm_quantization) norm_pq.decode(base_norm_codes[group_id].data() + m * norm_code_size, base_reconstructed_norm.data(), 1);
                        float term2 = use_norm_quantization ? base_reconstructed_norm[0] : base_norm[group_id][m];
                        float term3 = 2 * pq_L2sqr(code + m * code_size, precomputed_table.data());
                        float dist = term1 + term2 - term3;
                        query_search_dists[visited_vectors] = dist;

                        //Compute the actual distance
                        if (validation){
                            std::vector<base_data_type> base_vector(dimension); uint32_t dim;
                            base_input.seekg(base_sequence_ids[group_id][m] * dimension * sizeof(base_data_type) + base_sequence_ids[group_id][m] * sizeof(uint32_t), std::ios::beg);
                            base_input.read((char *) & dim, sizeof(uint32_t)); assert(dim == this->dimension);
                            base_input.read((char *) base_vector.data(), sizeof(base_data_type)*dimension);
                            std::vector<float> base_vector_float(dimension);
                            for (size_t temp = 0; temp < dimension; temp++){base_vector_float[temp] = base_vector[temp];}
                            std::vector<float> distance_vector(dimension);
                            faiss::fvec_madd(dimension, base_vector_float.data(), -1, query, distance_vector.data());
                            float actual_dist =  faiss::fvec_norm_L2sqr(distance_vector.data(), dimension);
                            std::vector<float> decoded_code(dimension);
                            pq.decode(code + m * code_size, decoded_code.data(), 1);
                            std::vector<float> decoded_base_vector(dimension);
                            decode(1, decoded_code.data(), & group_id, decoded_base_vector.data());
                            float actual_norm = faiss::fvec_norm_L2sqr(decoded_base_vector.data(), dimension);
                            query_actual_dists[visited_vectors] = actual_dist;
                            std::cout << "S: " << dist << " A: " << actual_dist << " LN: " << term2 << " AN: " << actual_norm << " "; // S for "Search" and A for "Actual"
                        }

                        query_search_labels[visited_vectors] = base_sequence_ids[group_id][m];
                        visited_vectors ++;
                        if (analysis){if (grountruth_set.count(base_sequence_ids[group_id][m]) != 0){visited_gt ++;}}
                    }
                }
                if (visited_vectors >= this->max_visited_vectors)
                    break;
            }

            if (validation){
            //Compute the distance sort for computed distance
                std::cout << std::endl;
                std::vector<idx_t> search_dist_index(visited_vectors);
                uint32_t x=0;
                std::iota(search_dist_index.begin(),search_dist_index.end(),x++);
                std::sort(search_dist_index.begin(),search_dist_index.end(), [&](int i,int j){return query_search_dists[i]<query_search_dists[j];} );

                //Compute the distance sort for actual distance
                std::vector<idx_t> actual_dist_index(visited_vectors);
                x = 0;
                std::iota(actual_dist_index.begin(), actual_dist_index.end(), x++);
                std::sort(actual_dist_index.begin(),actual_dist_index.end(), [&](int i,int j){return query_actual_dists[i]<query_actual_dists[j];} );

                assert(visited_vectors > validation_print_space);
                std::cout << "Search Labels     Search Dists     Actual Labels     Actual Dists" << std::endl;
                for (size_t temp = 0; temp < validation_print_space; temp++){
                    std::cout << query_search_labels[search_dist_index[temp]] << "        " << 
                    query_search_dists[search_dist_index[temp]] << "        " << 
                    query_search_labels[actual_dist_index[temp]] << "        " <<
                    query_actual_dists[actual_dist_index[temp]] << std::endl;
                }
            }

            if (use_reranking){
                assert(visited_vectors > reranking_space);
                std::ifstream reranking_input(path_base, std::ios::binary);
                std::vector<float> reranking_dists(reranking_space, 0);
                std::vector<idx_t> reranking_labels(reranking_space, 0);
                std::vector<float> reranking_actual_dists(reranking_space, 0);
                keep_k_min(visited_vectors, reranking_space, query_search_dists.data(), query_search_labels.data(), reranking_dists.data(), reranking_labels.data());

                for (size_t j = 0; j < reranking_space; j++){
                    std::vector<base_data_type> base_vector(dimension); std::vector<float> base_vector_float(dimension); uint32_t dim;
                    reranking_input.seekg(reranking_labels[j] * dimension * sizeof(base_data_type) + reranking_labels[j] * sizeof(uint32_t), std::ios::beg);
                    reranking_input.read((char *) & dim, sizeof(uint32_t)); assert(dim == dimension);
                    reranking_input.read((char *) base_vector.data(), sizeof(base_data_type) * dimension);
                    for (size_t temp = 0; temp < dimension; temp++){base_vector_float[temp] = base_vector[temp];}
                    std::vector<float> distance_vector(dimension);
                    faiss::fvec_madd(dimension, base_vector_float.data(), -1, query, distance_vector.data());
                    float actual_dist =  faiss::fvec_norm_L2sqr(distance_vector.data(), dimension);
                    reranking_actual_dists[j] = actual_dist;
                }

                keep_k_min(reranking_space, result_k, reranking_actual_dists.data(), reranking_labels.data(), query_dists + i * result_k, query_ids + i * result_k);
            }

            else{keep_k_min(visited_vectors, result_k, query_search_dists.data(), query_search_labels.data(), query_dists + i * result_k, query_ids + i * result_k);}

            if (analysis){
                size_t correct = 0; for (size_t temp=0;temp<result_k;temp++){if(grountruth_set.count(query_ids[temp])!=0) correct++;}
                recall[i] = float(correct) / result_k;
                time_consumption[i][layers]  = Trecorder.getTimeConsumption();
                visited_gt_proportion[i] = float(visited_gt) / result_k;
                actual_visited_vectors[i] = visited_vectors;
            }
        }

        if (analysis){
            float avg_visited_vectors = 0, avg_q_c_dist = 0, avg_visited_gt_proportion = 0, avg_recall = 0;
            std::vector<float> avg_time_consumption(layers+1, 0);
            for (size_t i = 0; i < n; i++){
                avg_visited_vectors += actual_visited_vectors[i];
                avg_q_c_dist += q_c_dists[i];
                avg_visited_gt_proportion += visited_gt_proportion[i];
                for (size_t j = 0; j < layers+1; j++){avg_time_consumption[j] += time_consumption[i][j];}
                avg_recall += recall[i];
            }
            std::vector<size_t> base_groups_size(final_group_num); for(size_t i = 0; i < final_group_num; i++){base_groups_size[i] = base_sequence_ids[i].size();}
            double sum = std::accumulate(std::begin(base_groups_size), std::end(base_groups_size), 0.0);
	        double mean =  sum / base_groups_size.size(); //Mean value
 
        	double accum  = 0.0;
	        std::for_each (std::begin(base_groups_size), std::end(base_groups_size), [&](const double d) {accum  += (d-mean)*(d-mean);});
	        double stdev = sqrt(accum/(base_groups_size.size()-1)); //Variance value

            std::cout << "Time consumption in different parts: ";for(size_t i = 0; i < layers+1; i++){std::cout << avg_time_consumption[i] / n << " ";}std::cout << std::endl;
            std::cout << "The average visited vectors: " << avg_visited_vectors / n << std::endl;
            std::cout << "The average query centroid distance: " << avg_q_c_dist / n << std::endl;
            std::cout << "The avarage groundtruth proportion is: " << avg_visited_gt_proportion / n << std::endl;
            std::cout << "The avarage recall is: " << avg_recall / n << std::endl;
            std::cout << "The mean and variance of index distribution is: " << mean << "  " << stdev << std::endl;
        }
    }


    void Bslib_Index::write_quantizers(const std::string path_quantizer){
        PrintMessage("Writing quantizers");
        std::ofstream quantizers_output(path_quantizer, std::ios::binary);
        size_t n_vq = 0;
        size_t n_lq = 0;
        size_t n_pq = 0;
        quantizers_output.write((char *) & this->max_group_size, sizeof(size_t));
        for (size_t i =0; i < this->layers; i++){
            quantizers_output.write((char *) & this->ncentroids[i], sizeof(size_t));
        }

        for (size_t i = 0; i < this->layers; i++){
            if (index_type[i] == "VQ"){
                PrintMessage("Writing VQ quantizer layer");
                const size_t nc = vq_quantizer_index[n_vq].nc;
                const size_t nc_upper = vq_quantizer_index[n_vq].nc_upper;
                const size_t nc_per_group = vq_quantizer_index[n_vq].nc_per_group;
                quantizers_output.write((char *) & nc, sizeof(size_t));
                quantizers_output.write((char *) & nc_upper, sizeof(size_t));
                quantizers_output.write((char *) & nc_per_group, sizeof(size_t));
                std::cout << nc << " " << nc_upper << " " << nc_per_group << std::endl;

                if (use_HNSW_VQ){
                    size_t M = vq_quantizer_index[n_vq].M;
                    size_t efConstruction = vq_quantizer_index[n_vq].efConstruction;
                    size_t efSearch = vq_quantizer_index[n_vq].efSearch;
                    quantizers_output.write((char * ) & M, sizeof(size_t));
                    quantizers_output.write((char * ) & efConstruction, sizeof(size_t));
                    quantizers_output.write((char * ) & efSearch, sizeof(size_t));

                    vq_quantizer_index[n_vq].write_HNSW(quantizers_output);
                }
                else{
                    std::cout << "Writing centroids " << std::endl;
                    for (size_t j = 0; j < nc_upper; j++){
                        std::cout << "Writing centroids " << std::endl;
                        size_t group_quantizer_data_size = nc_per_group * this->dimension;
                        assert(vq_quantizer_index[n_vq].L2_quantizers[j]->xb.size() == group_quantizer_data_size);
                        std::cout << "the centroid size is " << vq_quantizer_index[0].L2_quantizers.size() << std::endl;
                        std::cout << vq_quantizer_index[n_vq].L2_quantizers[j]->xb.size() << std::endl;
                        quantizers_output.write((char * ) vq_quantizer_index[n_vq].L2_quantizers[j]->xb.data(), group_quantizer_data_size * sizeof(float));
                    }
                }
                assert(n_vq + n_lq + n_pq == i);
                n_vq ++;
            }
            else if (index_type[i] == "LQ"){
                PrintMessage("Writing LQ quantizer layer");
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
                    assert(lq_quantizer_index[n_lq].nn_centroid_ids[j].size() == nc_per_group);
                    quantizers_output.write((char *)lq_quantizer_index[n_lq].nn_centroid_ids[j].data(), nc_per_group * sizeof(idx_t));
                }
                for (size_t j = 0; j < nc_upper; j++){
                    assert(lq_quantizer_index[n_lq].nn_centroid_dists[j].size() == nc_per_group);
                    quantizers_output.write((char * )lq_quantizer_index[n_lq].nn_centroid_dists[j].data(), nc_per_group * sizeof(float));
                }
                assert(n_vq + n_lq + n_pq == i);
                n_lq ++;
            }
            else if (index_type[i] == "PQ"){
                PrintMessage("Writing PQ quantizer layer");
                const size_t nc_upper = pq_quantizer_index[n_pq].nc_upper;
                const size_t M = pq_quantizer_index[n_pq].M;
                const size_t nbits = pq_quantizer_index[n_pq].nbits;
                const size_t ksub = pq_quantizer_index[n_pq].ksub;
                quantizers_output.write((char * ) & nc_upper, sizeof(size_t));
                quantizers_output.write((char * ) & M, sizeof(size_t));
                quantizers_output.write((char * ) & nbits, sizeof(size_t));

                assert(pq_quantizer_index[n_pq].PQs.size() == nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    size_t centroid_size = pq_quantizer_index[n_pq].PQs[j]->centroids.size();
                    assert(centroid_size == dimension * pq_quantizer_index[n_pq].PQs[j]->ksub);
                    quantizers_output.write((char * )pq_quantizer_index[n_pq].PQs[j]->centroids.data(), centroid_size * sizeof(float));
                }

                for (size_t j = 0; j < nc_upper; j++){
                    quantizers_output.write((char *) pq_quantizer_index[n_pq].centroid_norms[j].data(), M*ksub*sizeof(float));
                }

                assert(n_vq + n_lq + n_pq == i);
                n_pq ++;
            }
            else{
                std::cout << "Index type error: " << index_type[i] << std::endl;
                exit(0);
            }
        }
        PrintMessage("Write quantizer finished");
        quantizers_output.close();
    }


    void Bslib_Index::read_quantizers(const std::string path_quantizer){
        PrintMessage("Reading quantizers ");
        std::ifstream quantizer_input(path_quantizer, std::ios::binary);

        //For each layer, there is nc, nc_upper and nc_per_group
        size_t nc;
        size_t nc_upper;
        size_t nc_per_group;
        quantizer_input.read((char *) & this->max_group_size, sizeof(size_t));

        for (size_t i = 0; i < this->layers; i++){
            quantizer_input.read((char * ) & nc_per_group, sizeof(size_t));
            this->ncentroids[i] = nc_per_group;
        }

        for(size_t i = 0; i < this->layers; i++){
            if (index_type[i] == "VQ"){
                std::cout << "Reading VQ layer" << std::endl;
                quantizer_input.read((char *) & nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & nc_per_group, sizeof(size_t));
                std::cout << nc << " " << nc_upper << " " << nc_per_group << " " << std::endl;
                assert(nc_per_group * nc_upper == nc);
                size_t M;
                size_t efConstruction;
                size_t efSearch;
                if (use_HNSW_VQ){
                    quantizer_input.read((char *) & M, sizeof(size_t));
                    quantizer_input.read((char *) & efConstruction, sizeof(size_t));
                    quantizer_input.read((char *) & efSearch, sizeof(size_t));
                    VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, nc_per_group, M, efConstruction, efSearch, use_HNSW_VQ);
                    vq_quantizer.read_HNSW(quantizer_input);
                    this->vq_quantizer_index.push_back(vq_quantizer);
                }
                else{
                    VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, nc_per_group);
                    std::vector<float> centroids(nc_per_group * this->dimension);
                    for (size_t j = 0; j < nc_upper; j++){
                        quantizer_input.read((char *) centroids.data(), nc_per_group * dimension * sizeof(float));
                        faiss::IndexFlatL2 * centroid_quantizer = new faiss::IndexFlatL2(dimension);
                        centroid_quantizer->add(nc_per_group, centroids.data());
                        vq_quantizer.L2_quantizers[j] = centroid_quantizer;
                    }
                    this->vq_quantizer_index.push_back(vq_quantizer);
                }
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
                std::vector<idx_t> nn_centroid_ids(nc_upper * nc_per_group);
                std::vector<float> nn_centroid_dists(nc_upper * nc_per_group);

                quantizer_input.read((char *) alphas.data(), nc_upper * sizeof(float));
                quantizer_input.read((char *) upper_centroids.data(), nc_upper * this->dimension * sizeof(float));
                quantizer_input.read((char *) nn_centroid_ids.data(), nc_upper * nc_per_group * sizeof(idx_t));
                quantizer_input.read((char *) nn_centroid_dists.data(), nc_upper * nc_per_group * sizeof(float));

                LQ_quantizer lq_quantizer = LQ_quantizer(dimension, nc_upper, nc_per_group, upper_centroids.data(), nn_centroid_ids.data(), nn_centroid_dists.data());
                for (size_t j = 0; j < nc_upper; j++){
                    lq_quantizer.alphas[j] = alphas[j];
                }
                this->lq_quantizer_index.push_back(lq_quantizer);
            }

            else if (index_type[i] == "PQ"){
                std::cout << "Reading PQ layer " << std::endl;
                size_t M;
                size_t nbits;
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & M, sizeof(size_t));
                quantizer_input.read((char *) & nbits, sizeof(size_t));
                
                PQ_quantizer pq_quantizer = PQ_quantizer(dimension, nc_upper, M, nbits);
                
                pq_quantizer.PQs.resize(nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    faiss::ProductQuantizer * product_quantizer = new faiss::ProductQuantizer(dimension, M, nbits);
                    size_t centroid_size = dimension * product_quantizer->ksub;
                    quantizer_input.read((char *) product_quantizer->centroids.data(), centroid_size * sizeof(float));
                    pq_quantizer.PQs[j] = product_quantizer;
                }

                pq_quantizer.centroid_norms.resize(nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    pq_quantizer.centroid_norms[j].resize(M * pq_quantizer.ksub);
                    quantizer_input.read((char *) pq_quantizer.centroid_norms[j].data(), M * pq_quantizer.ksub * sizeof(float));
                }
                this->pq_quantizer_index.push_back(pq_quantizer);
            }
        }
        PrintMessage("Read quantizers finished");
        quantizer_input.close();
    }

    void Bslib_Index::write_index(const std::string path_index){
        std::ofstream output(path_index, std::ios::binary);
        output.write((char *) & this->final_group_num, sizeof(size_t));

        if (use_norm_quantization){
            assert((base_norm_codes.size() == base_codes.size()) && (base_codes.size() == base_sequence_ids.size()) && (base_sequence_ids.size() == final_group_num ));
            for (size_t i = 0; i < this->final_group_num; i++){
                assert((base_norm_codes[i].size() == base_codes[i].size() / this->code_size) && (base_norm_codes[i].size() == base_sequence_ids[i].size()));
                size_t group_size = base_norm_codes[i].size();
                output.write((char *) & group_size, sizeof(size_t));
                output.write((char *) base_norm_codes[i].data(), group_size * sizeof(uint8_t));
            }
        }
        else{
            assert((base_norm.size() == base_codes.size()) && (base_codes.size() == base_sequence_ids.size()) && (base_sequence_ids.size() == final_group_num));
            for (size_t i = 0; i < this->final_group_num; i++){
                assert((base_norm[i].size() == base_codes[i].size() / this->code_size) && (base_norm[i].size() == base_sequence_ids[i].size()));
                size_t group_size = base_norm[i].size();
                output.write((char *) & group_size, sizeof(size_t));
                output.write((char *) base_norm[i].data(), group_size * sizeof(float));
            }
        }
        output.write((char *) & this->final_group_num, sizeof(size_t));
        for (size_t i = 0; i < this->final_group_num; i++){
            size_t group_size = base_codes[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) base_codes[i].data(), group_size * sizeof(uint8_t));
        }

        output.write((char *) & this->final_group_num, sizeof(size_t));
        for (size_t i = 0; i < this->final_group_num; i++){
            size_t group_size = base_sequence_ids[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) base_sequence_ids[i].data(), group_size * sizeof(idx_t));
        }

        if (index_type[index_type.size() - 1] != "PQ"){
            output.write((char *) & this->final_group_num, sizeof(size_t));
            assert(centroid_norms.size() == this->final_group_num);
            output.write((char *) centroid_norms.data(), this->final_group_num * sizeof(float));
        }
        output.close();
    }

    void Bslib_Index::read_index(const std::string path_index){
        std::ifstream input(path_index, std::ios::binary);
        size_t final_nc_input;
        size_t group_size_input;

        this->base_codes.resize(this->final_group_num);
        
        this->base_sequence_ids.resize(this->final_group_num);

        if (use_norm_quantization){
            this->base_norm_codes.resize(this->final_group_num);
            input.read((char *) & final_nc_input, sizeof(size_t));
            assert(final_nc_input == this->final_group_num);
            for (size_t i = 0; i < this->final_group_num; i++){
                input.read((char *) & group_size_input, sizeof(size_t));
                this->base_norm_codes[i].resize(group_size_input);
                input.read((char *) base_norm_codes[i].data(), group_size_input * sizeof(uint8_t));
            }
        }
        else{
            this->base_norm.resize(this->final_group_num);
            input.read((char *) & final_nc_input, sizeof(size_t));
            assert(final_nc_input == this->final_group_num);
            for (size_t i = 0; i < this->final_group_num; i++){
                input.read((char *) & group_size_input, sizeof(size_t));
                this->base_norm[i].resize(group_size_input);
                input.read((char *) base_norm[i].data(), group_size_input * sizeof(float));
            }
        }

        input.read((char *) & final_nc_input, sizeof(size_t));
        assert(final_nc_input == this->final_group_num);
        for (size_t i = 0; i < this->final_group_num; i++){
            input.read((char *) & group_size_input, sizeof(size_t));
            base_codes[i].resize(group_size_input);
            input.read((char *) base_codes[i].data(), group_size_input * sizeof(uint8_t));
        }

        input.read((char *) & final_nc_input, sizeof(size_t));
        assert(final_nc_input == this->final_group_num);
        for (size_t i = 0; i < this->final_group_num; i++){
            input.read((char *) & group_size_input, sizeof(size_t));
            base_sequence_ids[i].resize(group_size_input);
            input.read((char *) base_sequence_ids[i].data(), group_size_input * sizeof(idx_t));
        }

        if (index_type[index_type.size()-1] != "PQ"){
            input.read((char *) & final_nc_input, sizeof(size_t));
            assert(final_nc_input == this->final_group_num);
            this->centroid_norms.resize(this->final_group_num);
            input.read((char *) centroid_norms.data(), this->final_group_num * sizeof(float));
        }
        input.close();
    }
}

