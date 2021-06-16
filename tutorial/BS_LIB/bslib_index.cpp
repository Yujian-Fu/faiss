#include "bslib_index.h"
#include "time.h"

namespace bslib{

    /**
     * The initialize function for BSLIB struct 
     * 
     * 
     * 
     **/
    Bslib_Index::Bslib_Index(const size_t dimension, const size_t layers, const std::string * index_type, 
    const bool use_reranking, const bool saving_index, const bool use_norm_quantization, const bool is_recording,
    const bool use_HNSW_VQ, const bool use_HNSW_group, const bool use_all_HNSW, const bool use_OPQ, const bool use_train_selector,
    const size_t train_size, const size_t M_PQ, const size_t nbits){
            
            this->dimension = dimension;
            this->layers = layers;

            this->use_reranking = use_reranking;
            this->use_HNSW_VQ = use_HNSW_VQ;
            this->use_HNSW_group = use_HNSW_group;
            this->use_all_HNSW = use_all_HNSW;
            
            this->use_norm_quantization = use_norm_quantization;
            this->use_OPQ = use_OPQ;
            this->use_train_selector = use_train_selector;

            this->is_recording = is_recording;
            this->saving_index = saving_index;

            this->index_type.resize(layers);
            this->ncentroids.resize(layers);

            for (size_t i = 0; i < layers; i++){
                this->index_type[i] = index_type[i];
            }

            this->train_size = train_size;
            this->M = M_PQ;
            this->nbits = nbits;
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
        
        VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, nc_per_group, M, efConstruction, efSearch, use_HNSW_VQ, use_all_HNSW);
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
        LQ_quantizer lq_quantizer = LQ_quantizer(dimension, nc_upper, nc_per_group, upper_centroids, upper_nn_centroid_idxs, upper_nn_centroid_dists, use_all_HNSW);
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
     * This is the function for doing OPQ for the training and searching part.
     * Input:
     * n: this is the number of vectors to be processed
     * dataset: the vector set to be processed, the result data will be stored in the same place
     * 
     **/
    void Bslib_Index::do_OPQ(idx_t n, float * dataset){
        std::cout << "Doing OPQ for the input data" << std::endl;
        assert(opq_matrix != NULL);
        std::vector<float> copy_dataset(n * dimension);
        memcpy(copy_dataset.data(), dataset, n * dimension * sizeof(float));
        opq_matrix->apply_noalloc(n, copy_dataset.data(), dataset);
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
        time_recorder Trecorder = time_recorder();
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
            std::cout << "Read train input from " << path_learn << std::endl;
            readXvecFvec<learn_data_type>(learn_input, origin_train_set.data(), dimension, total_train_size, false);
            if (use_OPQ){
                do_OPQ(total_train_size, origin_train_set.data());
            }

            std::vector<float> train_set_centroids(group_size * dimension);
            PrintMessage("Running Kmeans to generate the centroids");

            if (sub_train_size < total_train_size){
                std::vector<float> sub_train_set(sub_train_size * dimension);
                RandomSubset(origin_train_set.data(), sub_train_set.data(), dimension, total_train_size, sub_train_size);
                faiss::kmeans_clustering(dimension, sub_train_size, group_size, sub_train_set.data(), train_set_centroids.data(), 50);
            }
            else{
                faiss::kmeans_clustering(dimension, total_train_size, group_size, origin_train_set.data(), train_set_centroids.data(), 50);
            }
            Trecorder.print_time_usage("Run Kmeans to generate the centroids");

            PrintMessage("Computing the total train set distance to the closest centroids");
            faiss::IndexFlatL2 centroid_index(dimension);
            centroid_index.add(group_size, train_set_centroids.data());

            std::vector<idx_t> result_ids(total_train_size);
            std::vector<float> result_dists(total_train_size);
            centroid_index.search(total_train_size, origin_train_set.data(), 1, result_dists.data(), result_ids.data());

            Trecorder.print_time_usage("Computed the total train set distance to the closest centroids");
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
        this->train_data.resize(train_set_size * dimension, 0);
        //Train data ids is for locating the data vectors are used for which group training
        this->train_data_ids.resize(train_set_size, 0);

        if (total_size == train_set_size){
            std::ifstream learn_input(path_learn, std::ios::binary);
            readXvecFvec<learn_data_type>(learn_input, this->train_data.data(), dimension, train_set_size, true, true);
        }
        else if (use_train_selector){
            assert(this->train_set_ids.size() > 0);
            size_t group_size = train_set_ids.size();
            srand((unsigned)time(NULL));
            std::ifstream learn_input(path_learn, std::ios::binary);
            uint32_t dim;

            learn_data_type origin_data[dimension];
            for (size_t i = 0; i < train_set_size; i++){
                size_t group_id = i % group_size;
                size_t inner_group_id = rand() % this->train_set_ids[group_id].size();
                idx_t sequence_id = this->train_set_ids[group_id][inner_group_id];

                learn_input.seekg(sequence_id * dimension * sizeof (learn_data_type) + sequence_id * sizeof(uint32_t), std::ios::beg);
                learn_input.read((char *) & dim, sizeof(uint32_t));
                assert(dim == dimension);
                learn_input.read((char *) & origin_data, dimension * sizeof(learn_data_type));
                for (size_t j = 0; j < dimension; j++){
                    this->train_data[i * dimension + j] = 1.0 * origin_data[j];
                }
            }
        }
        else{
            std::ifstream learn_input(path_learn, std::ios::binary);
            std::vector<float> sum_train_data (total_size * dimension, 0);
            readXvecFvec<learn_data_type>(learn_input, sum_train_data.data(), dimension, total_size, true, true);
            std::cout << "Reading subset without selector" << std::endl;
            RandomSubset<float>(sum_train_data.data(), this->train_data.data(), dimension, total_size, train_set_size);
        }

        if (use_OPQ){
            do_OPQ(train_set_size, this->train_data.data());
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
                    std::cout << "Updating train set for the next layer" << std::endl;
                    //assign(train_data_ids.size(), train_data.data(), train_data_ids.data(), i+1);
                    vq_quantizer_index[vq_quantizer_index.size() - 1].search_all(train_data_ids.size(), 1, train_data.data(), train_data_ids.data());
                }
                
                std::cout << "Trainset Sample" << std::endl;
                
                for (size_t temp = 0; temp <2; temp++){
                    for (size_t temp1 = 0; temp1 < dimension; temp1++){
                        std::cout << this->train_data[temp * dimension + temp1] << " ";
                    }
                        std::cout << train_data_ids[temp];
                        std::cout << std::endl;
                }

                std::cout << i << "th VQ quantizer added, check it " << std::endl;
                std::cout << "The vq quantizer size is: " <<  vq_quantizer_index.size() << " the num of L2 quantizers (groups): " << vq_quantizer_index[vq_quantizer_index.size()-1].L2_quantizers.size() << 
                " the num of HNSW quantizers (groups): " <<  vq_quantizer_index[vq_quantizer_index.size()-1].HNSW_quantizers.size() << std::endl;
            }
            else if(index_type[i] == "LQ"){
                assert (i >= 1);
                upper_centroids.resize(nc_upper * dimension);
                nn_centroids_idxs.resize(nc_upper * nc_per_group);
                nn_centroids_dists.resize(nc_upper * nc_per_group);
                
                if (index_type[i-1] == "VQ"){
                    PrintMessage("Adding LQ quantizer with VQ upper layer");
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
                    std::cout << "Updating train set for the next layer" << std::endl;
                    //assign(train_data_ids.size(), train_data.data(), train_data_ids.data(), i+1);
                    lq_quantizer_index[lq_quantizer_index.size() - 1].search_all(train_data_ids.size(), 1, train_data.data(), train_data_ids.data());
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
        if(saving_index)
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

        assign(train_set_size, this->train_data.data(), train_data_ids.data(), this->layers);

        for (size_t i = train_set_size - 100; i < train_set_size; i++){std::cout << train_data_ids[i] << " ";} std::cout << std::endl;

        std::cout << "Encoding the train dataset with " << train_set_size<< " data points " << std::endl;
        encode(train_set_size, this->train_data.data(), train_data_ids.data(), residuals.data());

        std::cout << "Training the pq " << std::endl;
        this->pq.verbose = true;
        this->pq.train(train_set_size, residuals.data());

        if(saving_index){
            std::cout << "Writing PQ codebook to " << path_pq << std::endl;
            faiss::write_ProductQuantizer(& this->pq, path_pq.c_str());           
        }

        if (use_norm_quantization){
            std::cout << path_norm_pq << "norm PQ not implmented now" << std::endl; 
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
            this->final_group_num = pq_quantizer_index[pq_quantizer_index.size() -1].nc;
        }
    }


    /**
     * 
     * This is the function for adding a base batch 
     * 
     * Input:
     * n: the batch size of the batch data                                                 size: size_t
     * data: the base data                                                                 size: n * dimension
     * sequence_ids: the origin sequence id of data                                        size: n
     * group_ids: the group id of the data                                                 size: n
     * group_positions                                                                     size: n
     * 
     **/
    void Bslib_Index::add_batch(size_t n, const float * data, const idx_t * sequence_ids, const idx_t * group_ids, 
    const size_t * group_positions, float * base_norms, const bool base_norm_flag){
        time_recorder batch_recorder = time_recorder();
        bool show_batch_time = true;

        std::vector<float> residuals(n * dimension);
        //Compute residuals
        encode(n, data, group_ids, residuals.data());
        if (show_batch_time) batch_recorder.print_time_usage("compute residuals                 ");

        //Compute code for residuals
        std::vector<uint8_t> batch_codes(n * this->code_size);
        this->pq.compute_codes(residuals.data(), batch_codes.data(), n);
        if (show_batch_time) batch_recorder.print_time_usage("encode data residuals             ");

        //Add codes into index
        for (size_t i = 0; i < n; i++){
            idx_t group_id = group_ids[i];
            size_t group_position = group_positions[i];
            for (size_t j = 0; j < this->code_size; j++){this->base_codes[group_id][group_position * code_size + j] = batch_codes[i * this->code_size + j];}
            this->base_sequence_ids[group_id][group_position] = sequence_ids[i];
        }
        if (show_batch_time) batch_recorder.print_time_usage("add codes to index                ");

        std::vector<float> decoded_residuals(n * dimension);
        this->pq.decode(batch_codes.data(), decoded_residuals.data(), n);

        std::vector<float> reconstructed_x(n * dimension);
        decode(n, decoded_residuals.data(), group_ids, reconstructed_x.data());
        if (show_batch_time) batch_recorder.print_time_usage("compute reconstructed base vectors ");

       //This is the norm for reconstructed vectors
        if (!base_norm_flag){
            for (size_t i = 0; i < n; i++){base_norms[i] =  faiss::fvec_norm_L2sqr(reconstructed_x.data() + i * dimension, dimension);}
        }

        //The size of base_norm_code or base_norm should be initialized in main function
        if (use_norm_quantization){
            std::vector<uint8_t> xnorm_codes (n * norm_code_size);
            this->norm_pq.compute_codes(base_norms, xnorm_codes.data(), n);
            for (size_t i = 0 ; i < n; i++){
                idx_t sequence_id = sequence_ids[i];
                for (size_t j =0; j < this->norm_code_size; j++){
                    this->base_norm_codes[sequence_id * norm_code_size + j] = xnorm_codes[i * this->norm_code_size +j];
                }
            }
        }
        else{
            for (size_t i = 0; i < n; i++){
                idx_t sequence_id = sequence_ids[i];
                this->base_norms[sequence_id] = base_norms[i];
            }
        }
        if (show_batch_time) batch_recorder.print_time_usage("add base norms                     ");
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
    void Bslib_Index::compute_centroid_norm(std::string path_centroid_norm){
        this->centroid_norms.resize(final_group_num);
        
        if (exists(path_centroid_norm)){
            std::ifstream centroid_norm_input (path_centroid_norm, std::ios::binary);
            readXvecFvec<float> (centroid_norm_input, this->centroid_norms.data(), final_group_num, 1, false, false);
            centroid_norm_input.close();
        }

        else{
        if (this->index_type[layers -1] == "VQ"){
            size_t n_vq = vq_quantizer_index.size();
            size_t group_num = vq_quantizer_index[n_vq-1].nc_upper;
            size_t group_size = vq_quantizer_index[n_vq-1].nc_per_group;
            assert(final_group_num == group_num * group_size);

            std::cout << "Computing centroid norm for " << final_group_num << " centroids in VQ layer" << std::endl;

#pragma omp parallel for
            for (size_t i = 0; i < group_num; i++){
                for (size_t j = 0; j < group_size; j++){
                    std::vector<float> each_centroid(dimension);
                    vq_quantizer_index[n_vq-1].compute_final_centroid(i, j, each_centroid.data());
                    this->centroid_norms[i * group_size + j] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
                }
            }
        }
        else if (this->index_type[layers -1] == "LQ"){
            size_t n_lq = lq_quantizer_index.size();
            size_t group_num = lq_quantizer_index[n_lq - 1].nc_upper;
            size_t group_size = lq_quantizer_index[n_lq - 1].nc_per_group;
            assert(final_group_num == group_num * group_size);

            std::cout << "Computing centroid norm for " << final_group_num << " centroids in LQ layer" << std::endl;

#pragma omp parallel for
            for (size_t i = 0; i < group_num; i++){
                for (size_t j = 0; j < group_size; j++){
                    std::vector<float> each_centroid(dimension);
                    lq_quantizer_index[n_lq-1].compute_final_centroid(i, j, each_centroid.data());
                    this->centroid_norms[i * group_size + j] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
                }
            }
        }
        else if (this->index_type[layers - 1] == "PQ"){
            size_t n_pq = pq_quantizer_index.size();
            size_t group_num = pq_quantizer_index[n_pq - 1].nc_upper;
            size_t group_size = pq_quantizer_index[n_pq - 1].nc_per_group;
            assert(final_group_num == group_num * group_size);

            std::cout << "Computing centroid norm for " << final_group_num << " centroids in PQ layer" << std::endl;

#pragma omp parallel for
            for (size_t i = 0; i < final_group_num; i++){
                std::vector<float> each_centroid(dimension);
                pq_quantizer_index[n_pq - 1].compute_final_centroid(i, each_centroid.data());
                this->centroid_norms[i] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
            }
        }
        else{
            PrintMessage("The layer type not found for centroid norm computation");
            exit(0);
        }

        std::ofstream centroid_norm_output(path_centroid_norm, std::ios::binary);
        centroid_norm_output.write((char * )& final_group_num, sizeof(uint32_t));
        centroid_norm_output.write((char *) this->centroid_norms.data(), sizeof(float) * this->centroid_norms.size());
        centroid_norm_output.close();
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

    void Bslib_Index::assign(const size_t n, const float * assign_data, idx_t * assigned_ids, size_t assign_layer){

        /*
        if (index_type[layers - 1] == "VQ"){
            size_t n_vq = vq_quantizer_index.size();
            vq_quantizer_index[n_vq - 1].search_all(n, 1, assign_data, assigned_ids);
        }
        else if(index_type[layers - 1] == "LQ"){
            size_t n_lq = lq_quantizer_index.size();
            lq_quantizer_index[n_lq - 1].search_all(n, 1, assign_data, assigned_ids);
        }
        else if(index_type[layers - 1] == "PQ"){
            size_t n_pq = pq_quantizer_index.size();
            pq_quantizer_index[n_pq - 1].search_all(n, assign_data, assigned_ids);
        }*/

#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            size_t n_vq = 0, n_lq = 0, n_pq = 0; 
            std::vector<idx_t> query_search_id(1 , 0);
            std::vector<idx_t> query_result_ids;
            std::vector<float> query_result_dists;

            for (size_t j = 0; j < assign_layer; j++){
                assert(n_vq+ n_lq + n_pq == j);

                if (index_type[j] == "VQ"){

                    size_t group_size = vq_quantizer_index[n_vq].nc_per_group;
                    query_result_dists.resize(group_size, 0);
                    query_result_ids.resize(group_size, 0);
                    
                    const float * target_data = assign_data + i * dimension;
                    vq_quantizer_index[n_vq].search_in_group(target_data, query_search_id[0], query_result_dists.data(), query_result_ids.data(), 1);
                    if (use_HNSW_VQ){
                        query_search_id[j] = query_result_ids[0];
                    }
                    else{
                        std::vector<float> sub_dist(1);
                        keep_k_min(group_size, 1, query_result_dists.data(), query_result_ids.data(), sub_dist.data(), query_search_id.data()); 
                    }
                    n_vq ++;
                }

                else if(index_type[j] == "LQ") {

                    size_t group_size = lq_quantizer_index[n_lq].nc_per_group;
                    // Copy the upper search result for LQ layer 
                    size_t upper_group_size = query_result_ids.size();
                    
                    std::vector<idx_t> upper_result_ids(upper_group_size, 0);
                    std::vector<float> upper_result_dists(upper_group_size, 0);
                    memcpy(upper_result_ids.data(), query_result_ids.data(), upper_group_size * sizeof(idx_t));
                    memcpy(upper_result_dists.data(), query_result_dists.data(), upper_group_size * sizeof(float));
                    query_result_ids.resize(group_size, 0);
                    query_result_dists.resize(group_size, 0);

                    const float * target_data = assign_data + i * dimension;
                    lq_quantizer_index[n_lq].search_in_group(target_data, upper_result_ids.data(), upper_result_dists.data(), 1, query_search_id[0], query_result_dists.data(), query_result_ids.data());
                    std::vector<float> sub_dist(1);
                    keep_k_min(group_size, 1, query_result_dists.data(), query_result_ids.data(), sub_dist.data(), query_search_id.data());
                    n_lq ++;
                }

                else if(index_type[j] == "PQ"){
                    assert(j == this->layers-1);
                    const float * target_data = assign_data + i * dimension;
                    query_result_dists.resize(1);
                    query_result_ids.resize(1);

                    pq_quantizer_index[n_pq].search_in_group(target_data, query_search_id[0], query_result_dists.data(), query_result_ids.data(), 1);
                    query_search_id[0] = query_result_ids[0];
                    n_pq ++;
                }
                else{
                    std::cout << "The type name is wrong with " << index_type[j] << "!" << std::endl;
                    exit(0);
                }
            }
            assigned_ids[i] = query_search_id[0];
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
        for (size_t i = 0; i < this->M; i++) {
            result += precomputed_table[this->pq.ksub * i + code[i]];
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
     * Output:
     * id_position: this is the id position of next group id to be visited
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
        }
        else{
            std::cout << "k (" << k << ") should be smaller than m (" << m << ") in keep_k_min function " << std::endl;
            exit(0);
        }
    }

     /**
      * d = || x - y_C - y_R ||^2
      * d = || x - y_C ||^2 - || y_C ||^2 + || y_C + y_R ||^2 - 2 * (x|y_R)
      *        -----------------------------   -----------------   -----------
      *                     term 1                   term 2           term 3
     * 
     * distance = ||query - (centroids + residual_PQ)||^2
     *            ||query - centroids||^2 + ||residual_PQ|| ^2 - 2 * (query - centroids) * residual_PQ
     *            ||query - centroids||^2 + ||residual_PQ|| ^2 - 2 * query * residual_PQ + 2 * centroids * residual_PQ
     *            ||query - centroids||^2 + - ||centroids||^2 + ||residual_PQ + centroids||^2 - 2 * query * residual_PQ 
     *             accurate                                      error                          error
     * 
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

    void Bslib_Index::search(size_t n, size_t result_k, float * queries, float * query_dists, idx_t * query_ids, const size_t * keep_space, uint32_t * groundtruth, std::string path_base){
        
        /*
        std::ofstream dist_file;
        dist_file.open(path_dist, std::ios::app);
        dist_file << "The search analysis for recall @ " << result_k << std::endl;
        */

        // Variables for testing and validation and printing
        // Notice: they should only be activated when parallel is not used
        const bool validation = false; 
        size_t validation_print_space = 50; 
        const bool analysis = false; 
        const bool showmessage = false; 

        std::vector<float>  visited_gt_proportion;
        std::vector<size_t> actual_visited_vectors;
        std::vector<size_t> actual_visited_clusters;
        std::vector<std::vector<float>> time_consumption;
        std::vector<float> q_c_dists;
        std::vector<float> recall;

        if (analysis){
            visited_gt_proportion.resize(n, 0); actual_visited_vectors.resize(n, 0); actual_visited_clusters.resize(n, 0);
            time_consumption.resize(n); for (size_t i=0;i<n;i++){time_consumption[i].resize(layers+1, 0);}
            q_c_dists.resize(n, 0); recall.resize(n, 0);
        }

        std::ifstream base_input;
        if (validation){base_input = std::ifstream(path_base, std::ios::binary);}

//Use parallel in real use
//#pragma omp parallel for
        for (size_t i = 0; i < n; i++){

            //dist_file << "QUery " << i << std::endl;

            //Variables for analysis
            time_recorder Trecorder = time_recorder();
            std::unordered_set<idx_t> grountruth_set;
            if (analysis){for (size_t j = 0; j < result_k; j++){grountruth_set.insert(groundtruth[i * 100 + j]);}}

            const float * query = use_OPQ ? opq_matrix->apply(1, queries + i * dimension) : queries + i * dimension;
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
                    max_search_space = final_keep_space * ((ncentroids[j] > keep_space[j]) ? ncentroids[j] : keep_space[j]);
                }
                final_keep_space *= keep_space[j];
            }

            assert(max_search_space > 0 && final_keep_space > 0);
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
                        vq_quantizer_index[n_vq].search_in_group(query, query_group_ids[m], query_result_dists.data()+m*group_size, query_result_labels.data() + m*group_size, keep_space[j]);
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
                        lq_quantizer_index[n_lq].search_in_group(query, upper_result_labels.data(), upper_result_dists.data(), search_space, query_group_ids[m], query_result_dists.data()+m*group_size, query_result_labels.data()+m*group_size);
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
                        pq_quantizer_index[n_pq].search_in_group(query, query_group_ids[m], query_result_dists.data()+m*keep_space[j], query_result_labels.data()+m*keep_space[j], keep_space[j]);
                    }
                    assert(keep_result_space * keep_space[j] == final_keep_space);
                    memcpy(query_group_ids.data(), query_result_labels.data(), final_keep_space * sizeof(idx_t));
                    memcpy(query_group_dists.data(), query_result_dists.data(), final_keep_space * sizeof(float));
                    n_pq ++;
                }
                else{
                    std::cout << "The type name is wrong with " << index_type[j] << "!" << std::endl;
                    exit(0);
                }
                search_space = keep_result_space * group_size;
                keep_result_space = keep_result_space * keep_space[j];

                if (analysis){time_consumption[i][j] = Trecorder.getTimeConsumption(); Trecorder.reset();}
            }

            if (showmessage) std::cout << "Finished search in index centroids, show the results" << std::endl;
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

            size_t max_size = 0;for (size_t j = 0; j < final_keep_space; j++){if (base_sequence_ids[query_group_ids[j]].size() > max_size) max_size = base_sequence_ids[query_group_ids[j]].size();}
            
            if(showmessage) std::cout << "Assigning the space for dists and labels with size " << max_visited_vectors << " + " << max_size <<  std::endl;
            std::vector<float> query_search_dists(max_visited_vectors + max_size, 0);
            std::vector<idx_t> query_search_labels(max_visited_vectors + max_size, 0);
            
            
            // The dists for actual distance validation
            std::vector<float> query_actual_dists; if (validation){(query_actual_dists.resize(max_visited_vectors + max_size));}

            size_t j = 0;
            if (showmessage) std::cout << "Searching the base vectors " << std::endl;
            for (j = 0; j < final_keep_space; j++){

                std::pair<idx_t, float> result_idx_dist;
                get_next_group_idx(final_keep_space, query_group_ids.data(), query_group_dists.data(), result_idx_dist);
                
                idx_t group_id = result_idx_dist.first;
                float q_c_dist = result_idx_dist.second;
                if (showmessage) std::cout << "Searching in " << group_id << " th group with distance " << q_c_dist << std::endl;

                size_t group_size = this->base_sequence_ids[group_id].size();
                assert(group_size == this->base_codes[group_id].size() / this->code_size);

                float centroid_norm;
                idx_t centroid_id = group_id;

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

                    idx_t sequence_id = base_sequence_ids[group_id][m];
                    std::vector<float> base_reconstructed_norm(1);
                    if (use_norm_quantization) norm_pq.decode(base_norm_codes.data() + sequence_id * norm_code_size, base_reconstructed_norm.data(), 1);
                    float term2 = use_norm_quantization ? base_reconstructed_norm[0] : base_norms[sequence_id];
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
                        float actual_dist =  faiss::fvec_L2sqr(base_vector_float.data(), query, dimension);
                        std::vector<float> decoded_code(dimension);
                        pq.decode(code + m * code_size, decoded_code.data(), 1);
                        std::vector<float> decoded_base_vector(dimension);
                        decode(1, decoded_code.data(), & group_id, decoded_base_vector.data());
                        float actual_norm = faiss::fvec_norm_L2sqr(decoded_base_vector.data(), dimension);
                        
                        query_actual_dists[visited_vectors] = actual_dist;
                        float product_term3 = 2 * faiss::fvec_inner_product(query, decoded_code.data(), dimension);
                        
                        std::vector<float> b_c_residual(dimension);
                        std::vector<float> centroid(dimension);
                        get_final_centroid(centroid_id, centroid.data());
                        faiss::fvec_madd(dimension, base_vector_float.data(), -1.0, centroid.data(), b_c_residual.data());
                        float actual_term3 = 2 * faiss::fvec_inner_product(query, b_c_residual.data(), dimension);
                        std::vector<uint8_t> base_code(this->code_size);
                        pq.compute_code(b_c_residual.data(), base_code.data());
                        std::vector<float> reconstructed_residual(dimension);
                        pq.decode(base_code.data(), reconstructed_residual.data(), 1);
                        float test_term3 = 2 * faiss::fvec_inner_product(query, reconstructed_residual.data(), dimension);
                        

                        std::cout << " S: " << dist << " A: " << actual_dist << " LN: " << term2 << " AN: " << actual_norm << " " << " Term1: " << q_c_dist << " " << centroid_norm << " Term3: " << term3 << " " << "Term3 inner: " << product_term3 <<  " Term3 Test: " << test_term3 << " Actual term3: " << actual_term3 << std::endl; // S for "Search" and A for "Actual"
                    }

                    query_search_labels[visited_vectors] = base_sequence_ids[group_id][m];
                    visited_vectors ++;
                    if (analysis){if (grountruth_set.count(base_sequence_ids[group_id][m]) != 0){visited_gt ++;}}

                }
                if (visited_vectors >= this->max_visited_vectors)
                    break;
            }

            /*
            std::vector<float> temp_search_dist(100);
            std::vector<idx_t> temp_search_ids(100);
            std::vector<float> actual_search_dist(100);

            keep_k_min(visited_vectors, 100, query_search_dists.data(), query_search_labels.data(), temp_search_dist.data(), temp_search_ids.data());
            std::ifstream reranking_input(path_base, std::ios::binary);
            std::vector<float> reranking_dists(100, 0);
            std::vector<idx_t> reranking_labels(100, 0);
            std::vector<float> reranking_actual_dists(100, 0);

            for (size_t temp = 0; temp < 100; temp++){
                std::vector<base_data_type> base_vector(dimension);
                uint32_t dim;
                reranking_input.seekg(temp_search_ids[temp] * dimension * sizeof(base_data_type) + temp_search_ids[temp] * sizeof(uint32_t), std::ios::beg);
                reranking_input.read((char *)& dim, sizeof(uint32_t));
                reranking_input.read((char *) base_vector.data(), sizeof(base_data_type) * dimension);
                assert(dim == dimension);
                actual_search_dist[temp] = faiss::fvec_L2sqr(base_vector.data(), query, dimension);
            }
            std::vector<idx_t> search_dist_index(100);
            uint32_t x = 0;
            std::iota(search_dist_index.begin(), search_dist_index.end(), x++);
            std::sort(search_dist_index.begin(), search_dist_index.end(), [&](int i, int j){return temp_search_dist[i] < temp_search_dist[j];});

            
            for (size_t temp = 0; temp < 100; temp++){
                if (temp == 9){
                    dist_file << std::endl;
                }
                dist_file << temp_search_ids[search_dist_index[temp]] << " " << temp_search_dist[search_dist_index[temp]] << " " << actual_search_dist[search_dist_index[temp]] << " ";
            }
            dist_file << std::endl;

            for (size_t temp = 0; temp < 100; temp++){
                std::vector<base_data_type> base_vector(dimension);
                uint32_t dim;
                reranking_input.seekg(groundtruth[i* 100 + temp] * dimension * sizeof(base_data_type) + groundtruth[i* 100 + temp] * sizeof(uint32_t), std::ios::beg);
                reranking_input.read((char *)& dim, sizeof(uint32_t));
                reranking_input.read((char *) base_vector.data(), sizeof(base_data_type) * dimension);
                actual_search_dist[temp] = faiss::fvec_L2sqr(base_vector.data(), query, dimension);
            }
            for (size_t temp = 0; temp < 100; temp++){
                if (temp == 9){
                    dist_file << std::endl;
                }
                dist_file << groundtruth[i * 100 + temp] <<  " " << actual_search_dist[temp] << " ";
            }
            dist_file << std::endl;
            */

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

                size_t correct = 0; for (size_t temp=0;temp<result_k;temp++){if(grountruth_set.count(query_ids[ i * result_k + temp])!=0) correct++;}
                recall[i] = float(correct) / result_k;
                time_consumption[i][layers]  = Trecorder.getTimeConsumption();
                visited_gt_proportion[i] = float(visited_gt) / result_k;
                actual_visited_vectors[i] = visited_vectors;
                actual_visited_clusters[i] = j;
            }
        }

        if (analysis){
            float avg_visited_vectors = 0, avg_q_c_dist = 0, avg_visited_gt_proportion = 0, avg_recall = 0, avg_visited_clusters = 0;
            std::vector<float> avg_time_consumption(layers+1, 0);
            for (size_t i = 0; i < n; i++){
                avg_visited_vectors += actual_visited_vectors[i];
                avg_visited_clusters += actual_visited_clusters[i];
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
            std::cout << "The average visited clusters: " << avg_visited_clusters / n << std::endl;
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
                    std::cout << "Writing L2 centroids " << std::endl;
                    for (size_t j = 0; j < nc_upper; j++){
                        size_t group_quantizer_data_size = nc_per_group * this->dimension;
                        assert(vq_quantizer_index[n_vq].L2_quantizers[j]->xb.size() == group_quantizer_data_size);
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
            size_t base_size = base_norm_codes.size() / norm_code_size;
            output.write((char *) & base_size, sizeof(size_t));
            output.write((char *) base_norm_codes.data(), base_norm_codes.size() * sizeof(uint8_t));
        }
        else{
            size_t base_size = base_norms.size();
            output.write((char *) & base_size, sizeof(size_t));
            output.write((char *)base_norms.data(), base_norms.size() * sizeof(float));
        }

        output.write((char *) & this->final_group_num, sizeof(size_t));
        assert(base_codes.size() == final_group_num);
        for (size_t i = 0; i < this->final_group_num; i++){
            assert(base_codes[i].size() / code_size == base_sequence_ids[i].size());
            size_t group_size = base_codes[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) base_codes[i].data(), group_size * sizeof(uint8_t));
        }


        output.write((char *) & this->final_group_num, sizeof(size_t));
        assert(base_sequence_ids.size() == final_group_num);
        for (size_t i = 0; i < this->final_group_num; i++){
            size_t group_size = base_sequence_ids[i].size();
            output.write((char *) & group_size, sizeof(size_t));
            output.write((char *) base_sequence_ids[i].data(), group_size * sizeof(idx_t));
        }

        assert(centroid_norms.size() == final_group_num);
        output.write((char *) & this->final_group_num, sizeof(size_t));
        assert(centroid_norms.size() == this->final_group_num);
        output.write((char *) centroid_norms.data(), this->final_group_num * sizeof(float));

        output.close();
    }

    /*
        Read the index file to get the whole constructed index
        Input: 
            path_index:    str, the path to get the index
        
        Output:
            None (The index is updated)

    */
    void Bslib_Index::read_index(const std::string path_index){
        std::ifstream input(path_index, std::ios::binary);
        size_t final_nc_input;
        size_t group_size_input;

        this->base_codes.resize(this->final_group_num);
        
        this->base_sequence_ids.resize(this->final_group_num);
        input.read((char *) & final_nc_input, sizeof(size_t));

        if (use_norm_quantization){
            size_t base_size;
            input.read((char *) & base_size, sizeof(size_t));
            this->base_norm_codes.resize(base_size);
            input.read((char *) base_norm_codes.data(), base_size * sizeof(uint8_t));
        }
        else{
            size_t base_size;
            input.read((char *) & base_size, sizeof(size_t));
            this->base_norms.resize(base_size);
            input.read((char *) base_norms.data(), base_size * sizeof(float));
        }

        input.read((char *) & final_nc_input, sizeof(size_t));
        std::cout << final_nc_input << " " << final_group_num << std::endl;
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

        input.read((char *) & final_nc_input, sizeof(size_t));
        assert(final_nc_input == this->final_group_num);
        this->centroid_norms.resize(this->final_group_num);
        input.read((char *) centroid_norms.data(), this->final_group_num * sizeof(float));
        input.close();
    }

    /*
        Construct the index with the train set
        Input:
            path_learn:    str, the path to read the dataset
            

        Output: 
            None  (The index is updated)

    */
    void Bslib_Index::build_index(const size_t M_PQ, std::string path_learn, std::string path_groups, std::string path_labels,
    std::string path_quantizers, uint32_t VQ_layers, uint32_t PQ_layers, std::string path_OPQ, 
    const uint32_t * ncentroids, const size_t * M_HNSW, const size_t * efConstruction, 
    const size_t * efSearch, const size_t * M_PQ_layer, const size_t * nbits_PQ_layer, const size_t * num_train,
    size_t OPQ_train_size, size_t selector_train_size, size_t selector_group_size, std::ofstream & record_file){

        PrintMessage("Initializing the index");
        Trecorder.reset();
        if (use_OPQ){        
            if (exists(path_OPQ)){
                this->opq_matrix = static_cast<faiss::OPQMatrix *>(faiss::read_VectorTransform(path_OPQ.c_str()));
            }
            else{
                PrintMessage("Training the OPQ matrix");
                this->opq_matrix = new faiss::OPQMatrix(dimension, M_PQ);
                this->opq_matrix->verbose = true;
                std::ifstream learn_input(path_learn, std::ios::binary);
                std::vector<float>  origin_train_set(train_size * dimension);
                readXvecFvec<learn_data_type>(learn_input, origin_train_set.data(), dimension, train_size, false);
                
                if (OPQ_train_size < train_size){
                    std::vector<float> OPQ_train_set(OPQ_train_size * dimension);
                    RandomSubset(origin_train_set.data(), OPQ_train_set.data(), dimension, train_size, OPQ_train_size);
                    std::cout<< "Randomly select the train set for OPQ training" << std::endl;
                    this->opq_matrix->train(OPQ_train_size, OPQ_train_set.data());
                }
                else{
                    this->opq_matrix->train(train_size, origin_train_set.data());
                }
                faiss::write_VectorTransform(this->opq_matrix, path_OPQ.c_str());
            }
            if (is_recording)
            {
                std::string message = "Trained the OPQ matrix";
                Mrecorder.print_memory_usage(message);
                Mrecorder.record_memory_usage(record_file,  message);
                Trecorder.print_time_usage(message);
                Trecorder.record_time_usage(record_file, message);
            }
        }

        if (use_train_selector){
            this->build_train_selector(path_learn, path_groups, path_labels, train_size, selector_train_size, selector_group_size);
        }

        std::vector<HNSW_para> HNSW_paras;
        if (use_HNSW_VQ){
            for (size_t i = 0; i < VQ_layers; i++){
                HNSW_para new_para; new_para.first.first = M_HNSW[i]; new_para.first.second = efConstruction[i]; new_para.second = efSearch[i];
                HNSW_paras.push_back(new_para);
            }
        }
        std::vector<PQ_para> PQ_paras;
        for (size_t i = 0; i < PQ_layers; i++){
            PQ_para new_para; new_para.first = M_PQ_layer[i]; new_para.second = nbits_PQ_layer[i];
            PQ_paras.push_back(new_para);
        }

        PrintMessage("Constructing the quantizers");
        this->build_quantizers(ncentroids, path_quantizers, path_learn, num_train, HNSW_paras, PQ_paras);
        this->get_final_group_num();

        if (is_recording){
            std::string message = "Constructed the index, ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }

    /*
        Assign the whole dataset to the index
        Input: 
            path:            str, the path to read the dataset
            batch_size:      int, number of vectors in each batch
            nbatches:        int, number of batches to be partitioned

        Output:
            None    (The index is updated)
    */
    void Bslib_Index::assign_vectors(std::string path_ids, std::string path_base, 
        uint32_t batch_size, size_t nbatches, std::ofstream & record_file){

        PrintMessage("Assigning the points");
        Trecorder.reset();
        if (!exists(path_ids)){
            std::ifstream base_input (path_base, std::ios::binary);
            std::ofstream base_output (path_ids, std::ios::binary);

            std::vector <float> batch(batch_size * dimension);
            std::vector<idx_t> assigned_ids(batch_size);

            for (size_t i = 0; i < nbatches; i++){
                readXvecFvec<base_data_type> (base_input, batch.data(), dimension, batch_size, false, false);
                if (use_OPQ) {this->do_OPQ(batch_size, batch.data());}
                this->assign(batch_size, batch.data(), assigned_ids.data(), this->layers);
                base_output.write((char * ) & batch_size, sizeof(uint32_t));
                base_output.write((char *) assigned_ids.data(), batch_size * sizeof(idx_t));
                if (i % 10 == 0){
                    std::cout << " assigned batches [ " << i << " / " << nbatches << " ]";

                    Trecorder.print_time_usage("");
                    record_file << " assigned batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.record_time_usage(record_file, " ");
                }
            }
            base_input.close();
            base_output.close();
            if (is_recording){
                std::string message = "Assigned the base vectors in sequential mode";
                Mrecorder.print_memory_usage(message);
                Mrecorder.record_memory_usage(record_file,  message);
                Trecorder.print_time_usage(message);
                Trecorder.record_time_usage(record_file, message);
            }
        }
    }

    /*
        Train the PQ quantizer
        Input:
        Output:
    */
    void Bslib_Index::train_pq_quantizer(std::string path_pq, std::string path_pq_norm,
        size_t M_norm_PQ, std::string path_learn, size_t PQ_train_size,  std::ofstream & record_file){
        
        PrintMessage("Constructing the PQ compressor");
        Trecorder.reset();
        if (exists(path_pq)){
            std::cout << "Loading PQ codebook from " << path_pq << std::endl;
            this->pq = * faiss::read_ProductQuantizer(path_pq.c_str());
            this->code_size = this->pq.code_size;

            if(use_norm_quantization){
                std::cout << "Loading norm PQ codebook from " << path_pq_norm << std::endl;
                this->norm_pq = * faiss::read_ProductQuantizer(path_pq_norm.c_str());
                this->norm_code_size = this->norm_pq.code_size;
            }
        }
        else
        {
            this->norm_M = M_norm_PQ;

            std::cout << "Training PQ codebook" << std::endl;
            this->train_pq(path_pq, path_pq_norm, path_learn, PQ_train_size);
        }
        std::cout << "Checking the PQ with its code size:" << this->pq.code_size << std::endl;

        if (is_recording){
            std::string message = "Trained the PQ, ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }
    
    void Bslib_Index::load_index(std::string path_index, std::string path_ids, std::string path_base,
        std::string path_base_norm, std::string path_centroid_norm, size_t batch_size, size_t nbatches, size_t nb, std::ofstream & record_file){
        Trecorder.reset();
        if (exists(path_index)){
            PrintMessage("Loading pre-constructed index");
            
            read_index(path_index);
            if (is_recording){
                std::string message = "Loaded pre-constructed index ";
                Mrecorder.print_memory_usage(message);
                Mrecorder.record_memory_usage(record_file,  message);
                Trecorder.print_time_usage(message);
                Trecorder.record_time_usage(record_file, message);
            }
        }
        else{
            PrintMessage("Loading the index");
            std::vector<idx_t> ids(nb); 
            std::ifstream ids_input(path_ids, std::ios::binary);
            readXvec<idx_t> (ids_input, ids.data(), batch_size, nbatches); 
            //std::vector<idx_t> pre_hash_ids; if (use_hash) {pre_hash_ids.resize(nb, 0); memcpy(pre_hash_ids.data(), ids.data(), nb * sizeof(idx_t)); HashMapping(nb, pre_hash_ids.data(), ids.data(), index.final_group_num);}
            std::vector<size_t> groups_size(this->final_group_num, 0); std::vector<size_t> group_position(nb, 0);
            for (size_t i = 0; i < nb; i++){group_position[i] = groups_size[ids[i]]; groups_size[ids[i]] ++;}

            this->base_codes.resize(this->final_group_num);
            this->base_sequence_ids.resize(this->final_group_num);
            if (use_norm_quantization){this->base_norm_codes.resize(nb);} else{this->base_norms.resize(nb);}
            for (size_t i = 0; i < this->final_group_num; i++){
                this->base_codes[i].resize(groups_size[i] * this->code_size);
                this->base_sequence_ids[i].resize(groups_size[i]);
            }
            
            
            std::ifstream base_input(path_base, std::ios::binary);
            std::vector<float> base_batch(batch_size * dimension);
            std::vector<idx_t> batch_sequence_ids(batch_size);

            
            std::vector<float> xnorms(nb);
            bool base_norm_flag = false;
            if (exists(path_base_norm)){
                base_norm_flag = true;
                std::cout << "Loading pre-computed base norm " << std::endl;
                std::ifstream base_norm_input(path_base_norm, std::ios::binary);
                readXvecFvec<float>(base_norm_input, xnorms.data(), nb, 1, false, false);
                base_norm_input.close();
            }

            std::cout << "Start adding batches " << std::endl;
            for (size_t i = 0; i < nbatches; i++){
                readXvecFvec<base_data_type> (base_input, base_batch.data(), dimension, batch_size);
                if (use_OPQ) {this->do_OPQ(batch_size, base_batch.data());}
                for (size_t j = 0; j < batch_size; j++){batch_sequence_ids[j] = batch_size * i + j;}

                this->add_batch(batch_size, base_batch.data(), batch_sequence_ids.data(), ids.data() + i * batch_size, group_position.data()+i*batch_size, xnorms.data() + i * batch_size, base_norm_flag);
                if (i % 10 == 0){
                    std::cout << " adding batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.print_time_usage("");
                    record_file << " adding batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.record_time_usage(record_file, " ");
                }
            }

            if (!base_norm_flag){
                std::ofstream base_norm_output(path_base_norm, std::ios::binary);
                base_norm_output.write((char * )& nb, sizeof(uint32_t));
                base_norm_output.write((char *) xnorms.data(), xnorms.size() * sizeof(float));
                base_norm_output.close();
            }

            this->compute_centroid_norm(path_centroid_norm);

            //In order to save disk usage
            //Annotate the write_index function
            if (this->saving_index){
                //this->write_index(path_index);
            }
            
            std::string message = "Constructed and wrote the index ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }

    void Bslib_Index::index_statistic(){}


    void Bslib_Index::query_test(size_t num_search_paras, size_t num_recall, size_t nq, size_t ngt,
        const size_t * max_vectors, const size_t * result_k, const size_t * keep_space, const size_t * reranking_space,
        std::ofstream & record_file, std::ofstream & qps_record_file, 
        std::string search_mode, std::string path_base, std::string path_gt, std::string path_query){

        PrintMessage("Loading groundtruth");
        std::vector<uint32_t> groundtruth(nq * ngt);
        {
            std::ifstream gt_input(path_gt, std::ios::binary);
            readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, false, false);
        }

        PrintMessage("Loading queries");
        std::vector<float> queries(nq * dimension);
        {
            std::ifstream query_input(path_query, std::ios::binary);
            readXvecFvec<base_data_type>(query_input, queries.data(), dimension, nq, false, false);
        }

        for (size_t i = 0; i < num_search_paras; i++){

        this->max_visited_vectors = max_vectors[i];
        for (size_t j = 0; j < num_recall; j++){
            if (this->use_reranking) this->reranking_space = reranking_space[j];
            size_t recall_k = result_k[j];
            std::vector<float> query_distances(nq * recall_k);
            std::vector<faiss::Index::idx_t> query_labels(nq * recall_k);
            size_t correct = 0;
            
            Trecorder.reset();
            search(nq, recall_k, queries.data(), query_distances.data(), query_labels.data(), keep_space+ i * layers, groundtruth.data(), path_base);
            std::cout << "The qps for searching is: " << Trecorder.getTimeConsumption() / nq << " us " << std::endl;
            std::string message = "Finish Search ";
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
            Trecorder.record_time_usage(qps_record_file, message);
            

            for (size_t i = 0; i < nq; i++){
                std::unordered_set<idx_t> gt;

                //for (size_t j = 0; j < recall_k; j++){
                for (size_t j = 0; j < 1; j++){
                    gt.insert(groundtruth[ngt * i + j]);
                }

                //assert (gt.size() == recall_k);
                
                for (size_t j = 0; j < recall_k; j++){
                    if (gt.count(query_labels[i * recall_k + j]) != 0)
                        correct ++;
                }
            }
            //float recall = float(correct) / (recall_k * nq);
            float recall = float(correct) / (nq);
            Rrecorder.print_recall_performance(nq, recall, recall_k, search_mode, layers, keep_space + i * layers, max_vectors[i]);
            Rrecorder.record_recall_performance(record_file, nq, recall, recall_k, search_mode, layers, keep_space + i * layers, max_vectors[i]);
            Rrecorder.record_recall_performance(qps_record_file, nq, recall, recall_k, search_mode, layers, keep_space + i * layers, max_vectors[i]);

            if (use_reranking){
                std::cout << " with reranking parameter: " << this->reranking_space << std::endl;
            } 
            std::cout << std::endl;
        }
        }
    }


}

