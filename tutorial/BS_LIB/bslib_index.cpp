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
    bool * use_VQ_HNSW, const bool use_group_HNSW, const bool use_all_HNSW, const bool use_OPQ, const bool use_train_selector,
    const size_t train_size, const size_t M_pq, const size_t nbits, const size_t group_HNSW_thres){
            
            this->dimension = dimension;
            this->layers = layers;

            this->use_reranking = use_reranking;
            this->use_VQ_HNSW = use_VQ_HNSW;
            this->use_group_HNSW = use_group_HNSW;
            this->use_all_HNSW = use_all_HNSW;
            this->use_vector_alpha = false;

            this->use_norm_quantization = use_norm_quantization;
            this->use_OPQ = use_OPQ;
            this->use_train_selector = use_train_selector;

            this->use_recording = is_recording;
            this->use_saving_index = saving_index;

            this->index_type.resize(layers);
            this->ncentroids.resize(layers);

            for (size_t i = 0; i < layers; i++){
                this->index_type[i] = index_type[i];
            }

            this->train_size = train_size;
            this->M_pq = M_pq;
            this->nbits = nbits;
            this->group_HNSW_thres = group_HNSW_thres;
        }
    
    /**
     * The function for adding a VQ layer in the whole structure
     * 
     * Parameters required for building a VQ layer: 
     * If use L2 quantizer: nc_upper, nc_group 
     * Use HNSW quantizer: nc_upper, nc_group, M, efConstruction
     * 
     **/
    void Bslib_Index::add_vq_quantizer(size_t nc_upper, size_t nc_per_group, bool use_HNSW, size_t M, size_t efConstruction){
        VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, nc_per_group, M, efConstruction, use_HNSW, use_all_HNSW);
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
    void Bslib_Index::add_lq_quantizer(size_t nc_upper, size_t nc_per_group, const float * upper_centroids, const idx_t * upper_nn_centroid_idxs, const float * upper_nn_centroid_dists, size_t LQ_type){
        LQ_quantizer lq_quantizer = LQ_quantizer(dimension, nc_upper, nc_per_group, upper_centroids, upper_nn_centroid_idxs, upper_nn_centroid_dists, LQ_type, use_all_HNSW);
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
    void Bslib_Index::encode(size_t n, const float * encode_data, const idx_t * encoded_ids, float * encoded_data, const float * alphas){
        if (index_type[layers-1] == "VQ"){
            vq_quantizer_index[vq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data);
        }
        else if(index_type[layers-1] == "LQ"){
            lq_quantizer_index[lq_quantizer_index.size()-1].compute_residual_group_id(n, encoded_ids, encode_data, encoded_data, alphas);
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
    void Bslib_Index::decode(size_t n, const float * encoded_data, const idx_t * encoded_ids, float * decoded_data, const float * alphas){
        if (index_type[layers-1] == "VQ"){
            vq_quantizer_index[vq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data);
        }
        else if (index_type[layers-1] == "LQ"){
            lq_quantizer_index[lq_quantizer_index.size()-1].recover_residual_group_id(n, encoded_ids, encoded_data, decoded_data, alphas);
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
    void Bslib_Index::do_OPQ(size_t n, float * dataset){
        assert(& opq_matrix != NULL);
        std::vector<float> copy_dataset(n * dimension);
        memcpy(copy_dataset.data(), dataset, n * dimension * sizeof(float));
        opq_matrix.apply_noalloc(n, copy_dataset.data(), dataset);
    }

    void Bslib_Index::reverse_OPQ(size_t n, float * dataset){
        std::vector<float> copy_dataset(n * dimension);
        memcpy(copy_dataset.data(), dataset, n * dimension * sizeof(float));
        opq_matrix.transform_transpose(n, copy_dataset.data(), dataset);
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
            std::cout << "Reading subset with " << train_set_size << " without selector from " << train_size << " vectors" << std::endl;
            RandomSubset<float>(sum_train_data.data(), this->train_data.data(), dimension, total_size, train_set_size);
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
    void Bslib_Index::build_quantizers(const uint32_t * ncentroids, const std::string path_quantizer, const std::string path_learn, const size_t * num_train, 
    const std::vector<HNSW_PQ_para> HNSW_paras, const std::vector<HNSW_PQ_para> PQ_paras, const size_t * LQ_type, std::ofstream & record_file){
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
        time_recorder inner_Trecorder = time_recorder();
        
        size_t n_vq = 0;
        size_t n_lq = 0;
        size_t n_pq = 0;
        for (size_t i = 0; i < layers; i++){
            inner_Trecorder.reset();
            assert(n_vq + n_lq + n_pq == i);
            bool update_ids = (i == layers-1) ? false:true;
            if (i < layers-1 && index_type[i+1] == "LQ" && LQ_type[n_lq] == 2){update_ids = false;}
            nc_per_group = index_type[i] == "PQ" ? 0 : ncentroids[i];
            this->ncentroids[i] = nc_per_group;
            if (nc_per_group > this->max_group_size){this->max_group_size = nc_per_group;}

            if (index_type[i] == "VQ"){

                std::cout << "Adding VQ quantizer with parameters: " << nc_upper << " " << nc_per_group << std::endl;

                size_t existed_VQ_layers = this->vq_quantizer_index.size();
                HNSW_PQ_para para = HNSW_paras[existed_VQ_layers];
                add_vq_quantizer(nc_upper, nc_per_group, use_VQ_HNSW[existed_VQ_layers], para.first, para.second);


                inner_Trecorder.record_time_usage(record_file, "Trained VQ layer");
                nc_upper = vq_quantizer_index[vq_quantizer_index.size()-1].layer_nc;

                //Prepare train set for the next layer
                if (update_ids){
                    read_train_set(path_learn, this->train_size, num_train[i+1]);
                    std::cout << "Updating train set for the next layer" << std::endl;
                    assign(train_data_ids.size(), train_data.data(), train_data_ids.data(), i+1);
                    inner_Trecorder.record_time_usage(record_file, "Updated train set from VQ layer for next layer with assign function");
                    //vq_quantizer_index[vq_quantizer_index.size() - 1].search_all(train_data_ids.size(), 1, train_data.data(), train_data_ids.data());
                    //inner_Trecorder.record_time_usage(record_file, "Updated train set for next layer with search_all function");
                }
                
                std::cout << "Trainset Sample" << std::endl;
                for (size_t temp = 0; temp <2; temp++)
                {for (size_t temp1 = 0; temp1 < dimension; temp1++)
                    {std::cout << this->train_data[temp * dimension + temp1] << " ";}
                        std::cout << train_data_ids[temp];std::cout << std::endl;}

                std::cout << i << "th VQ quantizer added, check it " << std::endl;
                std::cout << "The vq quantizer size is: " <<  vq_quantizer_index.size() << " the num of L2 quantizers (groups): " << vq_quantizer_index[vq_quantizer_index.size()-1].L2_quantizers.size() << 
                " the num of HNSW quantizers (groups): " <<  vq_quantizer_index[vq_quantizer_index.size()-1].HNSW_quantizers.size() << std::endl;
                std::cout << "The total number of nc in this layer: " << vq_quantizer_index[vq_quantizer_index.size()-1].layer_nc << std::endl;
                n_vq++;
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
                    assert(vq_quantizer_index[last_vq].layer_nc > nc_per_group);
                    vq_quantizer_index[last_vq].compute_nn_centroids(nc_per_group, upper_centroids.data(), nn_centroids_dists.data(), nn_centroids_idxs.data());
                    add_lq_quantizer(nc_upper, nc_per_group, upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data(), LQ_type[n_lq]);
                }
                else if (index_type[i-1] == "LQ"){
                    PrintMessage("Adding LQ quantizer with LQ upper layer");
                    size_t last_lq = lq_quantizer_index.size() - 1;
                    PrintMessage("LQ computing nn centroids");
                    assert(lq_quantizer_index[last_lq].layer_nc > nc_per_group);
                    lq_quantizer_index[last_lq].compute_nn_centroids(nc_per_group, upper_centroids.data(), nn_centroids_dists.data(), nn_centroids_idxs.data());
                    add_lq_quantizer(nc_upper, nc_per_group, upper_centroids.data(), nn_centroids_idxs.data(), nn_centroids_dists.data(), LQ_type[n_lq]);
                }

                inner_Trecorder.record_time_usage(record_file, "Trained LQ layer");
                nc_upper = lq_quantizer_index[lq_quantizer_index.size()-1].layer_nc;

                //Prepare train set for the next layer
                if (update_ids){
                    read_train_set(path_learn, this->train_size, num_train[i+1]);
                    std::cout << "Updating train set for the next layer" << std::endl;
                    assign(train_data_ids.size(), train_data.data(), train_data_ids.data(), i+1);
                    inner_Trecorder.record_time_usage(record_file, "Updated train set from LQ for next layer with assign function");
                    //lq_quantizer_index[lq_quantizer_index.size() - 1].search_all(train_data_ids.size(), 1, train_data.data(), train_data_ids.data());
                    //inner_Trecorder.record_time_usage(record_file, "Updated train set for next layer with search_all function");
                }

                std::cout << i << "th LQ quantizer added, check it " << std::endl;
                std::cout << "The LQ quantizer size is: " <<  lq_quantizer_index.size() << " the num of alphas: " << lq_quantizer_index[lq_quantizer_index.size()-1].alphas.size() << std::endl;;
                std::cout << "The number of final nc in this layer: " << lq_quantizer_index[lq_quantizer_index.size()-1].layer_nc << std::endl;
                n_lq ++;
            }
            else if (index_type[i] == "PQ"){
                //The PQ layer should be placed in the last layer
                assert(i == layers-1);
                PrintMessage("Adding PQ quantizer");
                add_pq_quantizer(nc_upper, PQ_paras[0].first, PQ_paras[0].second);

                inner_Trecorder.record_time_usage(record_file, "Trained PQ layer");
                nc_upper = pq_quantizer_index[pq_quantizer_index.size()-1].layer_nc;
                std::cout << i << "th PQ quantizer added, check it " << std::endl;
                std::cout << "The number of final nc in this layer: " << pq_quantizer_index[pq_quantizer_index.size()-1].layer_nc << std::endl;
                n_pq ++;
            }
        }
        if(use_saving_index)
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
    void Bslib_Index::train_pq(const std::string path_pq, const std::string path_norm_pq, const std::string path_learn, const std::string path_OPQ, const size_t train_set_size){

        // Load the train set fot training
        read_train_set(path_learn, this->train_size, train_set_size);

        std::cout << "Initilizing index PQ quantizer " << std::endl;
        this->pq = faiss::ProductQuantizer(this->dimension, this->M_pq, this->nbits);
        this->code_size = this->pq.code_size;

        std::cout << "Assigning the train dataset to compute residual" << std::endl;
        std::vector<float> residuals(train_set_size * dimension);

        std::vector<float> train_vector_alphas;
        if (use_vector_alpha){
            train_vector_alphas.resize(train_set_size);
        }
        assign(train_set_size, this->train_data.data(), train_data_ids.data(), this->layers, train_vector_alphas.data());
        

        for (size_t i = train_set_size - 100; i < train_set_size; i++){std::cout << train_data_ids[i] << " ";} std::cout << std::endl;

        std::cout << "Encoding the train dataset with " << train_set_size<< " data points " << std::endl;
        encode(train_set_size, this->train_data.data(), train_data_ids.data(), residuals.data(), train_vector_alphas.data());

        if (use_OPQ){
            PrintMessage("Training the OPQ matrix");
            this->opq_matrix = faiss::OPQMatrix(dimension, M_pq);
            this->opq_matrix.verbose = true;
            this->opq_matrix.train(train_set_size, residuals.data());
            faiss::write_VectorTransform(& this->opq_matrix, path_OPQ.c_str());
            do_OPQ(train_set_size, residuals.data());
        }

        std::cout << "Training the pq " << std::endl;
        this->pq.verbose = true;
        this->pq.train(train_set_size, residuals.data());

        if(use_saving_index){
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
            this->final_group_num =  vq_quantizer_index[vq_quantizer_index.size() -1].layer_nc;
        }
        else if (this->index_type[layers -1] == "LQ"){
            this->final_group_num =  lq_quantizer_index[lq_quantizer_index.size() -1].layer_nc;
        }
        else if (this->index_type[layers - 1] == "PQ"){
            this->final_group_num = pq_quantizer_index[pq_quantizer_index.size() -1].layer_nc;
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
    const size_t * group_positions, const bool base_norm_flag, const bool alpha_flag, const float * vector_alpha, const float * vector_alpha_norm){
        time_recorder batch_recorder = time_recorder();
        bool show_batch_time = true;

        std::vector<float> residuals(n * dimension);
        //Compute residuals
        encode(n, data, group_ids, residuals.data(), vector_alpha);

        this->b_c_dist += faiss::fvec_norm_L2sqr(residuals.data(), n * dimension);
        std::cout << b_c_dist << std::endl;

        if (show_batch_time) batch_recorder.print_time_usage("compute residuals                 ");

        if (use_OPQ){
            do_OPQ(n, residuals.data());
        }

        //Compute code for residuals
        std::vector<uint8_t> batch_codes(n * this->code_size);
        this->pq.compute_codes(residuals.data(), batch_codes.data(), n);
        if (show_batch_time) batch_recorder.print_time_usage("PQ encode data residuals             ");

        //Add codes into index
        for (size_t i = 0; i < n; i++){
            idx_t group_id = group_ids[i];
            size_t group_position = group_positions[i];
            for (size_t j = 0; j < this->code_size; j++){this->base_codes[group_id][group_position * code_size + j] = batch_codes[i * this->code_size + j];}
            this->base_sequence_ids[group_id][group_position] = sequence_ids[i];

            if (use_vector_alpha && !alpha_flag){
                this->base_alphas[group_id][group_position] = vector_alpha[i];
                this->base_alpha_norms[group_id][group_position] = vector_alpha_norm[i];
            }
        }
        if (show_batch_time) batch_recorder.print_time_usage("add codes to index                ");

        if (!base_norm_flag){
            std::vector<float> decoded_residuals(n * dimension);
            this->pq.decode(batch_codes.data(), decoded_residuals.data(), n);

            if (use_OPQ){
                reverse_OPQ(n, decoded_residuals.data());
            }

            std::vector<float> reconstructed_x(n * dimension);
            decode(n, decoded_residuals.data(), group_ids, reconstructed_x.data(), vector_alpha);
            if (show_batch_time) batch_recorder.print_time_usage("compute reconstructed base vectors ");
            //This is the norm for reconstructed vectors
            for (size_t i = 0; i < n; i++){base_norms[sequence_ids[i]] =  faiss::fvec_norm_L2sqr(reconstructed_x.data() + i * dimension, dimension);}
            if (show_batch_time) batch_recorder.print_time_usage("add base norms                     ");
        }

        //The size of base_norm_code or base_norm should be initialized in main function
        /*
        if (use_norm_quantization){
            std::vector<uint8_t> xnorm_codes (n * norm_code_size);
            this->norm_pq.compute_codes(vector_norms, xnorm_codes.data(), n);
            for (size_t i = 0 ; i < n; i++){
                idx_t sequence_id = sequence_ids[i];
                for (size_t j =0; j < this->norm_code_size; j++){
                    this->base_norm_codes[sequence_id * norm_code_size + j] = xnorm_codes[i * this->norm_code_size +j];
                }
            }
        }
        */

        
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
    void Bslib_Index::get_final_centroid(size_t label, float * final_centroid){
        if (index_type[layers - 1] == "VQ"){
            size_t n_vq = vq_quantizer_index.size();
            vq_quantizer_index[n_vq - 1].compute_final_centroid(label, final_centroid);
        }
        else if(index_type[layers -1] == "LQ"){
            size_t n_lq = lq_quantizer_index.size();
            lq_quantizer_index[n_lq - 1].compute_final_centroid(label, final_centroid);
        }
        else{
            size_t n_pq = pq_quantizer_index.size();
            pq_quantizer_index[n_pq - 1].compute_final_centroid(label, final_centroid);
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
        if (this->use_vector_alpha){
            std::cout << "No centroid norms for LQ 2 type " << std::endl;
            return;
        }
        this->centroid_norms.resize(final_group_num);
        
        if (exists(path_centroid_norm)){
            std::ifstream centroid_norm_input (path_centroid_norm, std::ios::binary);
            readXvec<float> (centroid_norm_input, this->centroid_norms.data(), final_group_num, 1, false, false);
            centroid_norm_input.close();
        }

        else{
            if (this->index_type[layers -1] == "VQ"){
                size_t n_vq = vq_quantizer_index.size();
                std::cout << "Computing centroid norm for " << final_group_num << " centroids in VQ layer" << std::endl;
#pragma omp parallel for
                for (size_t label = 0; label < final_group_num; label++){
                    std::vector<float> each_centroid(dimension);
                    vq_quantizer_index[n_vq-1].compute_final_centroid(label, each_centroid.data());
                    this->centroid_norms[label] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
                }
            }
            else if (this->index_type[layers -1] == "LQ"){
                size_t n_lq = lq_quantizer_index.size();
                std::cout << "Computing centroid norm for " << final_group_num << " centroids in LQ layer" << std::endl;

#pragma omp parallel for
                for (size_t label = 0; label < final_group_num; label++){
                    std::vector<float> each_centroid(dimension);
                    lq_quantizer_index[n_lq-1].compute_final_centroid(label, each_centroid.data());
                    this->centroid_norms[label] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
                }
            }
            else if (this->index_type[layers - 1] == "PQ"){
                size_t n_pq = pq_quantizer_index.size();
                std::cout << "Computing centroid norm for " << final_group_num << " centroids in PQ layer" << std::endl;

 #pragma omp parallel for
                for (size_t label = 0; label < final_group_num; label++){
                    std::vector<float> each_centroid(dimension);
                    pq_quantizer_index[n_pq - 1].compute_final_centroid(label, each_centroid.data());
                    this->centroid_norms[label] = faiss::fvec_norm_L2sqr(each_centroid.data(), dimension);
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

    void Bslib_Index::assign(const size_t n, const float * assign_data, idx_t * assigned_ids, size_t assign_layer, float * alphas){

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
            std::vector<float> query_search_alpha(1, 0);
            std::vector<idx_t> query_result_ids;
            std::vector<float> query_result_dists;

            for (size_t j = 0; j < assign_layer; j++){
                assert(n_vq+ n_lq + n_pq == j);

                if (index_type[j] == "VQ"){
                    idx_t group_id = query_search_id[0];
                    size_t group_size = vq_quantizer_index[n_vq].exact_nc_in_groups[group_id];
                    query_result_dists.resize(group_size, 0);
                    query_result_ids.resize(group_size, 0);
                    
                    const float * query_data = assign_data + i * dimension;
                    vq_quantizer_index[n_vq].search_in_group(query_data, group_id, query_result_dists.data(), query_result_ids.data(), 1);
                    if (vq_quantizer_index[n_vq].use_HNSW){
                        query_search_id[j] = query_result_ids[0];
                    }
                    else{
                        std::vector<float> sub_dist(1);
                        keep_k_min(group_size, 1, query_result_dists.data(), query_result_ids.data(), sub_dist.data(), query_search_id.data()); 
                    }
                    n_vq ++;
                }

                else if(index_type[j] == "LQ") {

                    idx_t group_id = query_search_id[0];
                    size_t group_size = lq_quantizer_index[n_lq].max_nc_per_group;
                    // Copy the upper search result for LQ layer 
                    size_t upper_group_size = query_result_ids.size();
                    
                    std::vector<idx_t> upper_result_ids(upper_group_size, 0);
                    std::vector<float> upper_result_dists(upper_group_size, 0);
                    memcpy(upper_result_ids.data(), query_result_ids.data(), upper_group_size * sizeof(idx_t));
                    memcpy(upper_result_dists.data(), query_result_dists.data(), upper_group_size * sizeof(float));
                    query_result_ids.resize(group_size, 0);
                    query_result_dists.resize(group_size, 0);

                    const float * target_data = assign_data + i * dimension;
                    std::vector<float> query_result_alphas; if(use_vector_alpha){query_result_alphas.resize(group_size);}
                    
                    lq_quantizer_index[n_lq].search_in_group(target_data, upper_result_ids.data(), upper_result_dists.data(), upper_group_size, group_id, query_result_dists.data(), query_result_ids.data(), query_result_alphas.data());
                    std::vector<float> sub_dist(1);
                    if (use_vector_alpha){
                        keep_k_min_alpha(group_size, 1, query_result_dists.data(),query_result_ids.data(), query_result_alphas.data(), sub_dist.data(), query_search_id.data(), query_search_alpha.data());
                    }
                    else{
                        keep_k_min(group_size, 1, query_result_dists.data(), query_result_ids.data(), sub_dist.data(), query_search_id.data());
                    }
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
            if (use_vector_alpha && assign_layer == layers){
                alphas[i] = query_search_alpha[0];
            }
        }
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
      * With OPQ:
      * Since OPQ is a Orthogonal Matrix, it does not reflect the norm value
      * Only the query * residual_PQ will be reflected
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
      **/

    void Bslib_Index::search(size_t n, size_t result_k, float * queries, float * query_dists, idx_t * query_ids, const size_t * keep_space, uint32_t * groundtruth, std::string path_base){
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

            float * query = queries + i * dimension;
            size_t n_vq = 0;
            size_t n_lq = 0;
            size_t n_pq = 0;

            size_t upper_result_space = 1;
            size_t search_space = 1;
            size_t max_group_size = 1;

            // The final keep space is the number of groups that we kept for searching at last, the product of all keep space
            size_t final_keep_space = 1;
            // The max search space we meet: for one query in on layer, it is the keep space upper layer * ncentroids this layer
            size_t max_search_space = 1;

            for (size_t j = 0; j < layers; j++){
                // The ncentroids of PQ layer will be 0, only the keep_space will be considered
                if (final_keep_space * ncentroids[j] > max_search_space || final_keep_space * keep_space[j] > max_search_space){
                    max_search_space = final_keep_space * ((ncentroids[j] > keep_space[j]) ? ncentroids[j] : keep_space[j]);
                }
                final_keep_space *= keep_space[j];
            }

            assert(max_search_space > 0 && final_keep_space > 0);
            std::vector<float> query_result_dists(1 * max_search_space, 0);
            std::vector<idx_t> query_result_labels(1 * max_search_space, 0);

            
;
            std::vector<idx_t> query_group_ids(1 * final_keep_space, 0);
            std::vector<float> query_group_dists(1 * final_keep_space, 0);

            std::vector<float> vector_group_alphas;
            std::vector<float> vector_result_alphas; 
            if (use_vector_alpha){
                vector_group_alphas.resize(final_keep_space);
                vector_result_alphas.resize(max_search_space);}

//#pragma omp critical
            for (size_t j = 0; j < layers; j++){
                assert(n_vq+ n_lq + n_pq== j);
                
                if (index_type[j] == "VQ"){
                    if (showmessage) PrintMessage("Searching in VQ Layer");
                    max_group_size = vq_quantizer_index[n_vq].max_nc_per_group;
                    std::vector<size_t> layer_keep_spaces(upper_result_space);

//#pragma omp parallel for
                    for (size_t m = 0; m < upper_result_space; m++){
                        assert(query_group_ids[m] >=0);
                        size_t layer_group_size = vq_quantizer_index[n_vq].exact_nc_in_groups[query_group_ids[m]];
                        size_t layer_keep_space = keep_space[j] < layer_group_size ? keep_space[j] : layer_group_size;
                        layer_keep_spaces[m] = layer_keep_space;
                        vq_quantizer_index[n_vq].search_in_group(query, query_group_ids[m], query_result_dists.data()+m* max_group_size, query_result_labels.data() + m*max_group_size, layer_keep_space);
                    }
                    size_t sum_result_space = 0;
                    if (vq_quantizer_index[n_vq].use_HNSW){
                        for (size_t m = 0; m < upper_result_space; m++){
                            for (size_t k = 0; k < layer_keep_spaces[m]; k++){
                                query_group_ids[sum_result_space + k] = query_result_labels[m * max_group_size + k];
                                query_group_dists[sum_result_space + k] = query_result_dists[m * max_group_size + k];
                            }
                            sum_result_space += layer_keep_spaces[m];
                        }
                    }
                    else{
                        for (size_t m = 0; m < upper_result_space; m++){
                            keep_k_min(max_group_size, layer_keep_spaces[m], query_result_dists.data()+m*max_group_size, query_result_labels.data()+m*max_group_size, query_group_dists.data()+sum_result_space, query_group_ids.data()+sum_result_space); 
                            sum_result_space += layer_keep_spaces[m];
                        }
                    }
                    search_space = upper_result_space * max_group_size;
                    upper_result_space = sum_result_space;
                    n_vq ++;
                }
                else if(index_type[j] == "LQ") {
                    if (showmessage) PrintMessage("Searching in LQ layer");
                    max_group_size = lq_quantizer_index[n_lq].max_nc_per_group;
                    // Copy the upper search result for LQ layer 
                    std::vector<idx_t> upper_result_labels(search_space);
                    std::vector<float> upper_result_dists(search_space);
                    memcpy(upper_result_labels.data(), query_result_labels.data(), search_space * sizeof(idx_t));
                    memcpy(upper_result_dists.data(), query_result_dists.data(), search_space * sizeof(float));

                    //for (size_t m = 0; m < search_space; m++){upper_result_labels[m] = result_labels[m]; upper_result_dists[m] = result_dists[m];}
//#pragma omp parallel for

                    size_t layer_keep_space = keep_space[j] < max_group_size ? keep_space[j] : max_group_size;
                    for (size_t m = 0; m < upper_result_space; m++){
                        assert(query_group_ids[m] >=0);
                        lq_quantizer_index[n_lq].search_in_group(query, upper_result_labels.data(), upper_result_dists.data(), search_space, query_group_ids[m], query_result_dists.data()+m*max_group_size, 
                        query_result_labels.data()+m*max_group_size, vector_result_alphas.data() + m * max_group_size);
                    }
                    for (size_t m = 0; m < upper_result_space; m++){
                        if (use_vector_alpha && lq_quantizer_index[n_lq].LQ_type == 2){
                            keep_k_min_alpha(max_group_size, layer_keep_space, query_result_dists.data()+m*max_group_size, query_result_labels.data()+m*max_group_size, vector_result_alphas.data() + m * max_group_size,
                             query_group_dists.data()+m*layer_keep_space, query_group_ids.data()+m*layer_keep_space, vector_group_alphas.data() + m * layer_keep_space);
                        }
                        else{
                            keep_k_min(max_group_size, layer_keep_space, query_result_dists.data()+m*max_group_size, query_result_labels.data()+m*max_group_size, query_group_dists.data()+m*keep_space[j], query_group_ids.data()+m*keep_space[j]);
                        }
                    }
                    search_space = upper_result_space * max_group_size;
                    upper_result_space = upper_result_space * layer_keep_space;
                    n_lq ++;
                }

                else if(index_type[j] == "PQ"){
                    if (showmessage) PrintMessage("Searching in PQ layer");
                    assert(j == this->layers-1);

                    size_t sum_result_space = 0;
//#pragma omp parallel for
                    for (size_t m = 0; m < upper_result_space; m++){
                        assert(query_group_ids[m] >=0);
                        size_t layer_ksub = pq_quantizer_index[n_pq].exact_ksubs[query_group_ids[m]];
                        size_t layer_keep_space = keep_space[j] < layer_ksub ? keep_space[j] : layer_ksub;
                        pq_quantizer_index[n_pq].search_in_group(query, query_group_ids[m], query_result_dists.data()+sum_result_space, query_result_labels.data()+sum_result_space, layer_keep_space);
                        sum_result_space += layer_keep_space;
                    }

                    memcpy(query_group_ids.data(), query_result_labels.data(), sum_result_space * sizeof(idx_t));
                    memcpy(query_group_dists.data(), query_result_dists.data(), sum_result_space * sizeof(float));
                    upper_result_space = sum_result_space;
                    search_space = sum_result_space;
                    n_pq ++;
                }
                else{
                    std::cout << "The type name is wrong with " << index_type[j] << "!" << std::endl;
                    exit(0);
                }
                if (analysis){time_consumption[i][j] = Trecorder.getTimeConsumption(); Trecorder.reset();}
            }
            final_keep_space = upper_result_space;
            if (showmessage) std::cout << "Finished search in index centroids, show the results" << std::endl;
            assert((n_vq + n_lq + n_pq) == this->layers);
            
            std::vector<float> precomputed_table(pq.M * pq.ksub);
            if (use_OPQ) {do_OPQ(1,query);}
            this->pq.compute_inner_prod_table(query, precomputed_table.data());

            //The analysis variables
            size_t visited_vector_size = 0;
            size_t valid_result_length = 0;
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

            if (showmessage) std::cout << "Searching the base vectors " << std::endl;
            n_lq = lq_quantizer_index.size() - 1;
            size_t j = 0;
            for (j = 0; j < final_keep_space; j++){

                std::pair<idx_t, float> result_idx_dist;
                size_t list_id = get_next_group_idx(final_keep_space, query_group_ids.data(), query_group_dists.data(), result_idx_dist);
                
                idx_t all_group_id = result_idx_dist.first;
                float q_c_dist = result_idx_dist.second;
                float query_alpha = use_vector_alpha ? vector_group_alphas[list_id] : 0;

                idx_t lq_group_id, lq_inner_group_id;
                if (use_vector_alpha){lq_quantizer_index[n_lq].get_group_id(all_group_id, lq_group_id, lq_inner_group_id);}

                size_t group_size = this->base_sequence_ids[all_group_id].size();
                assert(group_size == this->base_codes[all_group_id].size() / this->code_size);


                if (showmessage) std::cout << "Searching in " << all_group_id << " th group with distance " << q_c_dist << "and size: " << group_size << std::endl;

                if (use_group_HNSW && group_size >= group_HNSW_thres){
                    assert(group_HNSW_thres > 0);
                    size_t group_HNSW_id = group_HNSW_idxs.at(all_group_id);
                    this->group_HNSW_list[group_HNSW_id]->q_c_dist = q_c_dist;
                    auto result_queue = this->group_HNSW_list[group_HNSW_id]->searchKnn(precomputed_table.data(), result_k);
                    for(size_t m = 0; m < result_k; m++){
                        query_search_dists[valid_result_length] = result_queue.top().first;
                        query_search_labels[valid_result_length] = base_sequence_ids[all_group_id][result_queue.top().second];
                        result_queue.pop();
                        valid_result_length ++;
                    }
                    visited_vector_size += group_size;
                }
                else{
                    float centroid_norm = 0;
                    if(!use_vector_alpha){
                        centroid_norm = centroid_norms[all_group_id];
                        assert(centroid_norm > 0);
                    }
                    // Validating the computation of q_c_dist and centroid_norm
                    /*if (validation){
                        float actual_centroid_norm, actual_q_c_dist;
                        std::vector<float> centroid(dimension);
                        std::vector<float> q_c_dis_vector(dimension);
                        get_final_centroid(group_id, centroid.data());
                        actual_centroid_norm = faiss::fvec_norm_L2sqr(centroid.data(), dimension);
                        faiss::fvec_madd(dimension, query, -1.0, centroid.data(), q_c_dis_vector.data());
                        actual_q_c_dist = faiss::fvec_norm_L2sqr(q_c_dis_vector.data(), dimension);
                        assert(abs(q_c_dist - actual_q_c_dist) < VALIDATION_EPSILON && abs(centroid_norm - actual_centroid_norm) < VALIDATION_EPSILON);
                    }*/

                    
                    const uint8_t * code = base_codes[all_group_id].data();
                    for (size_t m = 0; m < group_size; m++){

                        idx_t sequence_id = base_sequence_ids[all_group_id][m];
                        std::vector<float> base_reconstructed_norm(1);
                        if (use_norm_quantization) norm_pq.decode(base_norm_codes.data() + sequence_id * norm_code_size, base_reconstructed_norm.data(), 1);
                        float base_norm = use_norm_quantization ? base_reconstructed_norm[0] : base_norms[sequence_id];
                        float PQ_table_product = 2 * pq_L2sqr(code + m * code_size, precomputed_table.data(), pq.code_size, pq.ksub);

                        if(!use_vector_alpha){
                            query_search_dists[valid_result_length] =  q_c_dist - centroid_norm + base_norm - PQ_table_product;
                        }
                        else{
                            
                            float L2_C1_C2 = (query_alpha - base_alphas[all_group_id][m]) * (query_alpha - base_alphas[all_group_id][m]) * (lq_quantizer_index[n_lq].nn_centroid_dists[lq_group_id][lq_inner_group_id]);
                            query_search_dists[valid_result_length] = q_c_dist - base_alpha_norms[all_group_id][m] + base_norm + L2_C1_C2 - PQ_table_product;
                        }


                        query_search_labels[valid_result_length] = base_sequence_ids[all_group_id][m];
                        visited_vector_size ++;
                        valid_result_length ++;
                        //Compute the actual distance
                        /*
                        if (validation){
                            std::vector<base_data_type> base_vector(dimension); uint32_t dim;
                            base_input.seekg(base_sequence_ids[all_group_id][m] * dimension * sizeof(base_data_type) + base_sequence_ids[group_id][m] * sizeof(uint32_t), std::ios::beg);
                            base_input.read((char *) & dim, sizeof(uint32_t)); assert(dim == this->dimension);
                            base_input.read((char *) base_vector.data(), sizeof(base_data_type)*dimension);
                            std::vector<float> base_vector_float(dimension);
                            for (size_t temp = 0; temp < dimension; temp++){base_vector_float[temp] = base_vector[temp];}
                            float actual_dist =  faiss::fvec_L2sqr(base_vector_float.data(), query, dimension);
                            std::vector<float> decoded_code(dimension);
                            pq.decode(code + m * code_size, decoded_code.data(), 1);
                            std::vector<float> decoded_base_vector(dimension);
                            //decode(1, decoded_code.data(), & group_id, decoded_base_vector.data());
                            float actual_norm = faiss::fvec_norm_L2sqr(decoded_base_vector.data(), dimension);
                            
                            query_actual_dists[valid_result_length] = actual_dist;
                            float product_term3 = 2 * faiss::fvec_inner_product(query, decoded_code.data(), dimension);
                            
                            std::vector<float> b_c_residual(dimension);
                            std::vector<float> centroid(dimension);
                            get_final_centroid(group_id, centroid.data());
                            faiss::fvec_madd(dimension, base_vector_float.data(), -1.0, centroid.data(), b_c_residual.data());
                            float actual_term3 = 2 * faiss::fvec_inner_product(query, b_c_residual.data(), dimension);
                            std::vector<uint8_t> base_code(this->code_size);
                            pq.compute_code(b_c_residual.data(), base_code.data());
                            std::vector<float> reconstructed_residual(dimension);
                            pq.decode(base_code.data(), reconstructed_residual.data(), 1);
                            float test_term3 = 2 * faiss::fvec_inner_product(query, reconstructed_residual.data(), dimension);
                            
                            std::cout << " S: " << query_search_dists[valid_result_length] << " A: " << actual_dist << " LN: " << base_norm << " AN: " << actual_norm << " " << " Term1: " << q_c_dist << " " << centroid_norm << " Term3: " << PQ_table_product << " " << "Term3 inner: " << product_term3 <<  " Term3 Test: " << test_term3 << " Actual term3: " << actual_term3 << std::endl; // S for "Search" and A for "Actual"
                        }
                        */

                        if (analysis){if (grountruth_set.count(base_sequence_ids[all_group_id][m]) != 0){visited_gt ++;}}
                        }
                    }
                    if (visited_vector_size >= this->max_visited_vectors)
                        break;
                }

                if (validation){
                //Compute the distance sort for computed distance
                    std::cout << std::endl;
                    std::vector<idx_t> search_dist_index(valid_result_length);
                    uint32_t x=0;
                    std::iota(search_dist_index.begin(),search_dist_index.end(),x++);
                    std::sort(search_dist_index.begin(),search_dist_index.end(), [&](int i,int j){return query_search_dists[i]<query_search_dists[j];} );

                    //Compute the distance sort for actual distance
                    std::vector<idx_t> actual_dist_index(valid_result_length);
                    x = 0;
                    std::iota(actual_dist_index.begin(), actual_dist_index.end(), x++);
                    std::sort(actual_dist_index.begin(),actual_dist_index.end(), [&](int i,int j){return query_actual_dists[i]<query_actual_dists[j];} );

                    assert(valid_result_length > validation_print_space);
                    std::cout << "Search Labels     Search Dists     Actual Labels     Actual Dists" << std::endl;
                    for (size_t temp = 0; temp < validation_print_space; temp++){
                        std::cout << query_search_labels[search_dist_index[temp]] << "        " << 
                        query_search_dists[search_dist_index[temp]] << "        " << 
                        query_search_labels[actual_dist_index[temp]] << "        " <<
                        query_actual_dists[actual_dist_index[temp]] << std::endl;
                    }
                }
                if (use_reranking){
                    assert(valid_result_length > reranking_space);
                    std::ifstream reranking_input(path_base, std::ios::binary);
                    std::vector<float> reranking_dists(reranking_space, 0);
                    std::vector<idx_t> reranking_labels(reranking_space, 0);
                    std::vector<float> reranking_actual_dists(reranking_space, 0);
                    keep_k_min(valid_result_length, reranking_space, query_search_dists.data(), query_search_labels.data(), reranking_dists.data(), reranking_labels.data());
                    
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

                else{keep_k_min(valid_result_length, result_k, query_search_dists.data(), query_search_labels.data(), query_dists + i * result_k, query_ids + i * result_k);}

                if (analysis){
                    size_t correct = 0; for (size_t temp=0;temp<result_k;temp++){if(grountruth_set.count(query_ids[ i * result_k + temp])!=0) correct++;}
                    recall[i] = float(correct) / result_k;
                    time_consumption[i][layers]  = Trecorder.getTimeConsumption();
                    visited_gt_proportion[i] = float(visited_gt) / result_k;
                    actual_visited_vectors[i] = valid_result_length;
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
        quantizers_output.write((char *) this->ncentroids.data(), this->layers * sizeof(size_t));

        for (size_t i = 0; i < this->layers; i++){
            if (index_type[i] == "VQ"){
                PrintMessage("Writing VQ quantizer layer");
                const size_t layer_nc = vq_quantizer_index[n_vq].layer_nc;
                const size_t nc_upper = vq_quantizer_index[n_vq].nc_upper;
                const size_t max_nc_per_group = vq_quantizer_index[n_vq].max_nc_per_group;

                quantizers_output.write((char *) & layer_nc, sizeof(size_t));
                quantizers_output.write((char *) & nc_upper, sizeof(size_t));
                quantizers_output.write((char *) & max_nc_per_group, sizeof(size_t));
                quantizers_output.write((char *) vq_quantizer_index[n_vq].exact_nc_in_groups.data(), nc_upper * sizeof(size_t));
                quantizers_output.write((char *) vq_quantizer_index[n_vq].CentroidDistributionMap.data(), nc_upper * sizeof(idx_t));
                size_t HNSW_flag = vq_quantizer_index[n_vq].use_HNSW ? 1 : 0;
                quantizers_output.write((char *) & HNSW_flag, sizeof(size_t));


                std::cout << " nc in this layer: " << layer_nc << " nc in upper layer: " << nc_upper << std::endl;

                if (vq_quantizer_index[n_vq].use_HNSW){
                    std::cout << "Writing HNSW indexes " << std::endl;
                    size_t M = vq_quantizer_index[n_vq].M;
                    size_t efConstruction = vq_quantizer_index[n_vq].efConstruction;

                    quantizers_output.write((char * ) & M, sizeof(size_t));
                    quantizers_output.write((char * ) & efConstruction, sizeof(size_t));

                    vq_quantizer_index[n_vq].write_HNSW(quantizers_output);
                }
                else{
                    std::cout << "Writing L2 centroids " << std::endl;
                    for (size_t group_id = 0; group_id < nc_upper; group_id++){
                        size_t group_quantizer_data_size = vq_quantizer_index[n_vq].exact_nc_in_groups[group_id] * this->dimension;
                        assert(vq_quantizer_index[n_vq].L2_quantizers[group_id]->xb.size() == group_quantizer_data_size);
                        quantizers_output.write((char * ) vq_quantizer_index[n_vq].L2_quantizers[group_id]->xb.data(), group_quantizer_data_size * sizeof(float));
                    }
                }
                assert(n_vq + n_lq + n_pq == i);
                n_vq ++;
            }
            else if (index_type[i] == "LQ"){
                PrintMessage("Writing LQ quantizer layer");
                const size_t nc_upper = lq_quantizer_index[n_lq].nc_upper;
                const size_t max_nc_per_group = lq_quantizer_index[n_lq].max_nc_per_group;
                quantizers_output.write((char *) & lq_quantizer_index[n_lq].layer_nc, sizeof(size_t));
                quantizers_output.write((char *) & lq_quantizer_index[n_lq].nc_upper, sizeof(size_t));
                quantizers_output.write((char *) & lq_quantizer_index[n_lq].max_nc_per_group, sizeof(size_t));
                quantizers_output.write((char *) & lq_quantizer_index[n_lq].LQ_type, sizeof(size_t));

                assert(lq_quantizer_index[n_lq].upper_centroids.size() == nc_upper * dimension);
                quantizers_output.write((char *) lq_quantizer_index[n_lq].upper_centroids.data(), nc_upper * this->dimension*sizeof(float));

                for (size_t j = 0; j < nc_upper; j++){
                    assert(lq_quantizer_index[n_lq].nn_centroid_ids[j].size() == max_nc_per_group);
                    quantizers_output.write((char *)lq_quantizer_index[n_lq].nn_centroid_ids[j].data(), max_nc_per_group * sizeof(idx_t));
                }
                for (size_t j = 0; j < nc_upper; j++){
                    assert(lq_quantizer_index[n_lq].nn_centroid_dists[j].size() == max_nc_per_group);
                    quantizers_output.write((char * )lq_quantizer_index[n_lq].nn_centroid_dists[j].data(), max_nc_per_group * sizeof(float));
                }

                if (lq_quantizer_index[n_lq].LQ_type != 2){
                    assert(lq_quantizer_index[n_lq].alphas.size() == nc_upper);
                    for (size_t group_id = 0; group_id < nc_upper; group_id++){
                        quantizers_output.write((char *) lq_quantizer_index[n_lq].alphas[group_id].data(), max_nc_per_group * sizeof(float));
                    }
                }

                assert(n_vq + n_lq + n_pq == i);
                n_lq ++;
            }
            else if (index_type[i] == "PQ"){
                PrintMessage("Writing PQ quantizer layer");
                const size_t layer_nc = pq_quantizer_index[n_pq].layer_nc;
                const size_t nc_upper = pq_quantizer_index[n_pq].nc_upper;
    
                const size_t M = pq_quantizer_index[n_pq].M;
                const size_t max_nbits = pq_quantizer_index[n_pq].max_nbits;
                const size_t max_ksub = pq_quantizer_index[n_pq].max_ksub;

                quantizers_output.write((char *) & layer_nc, sizeof(size_t));
                quantizers_output.write((char * ) & nc_upper, sizeof(size_t));
                quantizers_output.write((char * ) & M, sizeof(size_t));
                quantizers_output.write((char * ) & max_nbits, sizeof(size_t));
                quantizers_output.write((char * ) & max_ksub, sizeof(size_t));

                quantizers_output.write((char * ) pq_quantizer_index[n_pq].exact_nbits.data(), nc_upper * sizeof(size_t));
                quantizers_output.write((char * ) pq_quantizer_index[n_pq].exact_ksubs.data(), nc_upper * sizeof(size_t));
                quantizers_output.write((char * ) pq_quantizer_index[n_pq].CentroidDistributionMap.data(), nc_upper * sizeof(idx_t));

                assert(pq_quantizer_index[n_pq].PQs.size() == nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    size_t centroid_size = pq_quantizer_index[n_pq].PQs[j]->centroids.size();
                    assert(centroid_size == dimension * pq_quantizer_index[n_pq].PQs[j]->ksub);
                    quantizers_output.write((char * )pq_quantizer_index[n_pq].PQs[j]->centroids.data(), centroid_size * sizeof(float));
                }

                for (size_t j = 0; j < nc_upper; j++){
                    quantizers_output.write((char *) pq_quantizer_index[n_pq].centroid_norms[j].data(), M* pq_quantizer_index[n_pq].exact_ksubs[j] *sizeof(float));
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
        size_t layer_nc;
        size_t nc_upper;
        size_t max_nc_per_group;
        quantizer_input.read((char *) & this->max_group_size, sizeof(size_t));
        quantizer_input.read((char *) this->ncentroids.data(), this->layers * sizeof(size_t));

        for(size_t i = 0; i < this->layers; i++){
            if (index_type[i] == "VQ"){
                std::cout << "Reading VQ layer" << std::endl;
                quantizer_input.read((char *) & layer_nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & max_nc_per_group, sizeof(size_t));
                std::vector<size_t> exact_nc_in_group(nc_upper);
                std::vector<size_t> CentroidDistributionMap(nc_upper);
                quantizer_input.read((char *) exact_nc_in_group.data(), nc_upper * sizeof(size_t));
                quantizer_input.read((char *)CentroidDistributionMap.data(), nc_upper * sizeof(idx_t));
                size_t VQ_HNSW_flag;
                quantizer_input.read((char *) & VQ_HNSW_flag, sizeof(size_t));

                std::cout << " nc in this layer: " << layer_nc << " nc in upper layer: " << nc_upper << std::endl;

                assert(max_nc_per_group * nc_upper >= layer_nc);

                if (VQ_HNSW_flag == 1){
                    size_t M;
                    size_t efConstruction;
                    quantizer_input.read((char *) & M, sizeof(size_t));
                    quantizer_input.read((char *) & efConstruction, sizeof(size_t));
                    
                    VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, max_nc_per_group, M, efConstruction, true);
                    vq_quantizer.layer_nc = layer_nc;
                    for (size_t j = 0; j < nc_upper; j++){
                        vq_quantizer.exact_nc_in_groups[j] = exact_nc_in_group[j];
                        vq_quantizer.CentroidDistributionMap[j] = CentroidDistributionMap[j];
                    }

                    vq_quantizer.read_HNSW(quantizer_input);
                    this->vq_quantizer_index.push_back(vq_quantizer);
                }
                else{
                    VQ_quantizer vq_quantizer = VQ_quantizer(dimension, nc_upper, max_nc_per_group);
                    vq_quantizer.layer_nc = layer_nc;
                    for (size_t j = 0; j < nc_upper; j++){
                        vq_quantizer.exact_nc_in_groups[j] = exact_nc_in_group[j];
                        vq_quantizer.CentroidDistributionMap[j] = CentroidDistributionMap[j];
                    }
                    
                    for (size_t j = 0; j < nc_upper; j++){
                        std::vector<float> centroids(exact_nc_in_group[j] * this->dimension);
                        quantizer_input.read((char *) centroids.data(), exact_nc_in_group[j] * dimension * sizeof(float));
                        faiss::IndexFlatL2 * centroid_quantizer = new faiss::IndexFlatL2(dimension);
                        centroid_quantizer->add(exact_nc_in_group[j], centroids.data());
                        vq_quantizer.L2_quantizers[j] = centroid_quantizer;
                    }
                    this->vq_quantizer_index.push_back(vq_quantizer);
                }
            }

            else if (index_type[i] == "LQ"){
                std::cout << "Reading LQ layer " << std::endl;
                size_t LQ_type;
                quantizer_input.read((char *) & layer_nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & max_nc_per_group, sizeof(size_t));
                quantizer_input.read((char *) & LQ_type, sizeof(size_t));

                assert(max_nc_per_group * nc_upper == layer_nc);
                std::cout << layer_nc << " " << nc_upper << " " << max_nc_per_group << " " << std::endl;
                std::vector<float> alphas(max_nc_per_group);
                std::vector<float> upper_centroids(nc_upper * dimension);
                std::vector<idx_t> nn_centroid_ids(nc_upper * max_nc_per_group);
                std::vector<float> nn_centroid_dists(nc_upper * max_nc_per_group);

                quantizer_input.read((char *) upper_centroids.data(), nc_upper * this->dimension * sizeof(float));
                quantizer_input.read((char *) nn_centroid_ids.data(), nc_upper * max_nc_per_group * sizeof(idx_t));
                quantizer_input.read((char *) nn_centroid_dists.data(), nc_upper * max_nc_per_group * sizeof(float));

                LQ_quantizer lq_quantizer = LQ_quantizer(dimension, nc_upper, max_nc_per_group, upper_centroids.data(), nn_centroid_ids.data(), nn_centroid_dists.data(), LQ_type);
                

                if (LQ_type != 2){
                    for (size_t group_id = 0; group_id < nc_upper; group_id++){
                        quantizer_input.read((char *) alphas.data(), max_nc_per_group * sizeof(float));
                        lq_quantizer.alphas[group_id].resize(max_nc_per_group);
                        for (size_t inner_group_id = 0; inner_group_id < max_nc_per_group; inner_group_id++){
                            lq_quantizer.alphas[group_id][inner_group_id] = alphas[inner_group_id];
                        }
                    }
                }
                
                this->lq_quantizer_index.push_back(lq_quantizer);
            }

            else if (index_type[i] == "PQ"){
                std::cout << "Reading PQ layer " << std::endl;
                size_t layer_nc;
                size_t M;
                size_t max_nbits;
                size_t max_ksub;

                quantizer_input.read((char *) & layer_nc, sizeof(size_t));
                quantizer_input.read((char *) & nc_upper, sizeof(size_t));
                quantizer_input.read((char *) & M, sizeof(size_t));
                quantizer_input.read((char *) & max_nbits, sizeof(size_t));
                quantizer_input.read((char *) & max_ksub, sizeof(size_t));
                
                
                PQ_quantizer pq_quantizer = PQ_quantizer(dimension, nc_upper, M, max_nbits);
                pq_quantizer.layer_nc = layer_nc;
                quantizer_input.read((char *) pq_quantizer.exact_nbits.data(), nc_upper * sizeof(size_t));
                quantizer_input.read((char *) pq_quantizer.exact_ksubs.data(), nc_upper * sizeof(size_t));
                quantizer_input.read((char *) pq_quantizer.CentroidDistributionMap.data(), nc_upper * sizeof(size_t));

                pq_quantizer.PQs.resize(nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    faiss::ProductQuantizer * product_quantizer = new faiss::ProductQuantizer(dimension, M, nbits);
                    size_t centroid_size = dimension * product_quantizer->ksub;
                    quantizer_input.read((char *) product_quantizer->centroids.data(), centroid_size * sizeof(float));
                    pq_quantizer.PQs[j] = product_quantizer;
                }

                pq_quantizer.centroid_norms.resize(nc_upper);
                for (size_t j = 0; j < nc_upper; j++){
                    pq_quantizer.centroid_norms[j].resize(M * pq_quantizer.exact_ksubs[j]);
                    quantizer_input.read((char *) pq_quantizer.centroid_norms[j].data(), M * pq_quantizer.exact_ksubs[j] * sizeof(float));
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

        if(!use_vector_alpha){
            assert(centroid_norms.size() == final_group_num);
            output.write((char *) & this->final_group_num, sizeof(size_t));
            assert(centroid_norms.size() == this->final_group_num);
            output.write((char *) centroid_norms.data(), this->final_group_num * sizeof(float));
        }
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

        if (!use_vector_alpha){
            input.read((char *) & final_nc_input, sizeof(size_t));
            assert(final_nc_input == this->final_group_num);
            this->centroid_norms.resize(this->final_group_num);
            input.read((char *) centroid_norms.data(), this->final_group_num * sizeof(float));
        }

        input.close();
    }

    /*
        Construct the index with the train set
        Input:
            path_learn:    str, the path to read the dataset
            

        Output: 
            None  (The index is updated)

    */
    void Bslib_Index::build_index(std::string path_learn, std::string path_groups, std::string path_labels,
    std::string path_quantizers, size_t VQ_layers, size_t PQ_layers, size_t LQ_layers,
    const uint32_t * ncentroids, const size_t * M_HNSW, const size_t * efConstruction, 
    const size_t * M_PQ_layer, const size_t * nbits_PQ_layer, const size_t * num_train,
    size_t selector_train_size, size_t selector_group_size, const size_t * LQ_type, std::ofstream & record_file){

        PrintMessage("Initializing the index");
        Trecorder.reset();

        if (use_train_selector){
            this->build_train_selector(path_learn, path_groups, path_labels, train_size, selector_train_size, selector_group_size);
        }

        std::vector<HNSW_PQ_para> HNSW_paras;
        for (size_t i = 0; i < VQ_layers; i++){
            HNSW_PQ_para new_para; new_para.first = M_HNSW[i]; new_para.second = efConstruction[i];
            HNSW_paras.push_back(new_para);
        }

        std::vector<HNSW_PQ_para> PQ_paras;
        for (size_t i = 0; i < PQ_layers; i++){
            HNSW_PQ_para new_para; new_para.first = M_PQ_layer[i]; new_para.second = nbits_PQ_layer[i];
            PQ_paras.push_back(new_para);
        }

        if (index_type[layers-1] == "LQ" && LQ_type[LQ_layers -1] == 2){
            use_vector_alpha = true;
        }

        PrintMessage("Constructing the quantizers");
        this->build_quantizers(ncentroids, path_quantizers, path_learn, num_train, HNSW_paras, PQ_paras, LQ_type, record_file);
        this->get_final_group_num();

        if (use_recording){
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
    void Bslib_Index::assign_vectors(std::string path_ids, std::string path_base, std::string path_alphas_raw,
        uint32_t batch_size, size_t nbatches,  std::ofstream & record_file){

        PrintMessage("Assigning the points");
        Trecorder.reset();
        if (!exists(path_ids)){
            std::ifstream base_input (path_base, std::ios::binary);
            std::ofstream base_output (path_ids, std::ios::binary);
            std::ofstream alphas_output;

            std::vector <float> batch(batch_size * dimension);
            std::vector<idx_t> assigned_ids(batch_size);
            std::vector<float> vector_alphas;
            if (use_vector_alpha){
                vector_alphas.resize(batch_size); 
                alphas_output = std::ofstream(path_alphas_raw, std::ios::binary);
            }

            for (size_t i = 0; i < nbatches; i++){
                readXvecFvec<base_data_type> (base_input, batch.data(), dimension, batch_size, false, false);
                this->assign(batch_size, batch.data(), assigned_ids.data(), this->layers, vector_alphas.data());
                base_output.write((char * ) & batch_size, sizeof(uint32_t));
                base_output.write((char *) assigned_ids.data(), batch_size * sizeof(idx_t));
                alphas_output.write((char * ) & batch_size, sizeof(uint32_t));
                alphas_output.write((char * ) vector_alphas.data(), batch_size * sizeof(float));

                if (i % 10 == 0){
                    std::cout << " assigned batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.print_time_usage("");
                    record_file << " assigned batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.record_time_usage(record_file, " ");
                }
            }
            base_input.close();
            base_output.close();
            if (use_recording){
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
    void Bslib_Index::train_pq_quantizer(const std::string path_pq, const std::string path_pq_norm,
        const size_t M_pq, const std::string path_learn, const std::string path_OPQ, const size_t PQ_train_size,  std::ofstream & record_file){
        
        PrintMessage("Constructing the PQ compressor");
        Trecorder.reset();
        if (exists(path_pq)){
            
            this->pq = * faiss::read_ProductQuantizer(path_pq.c_str());
            this->code_size = this->pq.code_size;
            std::cout << "Loading PQ codebook from " << path_pq << std::endl;

            if (use_OPQ){
                this->opq_matrix = * dynamic_cast<faiss::LinearTransform *>((faiss::read_VectorTransform(path_OPQ.c_str())));
                std::cout << "Loading OPQ matrix from " << path_OPQ << std::endl;
            }

            if(use_norm_quantization){
                std::cout << "Loading norm PQ codebook from " << path_pq_norm << std::endl;
                this->norm_pq = * faiss::read_ProductQuantizer(path_pq_norm.c_str());
                this->norm_code_size = this->norm_pq.code_size;
            }
        }
        else
        {
            this->M_pq = M_pq;

            std::cout << "Training PQ codebook" << std::endl;
            this->train_pq(path_pq, path_pq_norm, path_learn, path_OPQ, PQ_train_size);
        }
        std::cout << "Checking the PQ with its code size:" << this->pq.code_size << std::endl;

        if (use_recording){
            std::string message = "Trained the PQ, ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }
    
    void Bslib_Index::load_index(std::string path_index, std::string path_ids, std::string path_base,
        std::string path_base_norm, std::string path_centroid_norm,  std::string path_group_HNSW, std::string path_alphas_raw,
        std::string path_base_alphas,std::string path_base_alpha_norms, size_t group_HNSW_M, size_t group_HNSW_efCOnstruction,
        size_t batch_size, size_t nbatches, size_t nb, std::ofstream & record_file){
        Trecorder.reset();
        if (exists(path_index)){
            PrintMessage("Loading pre-constructed index");

            read_index(path_index);
            if (use_group_HNSW) {
                read_group_HNSW(path_group_HNSW);}

            this->base_norms.resize(nb);
            std::ifstream base_norms_input(path_base_norm, std::ios::binary);
            readXvec<float>(base_norms_input, base_norms.data(), nb, 1, false, false);

            if (use_vector_alpha){
                read_base_alphas(path_base_alphas);
                read_base_alpha_norms(path_base_alpha_norms);
            }

            if (use_recording){
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


            std::vector<float> vector_alphas(nb);
            std::ifstream base_alphas_raw_input = std::ifstream(path_alphas_raw, std::ios::binary);
            readXvec<float> (base_alphas_raw_input, vector_alphas.data(), batch_size, nbatches);
            base_alphas_raw_input.close();

            std::vector<float> vector_alpha_norms;

            // Whether the alpha vector is loaded in index
            bool alpha_flag = false;

            if (use_vector_alpha){
                if (exists(path_base_alpha_norms) && exists(path_base_alphas)){
                    read_base_alphas(path_base_alphas);
                    read_base_alpha_norms(path_base_alpha_norms);
                    alpha_flag = true;
                }
                else{
                    //Load the alpha and alpha_norm in nb size
                    size_t n_lq = lq_quantizer_index.size() - 1;
                    assert(index_type[layers-1] == "LQ" && lq_quantizer_index[n_lq].LQ_type == 2);
                    
                    vector_alpha_norms.resize(nb);

                    base_alphas.resize(this->final_group_num);
                    base_alpha_norms.resize(this->final_group_num);

                    for (size_t i = 0; i < final_group_num; i++){
                        base_alphas[i].resize(groups_size[i]);
                        base_alpha_norms[i].resize(groups_size[i]);
                    }

                    std::vector<float> centroid_vector(dimension);
                    std::vector<float> subcentroid(dimension);
                    for (size_t i = 0; i < nb; i++){
                        idx_t group_id, inner_group_id;
                        lq_quantizer_index[n_lq].get_group_id(ids[i], group_id, inner_group_id);
                        float * centroid = lq_quantizer_index[n_lq].upper_centroids.data() + group_id * dimension;
                        idx_t nn_id = lq_quantizer_index[n_lq].nn_centroid_ids[group_id][inner_group_id];
                        float * nn_centroid = lq_quantizer_index[n_lq].upper_centroids.data() + nn_id * dimension;
                        faiss::fvec_madd(dimension, nn_centroid, -1.0, centroid, centroid_vector.data());
                        faiss::fvec_madd(dimension, centroid, vector_alphas[i], centroid_vector.data(), subcentroid.data()); 
                        vector_alpha_norms[i] = faiss::fvec_norm_L2sqr(subcentroid.data(), dimension);
                        if (i % batch_size == 0){
                            std::cout << " Computing alpha norms [ " << i << " / " << nb << " ]";
                            Trecorder.print_time_usage("");
                            record_file << " adding batches [ " << i << " / " << nbatches << " ]";
                            Trecorder.record_time_usage(record_file, " ");
                        }
                    }
                }
            }

            std::ifstream base_input(path_base, std::ios::binary);
            std::vector<float> base_batch(batch_size * dimension);
            std::vector<idx_t> batch_sequence_ids(batch_size);

            bool base_norm_flag = false;
            if (exists(path_base_norm)){
                base_norm_flag = true;
                std::cout << "Loading pre-computed base norm " << std::endl;
                std::ifstream base_norms_input(path_base_norm, std::ios::binary);
                readXvec<float>(base_norms_input, base_norms.data(), nb, 1, false, false);
                base_norms_input.close();
            }

            std::cout << "Start adding batches " << std::endl;
            for (size_t i = 0; i < nbatches; i++){
                readXvecFvec<base_data_type> (base_input, base_batch.data(), dimension, batch_size);
                for (size_t j = 0; j < batch_size; j++){batch_sequence_ids[j] = batch_size * i + j;}

                if (alpha_flag){
                    this->add_batch(batch_size, base_batch.data(), batch_sequence_ids.data(), ids.data() + i * batch_size, 
                        group_position.data()+i*batch_size, base_norm_flag, alpha_flag, vector_alphas.data(), vector_alpha_norms.data());
                }
                else{
                    this->add_batch(batch_size, base_batch.data(), batch_sequence_ids.data(), ids.data() + i * batch_size, 
                    group_position.data()+i*batch_size, base_norm_flag, alpha_flag,
                    vector_alphas.data() + i * batch_size, vector_alpha_norms.data() + i * batch_size);
                }

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
                base_norm_output.write((char *) base_norms.data(), base_norms.size() * sizeof(float));
                base_norm_output.close();
            }

            if (use_vector_alpha && !alpha_flag){
                write_base_alphas(path_base_alphas);
                write_base_alpha_norms(path_base_alpha_norms);
            }

            this->compute_centroid_norm(path_centroid_norm);
            
            if (use_group_HNSW){
                std::cout << "Building group HNSW " << std::endl;
                // Build and save the HNSW index for all indexes
                this->group_HNSW_thres = group_HNSW_thres;

                if(!exists(path_group_HNSW)){
                    size_t num_group_HNSW = 0;
                    std::ofstream group_HNSW_output(path_group_HNSW, std::ios::binary);
                    group_HNSW_output.write((char *) & final_group_num, sizeof(size_t));
                    group_HNSW_output.write((char *) & group_HNSW_thres, sizeof(size_t));

                    for (size_t i = 0; i < final_group_num; i++){if (groups_size[i] >= group_HNSW_thres){num_group_HNSW ++;}}
                    group_HNSW_output.write((char *) & num_group_HNSW, sizeof(size_t));

                    num_group_HNSW = 0;
                    assert(group_HNSW_thres > 0);
                    for (size_t i = 0; i < final_group_num; i++){
                        if (groups_size[i] >= group_HNSW_thres){
                            num_group_HNSW++;
                            std::vector<base_data_type> group_vector(dimension);
                            std::vector<float> float_group_vector(dimension);
                            uint32_t dim;
                            hnswlib::HierarchicalNSW group_HNSW = hnswlib::HierarchicalNSW(dimension, groups_size[i], group_HNSW_M, 2 * group_HNSW_M, 
                                                                                            group_HNSW_efCOnstruction, false, false, pq.code_size, pq.ksub);
                            for (size_t j = 0; j < groups_size[i]; j++){
                                base_input.seekg(base_sequence_ids[i][j] * dimension * sizeof(base_data_type) + base_sequence_ids[i][j] * sizeof(uint32_t), std::ios::beg);
                                base_input.read((char *) & dim, sizeof(uint32_t)); assert(dim == this->dimension);
                                base_input.read((char *) group_vector.data(), sizeof(base_data_type) * dimension);
                                for (size_t temp = 0; temp < dimension; temp++){
                                    float_group_vector[temp] = group_vector[temp];
                                }
                                group_HNSW.addPoint(float_group_vector.data());
                            }

                            writeBinaryPOD(group_HNSW_output, i);
                            writeBinaryPOD(group_HNSW_output, group_HNSW.maxelements_);
                            writeBinaryPOD(group_HNSW_output, group_HNSW.enterpoint_node);
                            writeBinaryPOD(group_HNSW_output, group_HNSW.offset_data);
                            writeBinaryPOD(group_HNSW_output, group_HNSW.M_);
                            writeBinaryPOD(group_HNSW_output, group_HNSW.maxM_);
                            writeBinaryPOD(group_HNSW_output, group_HNSW.size_links_level0);
                            writeBinaryPOD(group_HNSW_output, group_HNSW.efConstruction_);
                            writeBinaryPOD(group_HNSW_output, group_HNSW.efSearch);

                            for (size_t temp = 0; temp < group_HNSW.maxelements_; temp++){
                                uint8_t *ll_cur = group_HNSW.get_linklist0(temp);
                                uint32_t size = *ll_cur;
                                group_HNSW_output.write((char *) &size, sizeof(uint32_t));
                                hnswlib::idx_t *data = (hnswlib::idx_t *)(ll_cur + 1);
                                group_HNSW_output.write((char *) data, sizeof(hnswlib::idx_t) * size);
                            }
                        }
                    }
                    group_HNSW_output.write((char *) & num_group_HNSW, sizeof(size_t));
                    group_HNSW_output.close();
                }
                read_group_HNSW(path_group_HNSW);
            }

            //In order to save disk usage
            //Annotate the write_index function
            if (this->use_saving_index){
                this->write_index(path_index);
            }
            std::string message = "Constructed and wrote the index ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }

    void Bslib_Index::index_statistic(std::string path_base, std::string path_ids, 
                                        std::string path_alphas_raw, size_t nb, size_t nbatch){
        // Average distance between the base vector and centroid
        std::ifstream vector_ids(path_ids, std::ios::binary);
        std::ifstream base_vectors(path_base, std::ios::binary);

        

        std::vector<float> vectors(dimension * 100);
        std::vector<idx_t> ids(nb);
    
        readXvec<idx_t>(vector_ids, ids.data(), nb / nbatch, 1);
        readXvecFvec<base_data_type> (base_vectors, vectors.data(), dimension, 100);

        std::vector<float> alphas_raw(nb / nbatch);

        if (use_vector_alpha){
            std::ifstream base_alphas(path_alphas_raw, std::ios::binary);
            readXvec<float>(base_alphas, alphas_raw.data(), nb / nbatch, 1);
        }

        size_t test_size = 100;
        std::vector<float> vector_residuals(dimension * test_size);

        encode(test_size, vectors.data(), ids.data(), vector_residuals.data(), alphas_raw.data());

        size_t n_lq = lq_quantizer_index.size() - 1;
        std::cout << "LQ Type: " << lq_quantizer_index[n_lq].LQ_type << std::endl;
        float avg_dist = 0;
        for (size_t i = 0; i < test_size; i++){
            float alpha;
            if (use_vector_alpha){
                alpha = alphas_raw[i];
            }
            else{
                idx_t group_id, inner_group_id;
                lq_quantizer_index[n_lq].get_group_id(ids[i], group_id, inner_group_id);
                alpha = lq_quantizer_index[n_lq].alphas[group_id][inner_group_id];
            }
            float dist = faiss::fvec_norm_L2sqr(vector_residuals.data() + i * dimension, dimension);
            avg_dist += dist;
            std::cout <<  dist << " " << alpha << " ";
        }
        std::cout << std::endl << "Ag dist: " << avg_dist / test_size<< std::endl;
        /*
        std::cout << "Avg b c distance: " << b_c_dist / 1000000 << std::endl;

        if (!use_vector_alpha){
            size_t n_lq = lq_quantizer_index.size() - 1;
            size_t n_g = lq_quantizer_index[n_lq].nc_upper;
            size_t n_max = lq_quantizer_index[n_lq].max_nc_per_group;
            for (size_t i = 0; i < n_g; i++){
                for (size_t j = 0; j < n_max; j++){
                    std::cout << lq_quantizer_index[n_lq].alphas[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
        */
    }


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

            if (use_OPQ){
                reverse_OPQ(nq, queries.data());
            }

            for (size_t i = 0; i < nq; i++){
                std::unordered_set<idx_t> gt;

                //for (size_t j = 0; j < recall_k; j++){
                for (size_t j = 0; j < 1; j++){
                    gt.insert(groundtruth[ngt * i + j]);
                }

                //assert (gt.size() == recall_k);

                for (size_t j = 0; j < recall_k; j++){
                    if (gt.count(query_labels[i * recall_k + j]) != 0){
                        correct ++;
                    }
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


    void Bslib_Index::read_group_HNSW(const std::string path_group_HNSW){
        
        // Load the index 
        std::ifstream group_HNSW_input(path_group_HNSW, std::ios::binary);
        size_t record_group_num, record_HNSW_thres, num_group_HNSW;
        readBinaryPOD(group_HNSW_input, record_group_num); assert(record_group_num == final_group_num);
        readBinaryPOD(group_HNSW_input, record_HNSW_thres); this->group_HNSW_thres = group_HNSW_thres;
        readBinaryPOD(group_HNSW_input, num_group_HNSW);
        this->group_HNSW_list.resize(num_group_HNSW);

        for (size_t i = 0; i < num_group_HNSW; i++){
            hnswlib::HierarchicalNSW * group_HNSW = new hnswlib::HierarchicalNSW(true, this->use_vector_alpha);
            group_HNSW->code_size = pq.code_size; group_HNSW->ksub = pq.ksub;

            size_t final_group_id;
            readBinaryPOD(group_HNSW_input, final_group_id); 
            group_HNSW_idxs[final_group_id] = i;
            if (use_vector_alpha){
                size_t n_lq = lq_quantizer_index.size() - 1;
                idx_t lq_group_id, inner_group_id;

                lq_quantizer_index[n_lq].get_group_id(final_group_id, lq_group_id, inner_group_id);
                group_HNSW->nn_dist = lq_quantizer_index[n_lq].nn_centroid_dists[lq_group_id][inner_group_id];
                group_HNSW->vector_alpha_norm = this->base_alpha_norms[final_group_id].data();
                group_HNSW->vector_alpha = this->base_alphas[final_group_id].data();
            }
            else{
                group_HNSW->centroid_norm = this->centroid_norms[final_group_id];
            }

            group_HNSW->base_norms = this->base_norms.data();
            group_HNSW->base_sequece_id_list = this->base_sequence_ids[final_group_id].data();
            group_HNSW->base_code_point = base_codes[final_group_id].data();

            readBinaryPOD(group_HNSW_input, group_HNSW->maxelements_);
            readBinaryPOD(group_HNSW_input, group_HNSW->enterpoint_node);
            readBinaryPOD(group_HNSW_input, group_HNSW->offset_data);
            readBinaryPOD(group_HNSW_input, group_HNSW->M_);
            readBinaryPOD(group_HNSW_input, group_HNSW->maxM_);
            readBinaryPOD(group_HNSW_input, group_HNSW->size_links_level0);
            readBinaryPOD(group_HNSW_input, group_HNSW->efConstruction_);
            readBinaryPOD(group_HNSW_input, group_HNSW->efSearch);
            group_HNSW->data_size_ = 0;
            group_HNSW->size_data_per_element = group_HNSW->size_links_level0;

            group_HNSW->data_level0_memory_ = (char *) malloc(group_HNSW->maxelements_ * group_HNSW->size_data_per_element);

            group_HNSW->efConstruction_ = 0;
            group_HNSW->cur_element_count = group_HNSW->maxelements_;

            group_HNSW->visitedlistpool = new hnswlib::VisitedListPool(1, group_HNSW->maxelements_);
            
            uint32_t edge_size;

            for (size_t temp = 0; temp < group_HNSW->maxelements_; temp++) {
                group_HNSW_input.read((char *) & edge_size, sizeof(uint32_t));

                uint8_t *ll_cur = group_HNSW->get_linklist0(temp);
                *ll_cur = edge_size;
                hnswlib::idx_t *data = (hnswlib::idx_t *)(ll_cur + 1);

                group_HNSW_input.read((char *) data, edge_size * sizeof(hnswlib::idx_t));
            }
            this->group_HNSW_list[i] = group_HNSW;
        }
    }

    void Bslib_Index::write_base_alphas(std::string path_base_alpha){
        assert(use_vector_alpha);
        assert(base_alphas.size() == this->final_group_num);
        std::ofstream base_alphas_output(path_base_alpha, std::ios::binary);
        base_alphas_output.write((char *) & final_group_num, sizeof(size_t));
        for (size_t i = 0; i < final_group_num; i++){
            size_t group_size = this->base_alphas[i].size();
            base_alphas_output.write((char * ) & group_size, sizeof(size_t));
            base_alphas_output.write((char * ) this->base_alphas[i].data(), group_size * sizeof(float));
        }
        base_alphas_output.close();
    }

    void Bslib_Index::read_base_alphas(std::string path_base_alpha){
        std::cout << "Loading base alphas " << std::endl;
        assert(use_vector_alpha);
        assert(this->base_alphas.size() == 0);
        std::ifstream base_alphas_input (path_base_alpha, std::ios::binary);
        size_t group_num;
        base_alphas_input.read((char *) & group_num, sizeof(size_t));
        
        assert(group_num == this->final_group_num);
        this->base_alphas.resize(group_num);
        size_t group_size;
        for (size_t i = 0; i < group_num; i++){
            base_alphas_input.read((char *) & group_size, sizeof(size_t));
            this-> base_alphas[i].resize(group_size);
            base_alphas_input.read((char *) this->base_alphas[i].data(), group_size * sizeof(float));
        }
    }

    void Bslib_Index::write_base_alpha_norms(std::string path_base_alpha_norm){
        
        assert(use_vector_alpha);
        assert(base_alpha_norms.size() == final_group_num);
        std::ofstream base_alphas_norms_output(path_base_alpha_norm, std::ios::binary);
        base_alphas_norms_output.write((char *) & final_group_num, sizeof(size_t));
        for (size_t i = 0; i < final_group_num; i++){
            size_t group_size = base_alpha_norms[i].size();
            base_alphas_norms_output.write((char *) & group_size, sizeof(size_t));
            base_alphas_norms_output.write((char *) this->base_alpha_norms[i].data(), group_size * sizeof(float));
        }
        base_alphas_norms_output.close();
    }

    void Bslib_Index::read_base_alpha_norms(std::string path_base_alpha_norm){
        std::cout << "Loading base alpha norms " << std::endl;
        assert(use_vector_alpha);
        assert(this->base_alpha_norms.size() == 0);
        std::ifstream base_alpha_norms_input(path_base_alpha_norm, std::ios::binary);
        size_t group_num;
        base_alpha_norms_input.read((char *) & group_num, sizeof(size_t));
        assert(group_num == this->final_group_num);
        this->base_alpha_norms.resize(group_num);
        size_t group_size;
        for (size_t i = 0; i < group_num; i++){
            base_alpha_norms_input.read((char *) & group_size, sizeof(size_t));
            this->base_alpha_norms[i].resize(group_size);
            base_alpha_norms_input.read((char *) this->base_alpha_norms[i].data(), group_size * sizeof(float));
        }
    }
}

