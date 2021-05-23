
#include <fstream>
#include <unordered_set>
#include <sys/resource.h>
#include <string>

#include "bslib_index.h"
#include "./parameters/parameters_billion/parameters_VQ.h"

using namespace bslib;

int main(){
    memory_recorder Mrecorder = memory_recorder();
    time_recorder Trecorder = time_recorder();
    recall_recorder Rrecorder = recall_recorder();
    std::string message;

/*Prepare the work space*/
    PrepareFolder((char *) folder_model.c_str());
    PrepareFolder((char *) (std::string(folder_model)+"/" + dataset).c_str());
    std::cout << "Preparing work space: " << folder_model << std::endl;

    //For recording 
    std::ofstream record_file;
    std::ofstream qps_record_file;
    qps_record_file.open(path_speed_record, std::ios::binary);
    qps_record_file << "The num of centroid is: " << std::to_string(ncentroids[0]) << std::endl;
    if (is_recording){
        record_file.open(path_record, std::ios::app);
        time_t now = time(0);
        char* dt = ctime(&now);
        record_file << std::endl << "The time now is " << dt << std::endl;
        record_file << "The memory usage format is ixrss, isrss, idrss, maxrss" << std::endl;
        record_file << "Now starting the indexing process " << std::endl;
    }

/*Train the residual PQ and norm PQ*/
    //Initialize the index
    PrintMessage("Initializing the index");
    Trecorder.reset();
    Bslib_Index index = Bslib_Index(dimension, layers, index_type, use_HNSW_VQ, use_norm_quantization, use_OPQ);
    index.train_size = train_size;
    index.M = M_PQ;
    index.nbits = nbits;
    index.use_train_selector = use_train_selector;

    if (use_OPQ){
        if (exists(path_OPQ)){
            index.opq_matrix = static_cast<faiss::OPQMatrix *>(faiss::read_VectorTransform(path_OPQ.c_str()));
        }
        else{
            PrintMessage("Training the OPQ matrix");
            index.opq_matrix = new faiss::OPQMatrix(dimension, M_PQ);
            index.opq_matrix->verbose = true;
            std::ifstream learn_input(path_learn, std::ios::binary);
            std::vector<float>  origin_train_set(train_size * dimension);
            readXvecFvec<learn_data_type>(learn_input, origin_train_set.data(), dimension, train_size, false);
            
            if (OPQ_train_size < train_size){
                std::vector<float> OPQ_train_set(OPQ_train_size * dimension);
                RandomSubset(origin_train_set.data(), OPQ_train_set.data(), dimension, train_size, OPQ_train_size);
                std::cout<< "Randomly select the train set for OPQ training" << std::endl;
                index.opq_matrix->train(OPQ_train_size, OPQ_train_set.data());
            }
            else{
                index.opq_matrix->train(train_size, origin_train_set.data());
            }
            faiss::write_VectorTransform(index.opq_matrix, path_OPQ.c_str());
        }
        message = "Trained the OPQ matrix, ";
        Mrecorder.print_memory_usage(message);
        Mrecorder.record_memory_usage(record_file,  message);
        Trecorder.print_time_usage(message);
        Trecorder.record_time_usage(record_file, message);
    }

    if (use_train_selector){
        index.build_train_selector(path_learn, path_groups, path_labels, train_size, selector_train_size, selector_group_size);
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

    index.build_quantizers(ncentroids, path_quantizers, path_learn, num_train, HNSW_paras, PQ_paras);
    //for (size_t i = 0; i < ncentroids[0]; i++){std::cout << index.lq_quantizer_index[0].alphas[i] << " ";}std::cout << std::endl;
    index.get_final_group_num();
    message = "Initialized the index, ";
    Mrecorder.print_memory_usage(message);
    Mrecorder.record_memory_usage(record_file,  message);
    Trecorder.print_time_usage(message);
    Trecorder.record_time_usage(record_file, message);

    //Precompute the base vector idxs
    if (!exists(path_ids)){
        Trecorder.reset();
        PrintMessage("Assigning the points");
        
        if (use_parallel_indexing){
            //The parallel version of assigning points
            std::ofstream base_output (path_ids, std::ios::binary);
            std::vector<idx_t> assigned_ids(nb);

#pragma omp parallel for 
            for (size_t i = 0; i < nbatches; i++){
                std::vector<float> batch(batch_size * dimension);
                std::ifstream base_input(path_base, std::ios::binary);
                base_input.seekg(i * batch_size * dimension * sizeof(base_data_type) + i * batch_size * sizeof(uint32_t), std::ios::beg);
                readXvecFvec<base_data_type> (base_input, batch.data(), dimension, batch_size);
                if (use_OPQ) {index.do_OPQ(batch_size, batch.data());}
                index.assign(batch_size, batch.data(), assigned_ids.data() + i * batch_size);
                base_input.close();
            }

            message = "[Assigned the index ]";
            Trecorder.print_time_usage(message);

            for (size_t i = 0; i < nbatches; i++){
                base_output.write((char *) & batch_size, sizeof(uint32_t));
                base_output.write((char *) assigned_ids.data() + i * batch_size * sizeof(idx_t), batch_size * sizeof(idx_t));
            }
            base_output.close();
            message = "Assigned the base vectors in parallel mode";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }

        else{
            std::ifstream base_input (path_base, std::ios::binary);
            std::ofstream base_output (path_ids, std::ios::binary);

            std::vector <float> batch(batch_size * dimension);
            std::vector<idx_t> assigned_ids(batch_size);

            for (size_t i = 0; i < nbatches; i++){
                readXvecFvec<base_data_type> (base_input, batch.data(), dimension, batch_size, false, true);
                if (use_OPQ) {index.do_OPQ(batch_size, batch.data());}
                index.assign(batch_size, batch.data(), assigned_ids.data());
                base_output.write((char * ) & batch_size, sizeof(uint32_t));
                base_output.write((char *) assigned_ids.data(), batch_size * sizeof(idx_t));
                if (i % 10 == 0){
                    std::cout << " assigned batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.print_time_usage("");
                }
            }
            base_input.close();
            base_output.close();
            message = "Assigned the base vectors in sequential mode";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
    }

    //Train the PQ quantizer
    PrintMessage("Constructing the PQ");
    Trecorder.reset();
    if (exists(path_pq)){
        std::cout << "Loading PQ codebook from " << path_pq << std::endl;
        index.pq = * faiss::read_ProductQuantizer(path_pq.c_str());
        index.code_size = index.pq.code_size;

        if(use_norm_quantization){
            std::cout << "Loading norm PQ codebook from " << path_pq_norm << std::endl;
            index.norm_pq = * faiss::read_ProductQuantizer(path_pq_norm.c_str());
            index.norm_code_size = index.norm_pq.code_size;
        }
    }
    else
    {
        index.norm_M = M_norm_PQ;
        
        std::cout << "Training PQ codebook" << std::endl;
        index.train_pq(path_pq, path_pq_norm, path_learn, PQ_train_size);
    }
    std::cout << "Checking the PQ " << index.pq.code_size << std::endl;
    message = "Trained the PQ, ";
    Mrecorder.print_memory_usage(message);
    Mrecorder.record_memory_usage(record_file,  message);
    Trecorder.print_time_usage(message);
    Trecorder.record_time_usage(record_file, message);

    //Build the index
    if (exists(path_index)){
        PrintMessage("Loading index");
        Trecorder.reset();
        index.read_index(path_index);

        message = "Loaded index ";
        Mrecorder.print_memory_usage(message);
        Mrecorder.record_memory_usage(record_file,  message);
        Trecorder.print_time_usage(message);
        Trecorder.record_time_usage(record_file, message);
    }
    else{
        std::vector<idx_t> ids(nb); std::ifstream ids_input(path_ids, std::ios::binary);
        readXvec<idx_t> (ids_input, ids.data(), batch_size, nbatches); 
        //std::vector<idx_t> pre_hash_ids; if (use_hash) {pre_hash_ids.resize(nb, 0); memcpy(pre_hash_ids.data(), ids.data(), nb * sizeof(idx_t)); HashMapping(nb, pre_hash_ids.data(), ids.data(), index.final_group_num);}
        std::vector<size_t> groups_size(index.final_group_num, 0); std::vector<size_t> group_position(nb, 0);
        for (size_t i = 0; i < nb; i++){group_position[i] = groups_size[ids[i]]; groups_size[ids[i]] ++;}

        PrintMessage("Constructing the index");
        index.base_codes.resize(index.final_group_num);
        index.base_sequence_ids.resize(index.final_group_num);
        if (use_norm_quantization){index.base_norm_codes.resize(nb);} else{index.base_norms.resize(nb);}
        //if (use_hash) index.base_pre_hash_ids.resize(index.final_group_num);
        for (size_t i = 0; i < index.final_group_num; i++){
            index.base_codes[i].resize(groups_size[i] * index.code_size);
            index.base_sequence_ids[i].resize(groups_size[i]);
            //if(use_hash) index.base_pre_hash_ids[i].resize(groups_size[i]);
        }

        Trecorder.reset();

        if (use_parallel_indexing){
        
#pragma omp parallel for
            for (size_t i = 0; i < nbatches; i++){
                std::vector<float> base_batch(batch_size * dimension);
                std::vector<idx_t> batch_sequence_ids(batch_size);
                std::ifstream base_input(path_base, std::ios::binary);
                base_input.seekg(i * batch_size * dimension * sizeof(base_data_type) + i * batch_size * sizeof(uint32_t), std::ios::beg);
                readXvecFvec<base_data_type> (base_input, base_batch.data(), dimension, batch_size);
                if (use_OPQ) {index.do_OPQ(batch_size, base_batch.data());}

                for (size_t j = 0; j < batch_size; j++){batch_sequence_ids[j] = batch_size * i + j;}
                index.add_batch(batch_size, base_batch.data(), batch_sequence_ids.data(), ids.data() + i * batch_size, group_position.data()+i * batch_size);
            }
        }
        else{
            std::ifstream base_input(path_base, std::ios::binary);
            std::vector<float> base_batch(batch_size * dimension);
            std::vector<idx_t> batch_sequence_ids(batch_size);


            for (size_t i = 0; i < nbatches; i++){
                readXvecFvec<base_data_type> (base_input, base_batch.data(), dimension, batch_size);
                if (use_OPQ) {index.do_OPQ(batch_size, base_batch.data());}
                for (size_t j = 0; j < batch_size; j++){batch_sequence_ids[j] = batch_size * i + j;}

                index.add_batch(batch_size, base_batch.data(), batch_sequence_ids.data(), ids.data() + i * batch_size, group_position.data()+i*batch_size);
                if (i % 10 == 0){
                    std::cout << " adding batches [ " << i << " / " << nbatches << " ]";
                    Trecorder.print_time_usage("");
                }
            }
        }
        message = "[Added batches to the index] ";
        Trecorder.print_time_usage(message);

        index.compute_centroid_norm();
        index.write_index(path_index);
        
        message = "Constructed and wrote the index ";
        Mrecorder.print_memory_usage(message);
        Mrecorder.record_memory_usage(record_file,  message);
        Trecorder.print_time_usage(message);
        Trecorder.record_time_usage(record_file, message);
    }

        std::cout << "The base id distribution in all groups is " << std::endl;
        int empty_clusters = 0;
        float average_number = 0;
        float total_number = 0;
        for (size_t i = 0; i < index.base_sequence_ids.size(); i++){
            total_number += index.base_sequence_ids[i].size();
        }
        average_number = total_number / index.base_sequence_ids.size();

        float std = 0;
        float smaller_10 = 0;
        float smaller_1 = 0;
        for (size_t i = 0; i < index.base_sequence_ids.size(); i++){
            //std::cout << index.base_sequence_ids[i].size() << " ";
            if (index.base_sequence_ids[i].size() == 0){
                empty_clusters ++;
            }
            if (index.base_sequence_ids[i].size() < 0.1 * average_number){
                smaller_10 += 1;
            }
            if (index.base_sequence_ids[i].size() < 0.01 * average_number){
                smaller_1 += 1;
            }
            float subtraction = index.base_sequence_ids[i].size() - average_number;
            std += subtraction * subtraction;
        }
        std /= index.base_sequence_ids.size();

        std::cout << std::endl << "The total number of empty clusters is: " << empty_clusters <<  std::endl;
        std::cout << "The aaverage number of data point in each cluster is: " << average_number << std::endl;
        std::cout << "The std of the data points distribution is: " << std << std::endl;
        std::cout << "The skewness is: " << sqrt(std) / average_number << std::endl;
        std::cout << "The 10 percent is: " << smaller_10 / index.base_sequence_ids.size() << std::endl;
        std::cout << "The 1 percent is: " << smaller_1 / index.base_sequence_ids.size() << std::endl;


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

    // Evaluating the search performance with various search performance settings
    for (size_t i = 0; i < num_search_paras; i++){
        index.use_reranking = use_reranking;

        index.max_visited_vectors = max_vectors[i];
        for (size_t j = 0; j < num_recall; j++){
            if (use_reranking) index.reranking_space = reranking_space[j];
            size_t recall_k = result_k[j];
            std::vector<float> query_distances(nq * recall_k);
            std::vector<faiss::Index::idx_t> query_labels(nq * recall_k);
            size_t correct = 0;
            
            Trecorder.reset();
            index.search(nq, recall_k, queries.data(), query_distances.data(), query_labels.data(), keep_space+ i * layers, groundtruth.data(), path_base);
            std::cout << "The qps for searching is: " << Trecorder.getTimeConsumption() / nq << " us " << std::endl;
            message = "Finish Search";
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
            Trecorder.record_time_usage(qps_record_file, message);
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file, message);
            

            for (size_t i = 0; i < nq; i++){
                std::unordered_set<idx_t> gt;

                for (size_t j = 0; j < recall_k; j++){
                    gt.insert(groundtruth[ngt * i + j]);
                }

                assert (gt.size() == recall_k);
                for (size_t j = 0; j < recall_k; j++){
                    if (gt.count(query_labels[i * recall_k + j]) != 0)
                        correct ++;
                }
            }
            float recall = float(correct) / (recall_k * nq);
            Rrecorder.print_recall_performance(nq, recall, recall_k, search_mode, layers, keep_space + i * layers, max_vectors[i]);
            Rrecorder.record_recall_performance(record_file, nq, recall, recall_k, search_mode, layers, keep_space + i * layers, max_vectors[i]);
            Rrecorder.record_recall_performance(qps_record_file, nq, recall, recall_k, search_mode, layers, keep_space + i * layers, max_vectors[i]);

            if (use_reranking){
                std::cout << " with reranking parameter: " << index.reranking_space << std::endl;
            } 
            std::cout << std::endl;
        }
    }
    record_file.close();
}




