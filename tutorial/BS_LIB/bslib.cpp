
#include <fstream>
#include <unordered_set>
#include <sys/resource.h>
#include <string>

#include "bslib_index.h"
#include "parameters_millions.h"


using namespace bslib;

int main(){
    memory_recorder Mrecorder = memory_recorder();
    time_recorder Trecorder = time_recorder();
    std::string message;

/*Prepare the work space*/
    PrepareFolder(folder_model);
    PrepareFolder((char *) (std::string(folder_model)+"/SIFT1M").c_str());
    std::cout << "Preparing work space: " << folder_model << std::endl;

    //For recording 
    std::ofstream record_file;
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
    Bslib_Index * index = new Bslib_Index(dimension, layers, index_type, use_HNSW_VQ, use_norm_quantization);
    index->train_size = train_size;

    index->build_train_selector(path_learn, path_groups, path_labels, train_size, selector_train_size, selector_group_size);
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

    index->build_quantizers(ncentroids, path_quantizers, path_learn, num_train, HNSW_paras, PQ_paras);
    index->get_final_nc();
    message = "Initialized the index, ";
    Mrecorder.print_memory_usage(message);
    Mrecorder.record_memory_usage(record_file,  message);
    Trecorder.print_time_usage(message);
    Trecorder.record_time_usage(record_file, message);


    //Precompute the base vector idxs
    if (!exists(path_idxs)){
        Trecorder.reset();
        PrintMessage("Assigning the points");
        std::ifstream base_input (path_base, std::ios::binary);
        std::ofstream base_output (path_idxs, std::ios::binary);

        std::vector <float> batch(batch_size * dimension);
        std::vector<idx_t> assigned_ids(batch_size);

        for (size_t i = 0; i < nbatches; i++){
            readXvecFvec<origin_data_type> (base_input, batch.data(), dimension, batch_size);
            index->assign(batch_size, batch.data(), assigned_ids.data());
            base_output.write((char * ) & batch_size, sizeof(uint32_t));
            base_output.write((char *) assigned_ids.data(), batch_size * sizeof(idx_t));
            if (i % 10 == 0){
                std::cout << " assigned batches [ " << i << " / " << nbatches << " ]";
                Trecorder.print_time_usage("");
            }
        }
        base_input.close();
        base_output.close();
        message = "Assigned the base vectors";
        Mrecorder.print_memory_usage(message);
        Mrecorder.record_memory_usage(record_file,  message);
        Trecorder.print_time_usage(message);
        Trecorder.record_time_usage(record_file, message);
    }

    //Train the PQ quantizer
    PrintMessage("Constructing the PQ");
    Trecorder.reset();
    if (exists(path_pq) && exists(path_pq_norm)){
        std::cout << "Loading PQ codebook from " << path_pq << std::endl;
        index->pq = * faiss::read_ProductQuantizer(path_pq);
        index->norm_pq = * faiss::read_ProductQuantizer(path_pq_norm);
        index->code_size = index->pq.code_size;
        index->norm_code_size = index->norm_pq.code_size;
    }
    else
    {
        index->M = M_PQ;
        index->norm_M = M_norm_PQ;
        index->nbits = nbits;
        
        std::cout << "Training PQ codebook" << std::endl;
        index->train_pq(path_pq, path_pq_norm, path_learn, PQ_train_size);
    }
    std::cout << "Checking the PQ " << index->pq.code_size << index->norm_pq.code_size << std::endl;
    message = "Trained the PQ, ";
    Mrecorder.print_memory_usage(message);
    Mrecorder.record_memory_usage(record_file,  message);
    Trecorder.print_time_usage(message);
    Trecorder.record_time_usage(record_file, message);
    
    //Build the index
    if (exists(path_index)){
        PrintMessage("Loading index");
        Trecorder.reset();
        index->read_index(path_index);
        
        message = "Loaded index";
        Mrecorder.print_memory_usage(message);
        Mrecorder.record_memory_usage(record_file,  message);
        Trecorder.print_time_usage(message);
        Trecorder.record_time_usage(record_file, message);
    }
    else{
        PrintMessage("Constructing the index");
        index->base_codes.resize(index->final_nc);
        if (use_norm_quantization)
            index->base_norm_codes.resize(index->final_nc);
        else
            index->base_norm.resize(index->final_nc);
        index->origin_ids.resize(index->final_nc);

        Trecorder.reset();
        std::ifstream base_input(path_base, std::ios::binary);
        std::ifstream idx_input(path_idxs, std::ios::binary);

        std::vector<idx_t> idxs(batch_size);
        std::vector<float> batch(batch_size * dimension);
        std::vector<idx_t> ids(batch_size);

        for (size_t i = 0; i < nbatches; i++){
            readXvec<idx_t>(idx_input, idxs.data(), batch_size, 1);
            readXvecFvec<origin_data_type> (base_input, batch.data(), dimension, batch_size);

            for (size_t j = 0; j < batch_size; j++){
                ids[j] = batch_size * i + j;
            }

            index->add_batch(batch_size, batch.data(), ids.data(), idxs.data());
            if (i % 10 == 0){
                std::cout << " adding batches [ " << i << " / " << nbatches << " ]";
                Trecorder.print_time_usage("");
            }
        }

        index->compute_centroid_norm();

        index->write_index(path_index);
        message = "Constructed and wrote the index ";
        Mrecorder.print_memory_usage(message);
        Mrecorder.record_memory_usage(record_file,  message);
        Trecorder.print_time_usage(message);
        Trecorder.record_time_usage(record_file, message);
    }

    record_file.close();
    PrintMessage("Loading groundtruth");
    std::vector<uint32_t> groundtruth(nq * ngt);
    {
        std::ifstream gt_input(path_gt, std::ios::binary);
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, true, false);
    }

    PrintMessage("Loading queries");
    std::vector<float> queries(nq * dimension);
    {
        std::ifstream query_input(path_query, std::ios::binary);
        readXvecFvec<origin_data_type>(query_input, queries.data(), dimension, nq, true, false);
    }

    index->use_reranking = use_reranking;
    index->reranking_space = reranking_space;

    index->max_visited_vectors = max_vectors;
    index->precomputed_table.resize(index->pq.M * index->pq.ksub);
    std::vector<float> query_distances(nq * result_k);
    std::vector<faiss::Index::idx_t> query_labels(nq * result_k);
    size_t correct = 0;
    
    index->search(nq, result_k, queries.data(), query_distances.data(), query_labels.data(), keep_space, groundtruth.data());
    
    std::cout << "The qps for searching is: " << Trecorder.getTimeConsumption() / nq << " us " << std::endl;
    message = "Finish Search";
    Trecorder.print_time_usage(message);
    Mrecorder.print_memory_usage(message);

    std::vector<float> base_dataset(nb * dimension);
    std::ifstream base_input(path_base, std::ios::binary);
    readXvecFvec<origin_data_type> (base_input, base_dataset.data(), dimension, nb);


    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt;

        for (size_t j = 0; j < result_k; j++){
            gt.insert(groundtruth[ngt * i + j]);
            /*
            std::cout << query_labels[i * result_k + j] << " ";
            float * nn = base_dataset.data() + query_labels[i * result_k + j] * dimension;
            float * query = queries.data() + i * dimension;
            std::vector<float> distance_vector(dimension);
            faiss::fvec_madd(dimension, nn, -1, query, distance_vector.data());
            std::cout << query_distances[i * result_k + j] << " " << faiss::fvec_norm_L2sqr(distance_vector.data(), dimension) << " ";
            */
        }


        assert (gt.size() == result_k);
        for (size_t j = 0; j < result_k; j++){
            if (gt.count(query_labels[i * result_k + j]) != 0)
                correct ++;
        }
    }

    std::cout << "The recall is: " << float(correct) / (result_k * nq) ;
    if (use_reranking){
        std::cout << " with reranking parameter: " << index->reranking_space << std::endl;
    } 
    std::cout << std::endl;

}




