
#include <fstream>
#include <unordered_set>
#include <sys/resource.h>

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
    ShowMessage("Initializing the index");
    Trecorder.reset();
    Bslib_Index * index = new Bslib_Index(dimension, layers, index_type, use_subset);
    index->nt = nt;
    index->subnt = subnt;

    index->build_quantizers(ncentroids, path_quantizers, path_learn);
    index->get_final_nc();
    message = "Initialized the index, ";
    Mrecorder.print_memory_usage(message);
    Mrecorder.record_memory_usage(record_file,  message);
    Trecorder.print_time_usage(message);
    Trecorder.record_time_usage(record_file, message);

    //Precompute the base vector idxs
    if (!exists(path_idxs)){
        Trecorder.reset();
        ShowMessage("Assigning the points");
        std::ifstream base_input (path_base, std::ios::binary);
        std::ofstream base_output (path_idxs, std::ios::binary);

        std::vector <float> batch(batch_size * dimension);
        std::vector<idx_t> assigned_idxs(batch_size);

        for (size_t i = 0; i < nbatches; i++){
            readXvecFvec<origin_data_type> (base_input, batch.data(), dimension, batch_size);
            index->assign(batch_size, batch.data(), assigned_idxs.data());
            base_output.write((char * ) & batch_size, sizeof(uint32_t));
            base_output.write((char *) assigned_idxs.data(), batch_size * sizeof(idx_t));
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
    ShowMessage("Training the PQ");
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
        index->nt = nt;
        index->M = bytes_per_code;
        index->norm_M = bytes_per_norm_code;
        index->nbits = nbits;
        
        std::cout << "Training PQ codebook" << std::endl;
        index->train_pq(path_pq, path_pq_norm, path_learn);
    }
    std::cout << "Checking the PQ " << index->pq.code_size << index->norm_pq.code_size << std::endl;
    message = "Trained the PQ, ";
    Mrecorder.print_memory_usage(message);
    Mrecorder.record_memory_usage(record_file,  message);
    Trecorder.print_time_usage(message);
    Trecorder.record_time_usage(record_file, message);
    



    //Build the index
    if (exists(path_index)){
        ShowMessage("Loading index");
        Trecorder.reset();
        index->read_index(path_index);
        message = "Loaded index";
        Mrecorder.print_memory_usage(message);
        Mrecorder.record_memory_usage(record_file,  message);
        Trecorder.print_time_usage(message);
        Trecorder.record_time_usage(record_file, message);
    }
    else{
        ShowMessage("Constructing the index");
        index->base_codes.resize(index->final_nc);
        index->base_norm_codes.resize(index->final_nc);
        index->origin_ids.resize(index->final_nc);


        Trecorder.reset();
        std::ifstream base_input(path_base, std::ios::binary);
        std::ifstream idx_input(path_idxs, std::ios::binary);

        std::vector<idx_t> idxs(batch_size);
        std::vector<float> batch(batch_size * dimension);
        std::vector<idx_t> origin_ids(batch_size);

        for (size_t i = 0; i < nbatches; i++){
            readXvec<uint32_t>(idx_input, idxs.data(), batch_size, 1);
            readXvecFvec<origin_data_type> (base_input, batch.data(), dimension, batch_size);

            for (size_t j = 0; j < batch_size; j++){
                origin_ids[i] = batch_size * i + j;
            }

            index->add_batch(batch_size, batch.data(), origin_ids.data(), idxs.data());
            if (i % 10 == 0){
                std::cout << " adding batches [ " << i << " / " << nbatches << " ]";
                Trecorder.print_time_usage("");
            }
        }

        index->centroid_norms.resize(index->final_nc);
        index->centroid_norm_codes.resize(index->final_nc * index->norm_code_size);

        index->compute_centroid_norm();
        index->write_index(path_index);
        message = "Constructed the index";
        Mrecorder.print_memory_usage(message);
        Mrecorder.record_memory_usage(record_file,  message);
        Trecorder.print_time_usage(message);
        Trecorder.record_time_usage(record_file, message);
    }

    record_file.close();
    ShowMessage("Loading groundtruth");
    std::vector<uint32_t> groundtruth(nq * ngt);
    {
        std::ifstream gt_input(path_gt, std::ios::binary);
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, true, false);
    }

    ShowMessage("Loading queries");
    std::vector<float> query(nq * dimension);
    {
        std::ifstream query_input(path_query, std::ios::binary);
        readXvecFvec<origin_data_type>(query_input, query.data(), dimension, nq, true, false);
    }

    exit(0);
    index->max_visited_vectors = max_vectors;
    index->precomputed_table.resize(index->pq.M * index->pq.ksub);
    std::vector<float> query_distances(nq * result_k);
    std::vector<faiss::Index::idx_t> query_labels(nq * result_k);
    size_t correct = 0;
    size_t k = 1;
    index->search(nq, result_k, query.data(), query_distances.data(), query_labels.data(), keep_space);
    std::cout << "The qps for searching is: " << Trecorder.getTimeConsumption() / nq << " us " << std::endl;
    message = "Finish Search";
    Trecorder.print_time_usage(message);
    Mrecorder.print_memory_usage(message);

    for (size_t i = 0; i < nq; i++){
        std::unordered_set<idx_t> gt;

        for (size_t j = 0; j < k; j++)
            gt.insert(groundtruth[ngt * i + j]);
        
        assert (gt.size() == k);
        for (size_t j = 0; j < k; j++){
            if (gt.count(query_labels[i * nq + j]))
                correct ++;
        }
    }

    std::cout << "The recall is: " << correct / (result_k * nq) << std::endl;
}




