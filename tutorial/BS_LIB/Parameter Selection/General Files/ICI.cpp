
#include <fstream>
#include <unordered_set>
#include <sys/resource.h>
#include <string>

#include "../../bslib_index.h"
#include "../../parameters/parameter_tuning/ICI/ICI.h"

using namespace bslib;

int main(int argc, char * argv[]){
    assert(argc == 4);
    size_t centroid1 = atoi(argv[1]);
    size_t cons_nbits = atoi(argv[2]);
    size_t build_times = atoi(argv[3]);

    std::string message;
    std::ofstream record_file;
    PrepareFolder((char *) folder_model.c_str());
    PrepareFolder((char *) (std::string(folder_model)+"/" + dataset).c_str());

   // 基于当前系统的当前日期/时间
   time_t now = time(0);
   
   // 把 now 转换为字符串形式
    char* dt = ctime(&now);
 
    path_record += "parameter_tuning_ICI" + std::to_string(M_PQ) + "_" + std::string(dt) + ".txt";
    if (build_times == 0){
        record_file.open(path_record, std::ios::binary);
    }
    else{
        record_file.open(path_record, std::ios::app);
    }
    
    record_file << "This is the record file for ICI with centroids and nbits: " << centroid1 << " and " << cons_nbits << std::endl;

    memory_recorder Mrecorder = memory_recorder();
    time_recorder Trecorder = time_recorder();
    //record_file << "The result for VQ Tree: " + std::to_string(centroid1) + " " + std::to_string(cons_nbits) << std::endl;
    
    Trecorder.reset();
    Bslib_Index index = Bslib_Index(dimension, layers, index_type, use_HNSW_VQ, use_norm_quantization, use_OPQ);
    index.train_size = train_size;
    index.M = M_PQ;
    index.nbits = nbits;
    index.use_train_selector = use_train_selector;
    index.use_reranking = use_reranking;
    index.save_index = false;
    
    // This is the part for using components or not
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
            message = "Trained the OPQ matrix, ";
            Mrecorder.print_memory_usage(message);
            Mrecorder.record_memory_usage(record_file,  message);
            Trecorder.print_time_usage(message);
            Trecorder.record_time_usage(record_file, message);
        }
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
    M_PQ_layer[0] = 2;
    nbits_PQ_layer[0] = cons_nbits;
    for (size_t i = 0; i < PQ_layers; i++){
        PQ_para new_para; new_para.first = M_PQ_layer[i]; new_para.second = nbits_PQ_layer[i];
        PQ_paras.push_back(new_para);
    }

    ncentroids[0] = centroid1;
    
    index.build_quantizers(ncentroids, path_quantizers, path_learn, num_train, HNSW_paras, PQ_paras);
    index.get_final_group_num();
    message = "Built index " + std::to_string(centroid1) + " " + std::to_string(cons_nbits);
    Trecorder.record_time_usage(record_file, message);


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

    message = "Assigned the index ";
    Trecorder.print_time_usage(message);

    for (size_t i = 0; i < nbatches; i++){
        base_output.write((char *) & batch_size, sizeof(uint32_t));
        base_output.write((char *) assigned_ids.data() + i * batch_size * sizeof(idx_t), batch_size * sizeof(idx_t));
    }
    base_output.close();
    message = "Assigned the base vectors in parallel mode ";
    Trecorder.record_time_usage(record_file, message);

    index.train_pq(path_pq, path_pq_norm, path_learn, PQ_train_size);
    message = "Trained the PQ ";
    Trecorder.record_time_usage(record_file, message);

    std::vector<idx_t> ids(nb); std::ifstream ids_input(path_ids, std::ios::binary);
    readXvec<idx_t> (ids_input, ids.data(), batch_size, nbatches);
    std::vector<size_t> groups_size(index.final_group_num, 0); std::vector<size_t> group_position(nb, 0);
    for (size_t i = 0; i < nb; i++){group_position[i] = groups_size[ids[i]]; groups_size[ids[i]] ++;}

    index.base_codes.resize(index.final_group_num);
    index.base_sequence_ids.resize(index.final_group_num);
    if (use_norm_quantization){index.base_norm_codes.resize(nb);} else{index.base_norms.resize(nb);}
    for (size_t i = 0; i < index.final_group_num; i++){
        index.base_codes[i].resize(groups_size[i] * index.code_size);
        index.base_sequence_ids[i].resize(groups_size[i]);
    }

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
    index.compute_centroid_norm();
    message = "Finished Construction: ";
    Trecorder.print_time_usage(message);
    Mrecorder.record_memory_usage(record_file, message);
    Trecorder.record_time_usage(record_file, message);

    std::vector<uint32_t> groundtruth(nq * ngt);
    {
        std::ifstream gt_input(path_gt, std::ios::binary);
        readXvec<uint32_t>(gt_input, groundtruth.data(), ngt, nq, false, false);
    }

    std::vector<float> queries(nq * dimension);
    {
        std::ifstream query_input(path_query, std::ios::binary);
        readXvecFvec<base_data_type>(query_input, queries.data(), dimension, nq, false, false);
    }


    for (size_t i = 0; i < num_recall; i++){
        size_t recall_k = result_k[i];
        std::vector<float> query_distances(nq * recall_k);
        std::vector<faiss::Index::idx_t> query_labels(nq * recall_k);
        size_t c1_start = 2;
        size_t c1_end = centroid1;
        size_t search_step1 = 4;

        size_t nbits_search_space = 1;
        for (size_t temp = 0; temp < cons_nbits * 2; temp++){
            nbits_search_space *= 2;
        }
        size_t c2_start = 4;
        size_t c2_end = nbits_search_space;
        size_t search_step2 = cons_nbits;
        record_file << "The result for recall = " << recall_k << std::endl;

        float c1_previous_recall = 0;
        float c1_second_previous_recall = 0;
        float c2_previous_recall = 0;
        float c2_second_previous_recall = 0;

        size_t search_space1 = c1_start;
        size_t search_space2 = c2_start;

        while(search_space1 < c1_end && search_space2 < c2_end){
            std::vector<size_t> search_space(2);
            search_space[0] = search_space1;
            search_space[1] = search_space2;
            index.max_visited_vectors = nb / index.final_group_num * (search_space[0] * search_space[1] * 1.5);
            size_t correct = 0;
            std::cout << search_space1 << " " << search_space2 << " ";
            Trecorder.reset();
            index.search(nq, recall_k, queries.data(), query_distances.data(), query_labels.data(), search_space.data(), groundtruth.data(), path_base);
            float qps = Trecorder.getTimeConsumption() / (nq * 1000);

            for (size_t temp1 = 0; temp1 < nq; temp1++){
                std::unordered_set<idx_t> gt;
                for (size_t temp2 = 0; temp2 < recall_k; temp2 ++){
                    gt.insert(groundtruth[ngt * temp1 + temp2]);
                }

                assert(gt.size() == recall_k);
                for (size_t temp2 = 0; temp2 < recall_k; temp2 ++){
                    if (gt.count(query_labels[temp1 * recall_k + temp2]) != 0)
                        correct ++;
                }
            }
            float recall = float(correct) / (recall_k * nq);

            record_file << search_space1 << " " << search_space2 << " " << recall << " " << qps << std::endl;
            std::cout << "The Recall and QPS " << recall << " " << qps << std::endl;
            if (recall <= c2_previous_recall && recall <= c2_second_previous_recall){
                if (recall <= c1_previous_recall && recall <= c1_second_previous_recall){
                    break;
                }
                c2_previous_recall = 0; c2_second_previous_recall = 0;
                c1_second_previous_recall = c1_previous_recall;
                c1_previous_recall = recall;
                search_space2 = c2_start;
                search_space1 += search_step1;
            }
            else{
                c2_second_previous_recall = c2_previous_recall;
                c2_previous_recall = recall;
                search_space2 += search_step2;
            }
        }
        record_file << std::endl;
    }
    record_file.close();
}