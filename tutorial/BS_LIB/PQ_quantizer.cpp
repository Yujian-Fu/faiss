#include "PQ_quantizer.h"
#include <time.h>

/**
 * pair (distance, index)
 * distance is the sum of all sub centroid distance
 * index size: M
 **/
typedef std::pair<float, std::vector<idx_t>> dist_pair;

namespace bslib{

    struct cmp
    {
        bool operator() ( dist_pair a, dist_pair b ){return a.first > b.first;}
    };


    /**
     * The initialization function for PQ layer
     * 
     * Input: 
     * M: number of sub-spaces
     * nbits: bits for storing centroids in each subspace
     * 
     * Notice: we should keep the total number of centroids smaller than the range of int64
     * nc_upper * pow(2, nbits * M)
     * 
     **/
    PQ_quantizer::PQ_quantizer(size_t dimension, size_t nc_upper, size_t M, size_t nbits):
        Base_quantizer(dimension, nc_upper, new_pow(2, nbits*M)), M(M), nbits(nbits){

            this->ksub = new_pow(2, nbits);
            this->dsub = dimension / M;
            // Tune this parameter to modify the size of Hash Table
            this->hash_size = nc_upper * ksub;
        }

    /**
     * the pow function for ksub^pow
     **/
    idx_t PQ_quantizer::new_pow(size_t ksub, size_t pow){
        idx_t result = 1;
        for (size_t i = 0; i < pow; i++){
            result *= ksub;
        }
        return result;
    }

    /**
     * 
     * The transfer function between index and id
     * 
     * Input: 
     * index: the id for sub vector in each subspace  size: M
     * group_id: the group_id for computing base_id
     * 
     * Output:
     * id: the id for an data vector in the whole range [0, nc_upper * nc_per_group]
     * 
     **/
    idx_t PQ_quantizer::index_2_id(const idx_t * index, const idx_t group_id){
        idx_t idx = 0;
        for (size_t i = 0; i < M; i++){
            idx += index[i] * new_pow(ksub, i);
        }
        idx_t result_idx = CentroidDistributionMap[group_id] + idx; assert(result_idx < nc);
        return result_idx;
    }


    /**
     * The transfer function between id and index
     * 
     * Input: 
     * the id for one centroid in the whole set of centroids
     * 
     * Output:
     * index: the id in each subspace  size: M
     * group_id: the group of the id that belongs to
     * 
     **/
    idx_t PQ_quantizer::id_2_index(idx_t idx, idx_t * index){
        assert(idx < nc);
        size_t group_id = idx / this->nc_per_group;
        idx = idx - CentroidDistributionMap[group_id];
        for (size_t i = 0; i < M; i++){index[i] = idx % ksub;idx = idx / ksub;}
        return group_id;
    }



    /**
     * Function for constructing PQ layer
     * 
     * Input:
     * train_data:      the train data for construct centroids         size: train_set_size * dimension
     * train_set_size:  the train data set size
     * train_data_ids:  the group_id for train data                    size: train_set_size
     * 
     * No Output
     **/
    void PQ_quantizer::build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_ids){
        std::cout << "Adding " << train_set_size << " train set data into " << nc_upper << " group " << std::endl;

        std::vector<std::vector<float>> train_set(this->nc_upper);

        for (size_t i = 0; i < train_set_size; i++){
            idx_t group_id = train_data_ids[i];
            assert(group_id < nc_upper);
            for (size_t j = 0; j < dimension; j++)
                train_set[group_id].push_back(train_data[i*dimension + j]);
        }

        this->PQs.resize(nc_upper);

        std::cout << "Building group quantizers for pq_quantizer" << std::endl;
        for (size_t i = 0; i < nc_upper; i++){std::cout << train_set[i].size() / 128 << " ";} std::cout << std::endl;
#pragma omp parallel for
        for (size_t i = 0; i < nc_upper; i++){
            faiss::ProductQuantizer * product_quantizer = new faiss::ProductQuantizer(dimension, M, nbits);
            size_t nt_sub = train_set[i].size() / this->dimension;
            product_quantizer->train(nt_sub, train_set[i].data());
            this->PQs[i] = product_quantizer;
        }

        this->centroid_norms.resize(nc_upper);
        for (size_t i = 0; i < this->nc_upper; i++){
            for (size_t j = 0; j < this->M * this->ksub; j++){
                this->centroid_norms[i].push_back(faiss::fvec_norm_L2sqr(this->PQs[i]->centroids.data() + j * dsub, dsub));
            }
        }

        std::cout << "finished training PQ quantizer" <<std::endl;
    }

    /**
     * This function is to check whether an index has already been visited
     * 
     * Input:
     * visited_index:        all visited index                                size: index_num * M
     * index_check:          index to be checked                              size: M
     * index_num:            number of index that has been stored             size_t
     * 
     * Output:
     * visited:             whether the index_check is visited                bool
     * 
     **/
    bool PQ_quantizer::traversed(const idx_t * visited_index, const idx_t * index_check, const size_t index_num){
        for (size_t i = 0; i < index_num; i++){
            bool index_flag = true;
            for (size_t j = 0; j < this->M; j++){
                if (visited_index[i * M + j] != index_check[j]){
                    index_flag = false;
                    break;
                }
            }
            if (index_flag == true)
                return true;
        }
        return false;
    }


    /**
     * 
     * The algorithm for selecting and keep the keep_space smallest distances
     * 
     * Input:
     * group_id:      the group that query belongs to
     * dist_sequence: the sequence for all query-subcentroids dists          size: M * ksub
     * keep_space:    the number of final cloest neighboor       
     * 
     * Output:            
     * result_dists:  the dist to the closest neighbor centroid              keep_space
     * result_labels: the label of the cloest centroid                       keep_space
     * 
     **/
   void PQ_quantizer::multi_sequence_sort(const idx_t group_id, const float * dist_sequence, size_t keep_space, float * result_dists, idx_t * result_labels){
       
       std::vector<std::vector<float>> dist_seqs(this->M, std::vector<float>(this->ksub));
       std::vector<std::vector<idx_t>> dist_index(this->M, std::vector<idx_t>(this->ksub));

        clock_t start_t;
        double period1, period2, period3, period4;
        start_t = clock();
#pragma omp parallel for
       for (size_t i = 0; i < this->M; i++){
           uint32_t x = 0;
           //From 0 to M-1
           std::iota(dist_index[i].begin(), dist_index[i].end(), x++);

           for (size_t j = 0; j < this->ksub; j++){
               dist_seqs[i][j] = dist_sequence[i * this->ksub + j];
           }
            std::sort(dist_index[i].begin(), dist_index[i].end(), [&](int a,int b){return dist_seqs[i][a]<dist_seqs[i][b];} );
       }
        period1 = (double) (clock() - start_t);
        
        start_t = clock();
       std::priority_queue<dist_pair, std::vector<dist_pair>, cmp> dist_queue;
       std::vector<dist_pair> result_sequence(keep_space);
       std::vector<idx_t> visited_index;

       std::vector<idx_t> dist_ids(this->M, 0);
       float origin_dist_sum = 0;
       for (size_t i = 0; i < this->M; i++){origin_dist_sum += dist_seqs[i][dist_index[i][dist_ids[i]]];}
        dist_pair origin_pair(origin_dist_sum, dist_ids);

        for (size_t i = 0; i < this->M; i++){visited_index.push_back(origin_pair.second[i]);}
        dist_queue.push(origin_pair);
        result_sequence[0] = origin_pair;
        period2 = (double) (clock() - start_t);
        

        start_t = clock();
        for (size_t i = 1; i < keep_space; i++){
           dist_pair top_pair = dist_queue.top();
           dist_queue.pop();
           for (size_t j = 0; j < this->M; j++){
               //Check if there is an zero value (j)
               if (top_pair.second[j] == 0){
                   for (size_t m = 0; m < this->M; m++){
                        if (m == j || top_pair.second[m] == this->ksub - 1){
                            //keep index j (0) the same as before 
                            // skip if index m reaches the bound 
                            continue; 
                        }

                        std::vector<idx_t> new_dist_idxs(this->M, 0);
                        float new_dist_sum = 0;

                        //Initialize the index m to be inserted
                        for (size_t k = 0; k < this->M; k++){
                            //Add 1 for index m 
                            new_dist_idxs[k] = (m == k) ? top_pair.second[k] + 1 : top_pair.second[k];
                            new_dist_sum += dist_seqs[k][dist_index[k][new_dist_idxs[k]]];
                        }
                        
                        if (!traversed(visited_index.data(), new_dist_idxs.data(), visited_index.size() / M)){
                            dist_pair new_pair(new_dist_sum, new_dist_idxs);
                            for (size_t k = 0; k < this->M; k++){visited_index.push_back(new_pair.second[k]);}
                            dist_queue.push(new_pair);
                        }
                    }
               }

               //If top_pair.second[j] - 1 and top_pair.second[m] + 1 is already visited (m is another index) 
               // Add top_pair.second[j], top_pair.second[m] + 1 
               else{
                    std::vector<idx_t> new_dist_idxs(this->M, 0);
                    std::vector<idx_t> test_dist_idxs(this->M, 0);
                    float new_dist_sum = 0;
                    for (size_t m = 0; m < this->M; m++){
                        //m is the choice for another index
                        if (m == j){continue;}
                        for (size_t k = 0; k < this->M; k++){
                            test_dist_idxs[k] = (k == j) ? top_pair.second[j] - 1 :(k == m) ? top_pair.second[m] + 1 : top_pair.second[k];
                        }
                        if (traversed(visited_index.data(), test_dist_idxs.data(), visited_index.size() / this->M)){
                            for (size_t k = 0; k < this->M; k++){
                                new_dist_idxs[k] = (k == m) ? top_pair.second[m] + 1 : top_pair.second[k];
                                new_dist_sum += dist_seqs[k][dist_index[k][new_dist_idxs[k]]];
                            }
                            if (! traversed(visited_index.data(), new_dist_idxs.data(), visited_index.size() / this->M)){
                                dist_pair new_pair(new_dist_sum, new_dist_idxs);
                                for (size_t k = 0; k < this->M; k++){visited_index.push_back(new_pair.second[k]);}
                                dist_queue.push(new_pair);
                            }
                        }
                    }
                }
            }
            result_sequence[i] = dist_queue.top();
        }
        period3 = (double) (clock() - start_t);

        start_t = clock();
#pragma omp parallel for
        for (size_t i = 0; i < keep_space; i++){
            
            dist_pair new_pair = result_sequence[i];

            std::vector<idx_t> recovered_index(this->M);

            for (size_t j = 0; j < this->M; j++){
                recovered_index[j] = dist_index[j][new_pair.second[j]];
            }
            result_labels[i] = index_2_id(recovered_index.data(), group_id);
            
            result_dists[i] = new_pair.first;
        }
        period4 = (double) (clock() - start_t);

        double sum_period = period1 + period2 + period3 + period4;
        std::cout << "The time for several parts: " << period1 / sum_period << " " << period2 / sum_period << " " << period3 / sum_period << " " << period4 / sum_period;
        std::cout << std::endl;
    }

    /**
     * This is the search function for PQ layer in one group
     * 
     * Input:
     * queries:      the query data                                        size: n * dimension
     * group_ids:    the group id for query data                           size: n
     * keep space:   how many result do we want to keep                    size_t
     * 
     * Output:
     * result_dists:  the result distances between queries and centroids   size: n * keep_space
     * result_labels: the result labels for queries                        size: n * keep_space
     * 
     * 
     **/
    void PQ_quantizer::search_in_group(size_t n, const float * queries, const idx_t * group_idxs, float * result_dists, idx_t * result_labels, size_t keep_space){
//#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            idx_t group_id = group_idxs[i];
            std::vector<float> distance_table(this->M * this->ksub);
            this->PQs[group_id]->compute_distance_table(queries + i * dimension, distance_table.data());
            multi_sequence_sort(group_id, distance_table.data(), keep_space, result_dists + i * keep_space, result_labels + i * keep_space);
        }
    }


    /**
     * This is the function for search in all groups
     **/
    void PQ_quantizer::search_all(const size_t n, const float * query_data, idx_t * query_data_ids){
        std::vector<idx_t> query_group_labels(n  * nc_upper);
        std::vector<float> query_group_dists(n * nc_upper);
#pragma omp parallel for
        for (size_t i = 0; i < nc_upper; i++){
            std::vector<idx_t> query_group_idxs(n, i);
            search_in_group(n, query_data, query_group_idxs.data(), query_group_dists.data() + i * n, query_group_labels.data() + i * n, 1);
        }

        for (size_t i = 0; i < n; i++){
            float min_dis = query_group_dists[i];
            query_data_ids[i] = query_group_labels[i];

            for (size_t j = 1; j < nc_upper; j++){
                if (query_group_dists[i * j] < min_dis){
                    min_dis = query_group_dists[i * j];
                    query_data_ids[i] = query_group_labels[i * j];
                }
            }
        }
    }

    /**
     * 
     * This is the function for computing final centroid in the PQ layer
     * 
     * Input:
     * centroid_idx:    the id for target centroid        idx_t 
     * 
     * Output:
     * final_centroid:  the final out centroid required   size: dimension 
     * 
     **/
    void PQ_quantizer::compute_final_centroid(const idx_t centroid_idx, float * final_centroid){
        std::vector<idx_t> group_index(this->M);
        size_t group_id = id_2_index(centroid_idx, group_index.data());

        for (size_t i = 0; i < this->M; i++){
            for (size_t j = 0; j < this->dsub; j++){
                final_centroid[i * dsub + j] = this->PQs[group_id]->centroids[i * ksub * dsub + group_index[i] * dsub + j];
            }
        }
    }


    /**
     * Function for computing residuals
     * 
     * Input: centroid_idxs:
     *  
     * Output: residuals:
     * 
     **/ 
    void PQ_quantizer::compute_residual_group_id(size_t n, const idx_t * centroid_idxs, const float * x, float * residuals){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(this->dimension);
            compute_final_centroid(centroid_idxs[i], final_centroid.data());
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, final_centroid.data(), residuals + i * dimension);
        }
    }


    void PQ_quantizer::recover_residual_group_id(size_t n, const idx_t * centroid_idxs, const float * residuals, float * x){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            compute_final_centroid(centroid_idxs[i], final_centroid.data());
            faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, final_centroid.data(), x + i * dimension);
        }
    }

    
    float PQ_quantizer::get_centroid_norms(const idx_t centroid_idx){
        float result = 0;
        std::vector<idx_t> index(this->M);
        idx_t group_id = id_2_index(centroid_idx, index.data());
        for (size_t i = 0; i < this->M; i++){
            result += this->centroid_norms[group_id][i * ksub + index[i]];
        }
        return result;
    }
}