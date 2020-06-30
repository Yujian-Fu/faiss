#include "PQ_quantizer.h"

typedef std::pair<float, std::vector<idx_t>> dist_pair;

namespace bslib{

    struct cmp
    {
        bool operator() ( dist_pair a, dist_pair b ){return a.first > b.first;}
    };
    

    PQ_quantizer::PQ_quantizer(size_t dimension, size_t nc_upper, size_t M, size_t nbits):
        Base_quantizer(dimension, nc_upper, pow(2, nbits*M)), M(M), nbits(nbits){
            this->ksub = pow(2, nbits);
            this->dsub = dimension / M;
        }


    void PQ_quantizer::build_centroids(const float * train_data, size_t train_set_size, idx_t * train_data_idxs, bool update_idxs){

        std::cout << "Adding " << train_set_size << " train set data into " << nc_upper << " group " << std::endl;
        PQs.resize(this->nc_upper);
        
        std::vector<std::vector<float>> train_set(this->nc_upper);

        for (size_t i = 0; i < train_set_size; i++){
            idx_t idx = train_data_idxs[i];
            for (size_t j = 0; j < dimension; j++)
                train_set[idx].push_back(train_data[i*dimension + j]);
        }


        std::cout << "Building PQ quantizers" << std::endl;
#pragma omp parallel for
        for (size_t i = 0; i < nc_upper; i++){
            faiss::ProductQuantizer product_quantizer(dimension, M, nbits);
            size_t nt_sub = train_set[i].size() / this->dimension;
            product_quantizer.train(nt_sub, train_set[i].data());
            this->PQs[i] = product_quantizer;
        }

        std::cout << "finished training PQ quantizer" <<std::endl;
    }


    /*
    The return size of result_dists: keep_space * sizeof(float)
    The return size for result_labels: keep_space * sizeof(idx_t) (aassert the nc_per_group is smaller than the valid range of long)
    */


   void PQ_quantizer::multi_sequence_sort(const float * dist_sequence, size_t keep_space, float * result_dists, idx_t * result_labels){
       std::vector<std::vector<float>> dist_seqs(this->M, std::vector<float>(this->ksub));
        std::vector<std::vector<idx_t>> dist_index(this->M, std::vector<idx_t>(this->ksub));

       for (size_t i = 0; i < this->M; i++){
           uint32_t x = 0;
           std::iota(dist_index[i].begin(), dist_index[i].end(), x++);

           for (size_t j = 0; j < this->ksub; j++){
               dist_seqs[i][j] = dist_sequence[i * this->ksub + j];
           }
            std::sort(dist_index[i].begin(), dist_index[i].end(), [&](int a,int b){return dist_seqs[i][a]<dist_seqs[i][b];} );
       }

       std::priority_queue<dist_pair, std::vector<dist_pair>, cmp> dist_queue;
       
       std::vector<idx_t> dist_idxs(this->M, 0);
       float dist_sum = 0;
       for (size_t i = 0; i < this->M; i++){
           dist_sum += dist_seqs[i][dist_index[i][dist_idxs[i]]];
       }
        dist_pair origin_pair(dist_sum, dist_idxs);
        dist_queue.push(origin_pair);

        std::vector<bool> visited_flag(this->nc_per_group, false);
        visited_flag[0] = true;
        //std::vector< std::vector<bool> > visited_flag(this->M, std::vector<bool>(this->ksub, false));
        //visited_flag[0][0] = true;

       for (size_t i = 0; i < keep_space; i++){
           dist_pair top_pair = dist_queue.top();
           dist_queue.pop();
           for (size_t j = 0; j < this->M; j++){

               if (top_pair.second[j] == 0){
                   std::vector<idx_t> dist_idxs(this->M, 0);
                   float dist_sum = 0;
                   idx_t new_id = 0;

                   for (size_t k = 0; k < this->M; k++){
                       dist_idxs[k] = (j == k) ? 1 : top_pair.second[j];
                       dist_sum += dist_seqs[k][dist_index[k][dist_idxs[k]]];
                       new_id += dist_idxs[k] * this->ksub ;
                   }
                   dist_pair new_pair(dist_sum, dist_idxs);
                   dist_queue.push(new_pair);
               }
               else{
                   for (size_t )
               }
           }


           
       }
   }

    void PQ_quantizer::search_in_group(size_t n, const float * queries, const idx_t * group_idxs, float * result_dists, uint8_t * result_labels, size_t keep_space){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            idx_t idx = group_idxs[i];
            std::vector<float> distance_table(this->M * this->ksub);
            this->PQs[idx].compute_distance_table(queries + i * dimension, distance_table.data());
            
            for (size_t j = 0; j < this->nc_per_group; j++){
                float dist = 0;
                for (size_t m = 0; m < M; m++){
                    dist += distance_table[m * this->ksub + j % (this->ksub>>(M-(m+1)))];
                }
                result_dists[i * this->nc_per_group + j] = dist;
            }
        }
    }


    void PQ_quantizer::compute_final_centroid(idx_t group_idx, idx_t group_label, float * final_centroid){
        for (size_t i = 0; i < this->M; i++){
            idx_t sub_idx = group_label % (this->ksub>>(M-(i+1)));
            for (size_t j = 0; j < this->dsub; j++){
                final_centroid[i * dsub + j] = this->PQs[j].centroids[i * ksub * dsub + sub_idx * dsub + j];
            }
        }
    }


    void PQ_quantizer::compute_residual_group_id(size_t n, const idx_t * group_idxs, const idx_t * group_labels, const float * x, float * residuals){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(this->dimension);
            compute_final_centroid(group_idxs[i], group_labels[i], final_centroid.data());
            faiss::fvec_madd(dimension, x + i * dimension, -1.0, final_centroid.data(), residuals + i * dimension);
        }
    }

    void PQ_quantizer::recover_residual_group_id(size_t n, const idx_t * group_idxs, const idx_t * group_labels, const float * residuals, float * x){
#pragma omp parallel for
        for (size_t i = 0; i < n; i++){
            std::vector<float> final_centroid(dimension);
            compute_final_centroid(group_idxs[i], group_labels[i], final_centroid.data());
            faiss::fvec_madd(dimension, residuals + i * dimension, 1.0, final_centroid.data(), x + i * dimension);
        }
    }



}