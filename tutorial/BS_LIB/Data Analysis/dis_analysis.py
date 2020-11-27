import numpy as np
import pandas
import matplotlib.pyplot as plt


base_path = "/home/yujian/Desktop/extra/Similarity_Search/similarity_search_datasets/models_VQ/"

models = ["inverted_index"]

datasets = ["SIFT10K"]



class centroid_analysis:
    def __init__(self, dataset_name, model_name, dataset_size, query_size):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.dataset_size = dataset_size
        self.query_size = query_size
        self.nc = []
        self.b_c_dis_lists = []
        self.avg_b_c_dis_list = []
        self.b_res_lists = []
        self.avg_b_res_list = []
        self.c_res_lists = []
        self.avg_c_res_list = []
        self.com_diff_norm_lists = []
        self.avg_com_diff_norm_list = []
        self.b_com_diff_prod_lists = []
        self.avg_b_com_diff_prod_list = []
        self.c_com_diff_prod_lists = []
        self.avg_c_com_diff_prod_list = []


        self.q_gt_1 = []
        self.q_gt_res_1 = []
        self.q_gt_diff_1 = []
        self.q_gt_dis_1 = []

        self.q_label_1 = []
        self.q_label_res_1 = []
        self.q_label_diff_1 = []
        self.q_label_dis_1 = []


        self.q_gt_10 = []
        self.q_gt_res_10 = []
        self.q_gt_diff_10 = []
        self.q_gt_dis_10 = []

        self.q_label_10 = []
        self.q_label_res_10 = []
        self.q_label_diff_10 = []
        self.q_label_dis_10 = []
    

    def check(self):
        num_conf = len(self.nc)
        assert(len(self.b_c_dis_lists) == num_conf)
        assert(len(self.b_c_dis_lists[0]) == self.dataset_size)
        assert(len(self.b_res_lists) == num_conf)
        assert(len(self.b_res_lists[0]) == self.dataset_size)
        assert(len(self.c_res_lists) == num_conf)
        assert(len(self.c_res_lists[0]) == self.dataset_size)
        assert(len(self.q_gt_1) == num_conf)
        assert(len(self.q_gt_1[0]) == self.query_size)


def read_and_process():
    result_list = []
    for dataset in datasets:
        for model in models:
            dataset_size = 0
            if "K" in dataset:
                dataset_size = int(dataset.split("K")[0].split("SIFT")[1]) * 1000
            elif "M" in dataset:
                dataset_size = int(dataset.split("M")[0].split("SIFT")[1]) * 1000000

            new_record = centroid_analysis(dataset, model, dataset_size, 100)

            folder_path = base_path + dataset + "/"
            file_path = folder_path + "dist_NN_record_20201116-105747.txt"
            print(file_path)
            file = open(file_path)
            f1 = file.readlines()
            index_pos = 0
            query_state_1 = False
            query_state_10 = False

            for x in f1:

                index_pos += 1
                if "The record for index with centroid: " in x:
                    new_record.nc.append(int(x.split("The record for index with centroid: ")[1].split("\n")[0]))
                    index_pos = 0
                    query_state_1 = False
                    query_state_10 = False

                if "The b_c_dis of each base vector: " in x:
                    index_pos = 10

                if index_pos == 11:
                    new_record.b_c_dis_lists.append([float(value) for value in x.split(" ")[0:-1]])
                    index_pos = 0

                if "The average distance between base vectors and centroids: " in x:
                    new_record.avg_b_c_dis_list.append(float(x.split("The average distance between base vectors and centroids: ")[1].split("\n")[0]))
                    index_pos = 20

                if index_pos == 22:
                    new_record.b_res_lists.append([float(value) for value in x.split(" ")[0:-1]])
                    index_pos = 0
                
                if "The avg b_res_prod: " in x:
                    new_record.avg_b_res_list.append(float(x.split("The avg b_res_prod: ")[1].split("\n")[0]))
                    index_pos = 30
                
                if index_pos == 32:
                    new_record.c_res_lists.append([float(value) for value in x.split(" ")[0:-1]])
                    index_pos = 0
                
                if "The avg c_res_prod: " in x:
                    new_record.avg_c_res_list.append(float(x.split("The avg c_res_prod: ")[1].split("\n")[0]))
                    index_pos = 0
                
                if "The com_diff_norm for each base vector: " in x:
                    index_pos = 40
                
                if index_pos == 41:
                    new_record.com_diff_norm_lists.append([float(value) for value in x.split(" ")[0:-1]])
                
                if "The avg com_diff_norm " in x:
                    new_record.avg_com_diff_norm_list.append(float(x.split("The avg com_diff_norm ")[1].split("\n")[0]))
                    index_pos = 50
                
                if index_pos == 52:
                    new_record.b_com_diff_prod_lists.append([float(value) for value in x.split(" ")[0:-1]])

                if "The avg b_com_diff_prod: " in x:
                    new_record.avg_b_com_diff_prod_list.append(float(x.split("The avg b_com_diff_prod: ")[1].split("\n")[0]))
                    index_pos = 60
                
                if index_pos == 62:
                    new_record.c_com_diff_prod_lists.append([float(value) for value in x.split(" ")[0:-1]])

                if "The avg c_com_diff_prod: " in x:
                    new_record.avg_c_com_diff_prod_list.append(float(x.split("The avg c_com_diff_prod: ")[1].split("\n")[0]))
                

                if "The gt record for recall =  1" in x and "The gt record for recall =  10" not in x:
                    new_record.q_gt_1.append([])
                    new_record.q_gt_res_1.append([])
                    new_record.q_gt_diff_1.append([])
                    new_record.q_gt_dis_1.append([])
                    new_record.q_label_1.append([])
                    new_record.q_label_res_1.append([])
                    new_record.q_label_diff_1.append([])
                    new_record.q_label_dis_1.append([])
                    query_state_1 = True
                    index_pos = 0
                
                if "The gt record for recall =  10" in x:
                    new_record.q_gt_10.append([])
                    new_record.q_gt_res_10.append([])
                    new_record.q_gt_diff_10.append([])
                    new_record.q_gt_dis_10.append([])

                    new_record.q_label_10.append([])
                    new_record.q_label_res_10.append([])
                    new_record.q_label_diff_10.append([])
                    new_record.q_label_dis_10.append([])
                    
                    query_state_10 = True
                    query_state_1 = False
                    index_pos = 0


                if query_state_1 or query_state_10:
                    index_pos = 0
                    if "GT: " in x:
                        gt_record_list = []
                        gt_dis_list = []
                        gt_res_record_list = []
                        gt_diff_record_list = []


                        gt_list = x.split(" ")[1:-1]
                        for i in range(int(len(gt_list) / 4)):
                            gt_record_list.append(int(gt_list[4 * i]))
                            gt_dis_list.append(float(gt_list[4 * i + 1]))
                            gt_res_record_list.append(float(gt_list[4 * i + 2]))
                            gt_diff_record_list.append(float(gt_list[4 * i + 3]))
                        if query_state_1:
                            new_record.q_gt_1[-1].append(gt_record_list)
                            new_record.q_gt_dis_1[-1].append(gt_dis_list)
                            new_record.q_gt_res_1[-1].append(gt_res_record_list)
                            new_record.q_gt_diff_1[-1].append(gt_diff_record_list)
                        else:
                            new_record.q_gt_10[-1].append(gt_record_list)
                            new_record.q_gt_dis_10[-1].append(gt_dis_list)
                            new_record.q_gt_res_10[-1].append(gt_res_record_list)
                            new_record.q_gt_diff_10[-1].append(gt_diff_record_list)

                    elif "Search: " in x:
                        search_record_list = []
                        search_dis_list = []
                        search_res_record_list = []
                        search_diff_record_list = []

                        search_list = x.split(" ")[1:-1]
                        for i in range(int(len(gt_list) / 4)):
                            search_record_list.append(int(search_list[4 * i]))
                            search_dis_list.append(float(search_list[4 * i + 1]))
                            search_res_record_list.append(float(search_list[4 * i + 2]))
                            search_diff_record_list.append(float(search_list[4 * i + 3]))
                        if query_state_1:
                            new_record.q_label_1[-1].append(search_record_list)
                            new_record.q_label_dis_1[-1].append(search_dis_list)
                            new_record.q_label_res_1[-1].append(search_res_record_list)
                            new_record.q_label_diff_1[-1].append(search_diff_record_list)
                        else:
                            new_record.q_label_10[-1].append(search_record_list)
                            new_record.q_label_dis_10[-1].append(search_dis_list)
                            new_record.q_label_res_10[-1].append(search_res_record_list)
                            new_record.q_label_diff_10[-1].append(search_diff_record_list)

            result_list.append(new_record)
    return result_list


def compare(rl, compare1, compare2):
    print("Compare ", compare2, " from ", compare1)
    compare1 = rl.nc.index(compare1)
    compare2 = rl.nc.index(compare2)
    print("The recall@1: ")
    for i in range(rl.query_size):
        print("Query ", i, "misses: ")
        for j in range(len(rl.q_label_1[compare1][i])):
            if rl.q_label_1[compare1][i][j] in rl.q_gt_1[compare1][i] and rl.q_label_1[compare1][i][j] not in rl.q_label_1[compare2][i]:
                base_id = rl.q_label_1[compare1][i][j]
                pos = rl.q_gt_1[compare1][i].index(rl.q_label_1[compare1][i][j])
                print(base_id, 
                "q_dist: ", rl.q_label_dis_1[compare1][i][j], rl.q_gt_dis_1[compare2][i][pos],
                "q_res_prod: ", rl.q_label_res_1[compare1][i][j], rl.q_gt_res_1[compare2][i][pos], 
                "q_diff_prod: ", rl.q_label_diff_1[compare1][i][j], rl.q_label_diff_1[compare2][i][pos],
                "b_res_prod", rl.b_res_lists[compare1][base_id], rl.b_res_lists[compare2][base_id], 
                "c_res_prod", rl.c_res_lists[compare1][base_id], rl.c_res_lists[compare2][base_id],
                "b_c_dis:", rl.b_c_dis_lists[compare1][base_id], rl.b_c_dis_lists[compare2][base_id]
                )
    
def divide(rl, compare, test_recall):
    print("The analysis of index ", compare)
    compare = rl.nc.index(compare)

    for i in range(rl.query_size):
        correct_list = []
        mis_list = []
        gt_list = []
        label_list = []
        if test_recall == 1:
            gt_list = rl.q_gt_1
            label_list = rl.q_label_1
        else:
            gt_list = rl.q_gt_10
            label_list = rl.q_label_10

        for j in range(len(gt_list[compare][i])):
            if gt_list[compare][i][j] in label_list[compare][i]:
                correct_list.append(j)
            else:
                mis_list.append(j)
        if len(correct_list) > 0:
            print("Correct List: ")
        mean_b_c_dis = 0
        mean_b_res = 0
        mean_c_res = 0
        mean_b_com = 0
        for m in range(len(correct_list)):
            base_id = gt_list[compare][i][correct_list[m]]
            print(base_id, rl.b_c_dis_lists[compare][base_id], rl.b_res_lists[compare][base_id], rl.c_res_lists[compare][base_id], rl.b_com_diff_prod_lists[compare][base_id])
            mean_b_c_dis += rl.b_c_dis_lists[compare][base_id]
            mean_b_res += abs(rl.b_res_lists[compare][base_id])
            mean_c_res += abs(rl.c_res_lists[compare][base_id])
            mean_b_com += abs(rl.b_com_diff_prod_lists[compare][base_id])
        if len(correct_list) > 0:
            mean_b_c_dis /= len(correct_list)
            mean_b_res /= len(correct_list)
            mean_c_res /= len(correct_list)
            mean_b_com /= len(correct_list)

            print("Correct Average: ")
            print("   ", round(mean_b_c_dis, 2), round(mean_b_res,2), round(mean_c_res, 2), round(mean_b_com, 2))


        if len(mis_list) > 0:
            print("Mis List: ")
        mean_b_c_dis_ = 0
        mean_b_res_ = 0
        mean_c_res_ = 0
        mean_b_com_ = 0
        for m in range(len(mis_list)):
            base_id = gt_list[compare][i][mis_list[m]]
            print(base_id, rl.b_c_dis_lists[compare][base_id], rl.b_res_lists[compare][base_id], rl.c_res_lists[compare][base_id], rl.b_com_diff_prod_lists[compare][base_id])
            mean_b_c_dis_ += rl.b_c_dis_lists[compare][base_id]
            mean_b_res_ += abs(rl.b_res_lists[compare][base_id])
            mean_c_res_ += abs(rl.c_res_lists[compare][base_id])
            mean_b_com_ += abs(rl.b_com_diff_prod_lists[compare][base_id])
        if len(mis_list) > 0:
            mean_b_c_dis_ /= len(mis_list)
            mean_b_res_ /= len(mis_list)
            mean_c_res_ /= len(mis_list)
            mean_b_com_ /= len(mis_list)
            print("Mis Average: ")
            print("   ", round(mean_b_c_dis_, 2), round(mean_b_res_,2), round(mean_c_res_, 2), round(mean_b_com_, 2))
            print("   ", round(mean_b_c_dis_-mean_b_c_dis, 2), round(mean_b_res_-mean_b_res, 2), round(mean_c_res_-mean_c_res, 2), round(mean_b_com_-mean_b_com, 2))

if __name__ == "__main__":
    rl = read_and_process()[0]
    #rl.check()
    compare(rl, 11, 10)
    #divide(rl, 11, 10)
    
    std = []
    mean = []
    for i in range(len(rl.nc)):
        plt.figure()
        plt.hist(rl.b_c_dis_lists[i], bins = 40, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.xlabel("Residual Norm")
        plt.ylabel("Num")
        plt.title("Centroid: " + str(rl.nc[i]) + " Avg: " + str(round(np.mean(rl.b_c_dis_lists[i]), 1)) + " std: " + str(round(np.std(rl.b_c_dis_lists[i]), 1)))
        plt.show()



    


            






