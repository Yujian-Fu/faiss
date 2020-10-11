import numpy as np 
import matplotlib.pyplot as plt 

base_path = "/home/yujian/Desktop/exp_record/QE_multi_layer/"

models = ["inverted_index", "ICI", "IMI", "IVFADC", "VQTree"]

datasets = ["SIFT1M", "DEEP1M", "GIST1M"]
recall = []
time = []
index_conf = ""
recall_result = 0
for dataset in datasets:
    for model in models:
        for recall_test in [1, 10]:

            folder_path = base_path + model + "/"
            file_path = folder_path + dataset + "_parameter_tuning_" + model + "_16.txt"
            print(file_path)
            file = open(file_path, "r")
            f1 = file.readlines()
            plt.figure()
            time = []
            recall = []
            recall_result = 0

            for x in f1:
                if "Built index " in x:
                    index_conf = x.split("Built index")[1].split("The time usage")[0]
                
                if recall_result > 0 and len(x.split(" ")) > 1 and len(x.split(" ")) < 6:
                    recall.append(float(x.split(" ")[-2]))
                    time.append(float(x.split(" ")[-1].split("\n")[0]))
                
                if recall_result > 0 and len(x.split(" ")) <= 1:
                    inds = (-np.array(time)).argsort()
                    sorted_recall = np.array(recall)[inds]
                    sorted_time = np.array(time)[inds]
                    plt.plot(list(sorted_time), list(sorted_recall), label = index_conf + " recall@" + str(recall_result))
                    time = []
                    recall = []
                    recall_result = 0

                if "The result for recall = "  in x:
                    if float(x.split("The result for recall = ")[1].split("\n")[0]) == recall_test:
                        recall_result = recall_test
                    else:
                        recall_result = 0

            plt.xlim((0,10))
            plt.legend()
            plt.show()





