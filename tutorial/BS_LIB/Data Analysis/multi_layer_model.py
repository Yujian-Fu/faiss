import numpy as np 
import matplotlib.pyplot as plt 

base_path = "/home/yujian/Desktop/exp_record/QE_multi_layer/"

models = ["inverted_index", "ICI", "IMI", "IVFADC", "VQTree"]

datasets = ["SIFT10K", "SIFT1M", "DEEP1M", "GIST1M"]
recall = []
time = []
index_conf = ""
recall_result = 0
all_colors = [
#'aliceblue',
#'antiquewhite',
'aqua',
'aquamarine',
#'azure',
#'beige',
#'bisque',
'black',
#'blanchedalmond',
'blue',
'blueviolet',
'brown',
'burlywood',
'cadetblue',
'chartreuse',
'chocolate',
'coral',
'cornflowerblue',
#'cornsilk',
'crimson',
#'cyan',
'darkblue',
'darkcyan',
'darkgoldenrod',
'darkgray',
'darkgreen']

all_types = [
'-',
'--',
'-.',
':' 
]

all_nodes = [
'.',
',',
'o',
'v', 
'^'
]


color_length = len(all_colors)

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
            color_index = 0

            for x in f1:
                if "Built index " in x:
                    index_conf = x.split("Built index")[1].split("The time usage")[0]
                
                if "This is the record for IMI with nbits =  " in x:
                    index_conf = x.split("This is the record for IMI with nbits =  ")[1]

                if recall_result > 0 and len(x.split(" ")) > 1 and len(x.split(" ")) < 6:
                    recall.append(float(x.split(" ")[-2]))
                    time.append(float(x.split(" ")[-1].split("\n")[0]))
                
                if recall_result > 0 and len(x.split(" ")) <= 1:
                    inds_time = (-np.array(time)).argsort()
                    sorted_time = np.array(time)[inds_time]
                    inds_recall = (-np.array(recall)).argsort()
                    sorted_recall = np.array(recall)[inds_recall]

                    plt.plot(list(sorted_time), list(sorted_recall), label = index_conf, color = all_colors[color_index % color_length], linestyle = all_types[int(color_index / color_length)%4], marker = all_nodes[int(color_index / (color_length * 4))%4])
                    color_index += 1
                    print(color_index, color_length)
                    time = []
                    recall = []
                    recall_result = 0

                if "The result for recall = "  in x:
                    if float(x.split("The result for recall = ")[1].split("\n")[0]) == recall_test:
                        recall_result = recall_test
                    else:
                        recall_result = 0

            plt.xlim()
            plt.title(dataset + " " + model)
            plt.ylabel("Recall@" + str(recall_test))
            plt.xlabel("Time / ms")
            plt.legend(prop={'size':5})
            plt.show()





