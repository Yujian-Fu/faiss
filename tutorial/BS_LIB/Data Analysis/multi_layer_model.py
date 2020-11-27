import numpy as np 
import matplotlib.pyplot as plt 



base_path = "/home/yujian/Desktop/extra/Similarity_Search/similarity_search_datasets/models_VQ/"

#base_path = "/home/yujian/Desktop/exp_record/models_VQ/"
models = ["inverted_index"]

datasets = ["Random100K_128"]
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
#'--',
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

convergence_gap = 0.01

def get_best_point(sorted_time, sorted_recall):
    best_recall = 0
    for i in range(len(sorted_recall)):
        if (sorted_recall[-1] - best_recall < convergence_gap):
            return (sorted_time[i], sorted_recall[i])
        else:
            best_recall = sorted_recall[i]
    print(sorted_time)
    print(sorted_recall)

    return (sorted_time[-1], sorted_recall[-1])


color_length = len(all_colors)
type_length = len(all_types)

for dataset in datasets:
    for model in models:
        for recall_test in [1, 10]:
            fig,ax=plt.subplots(1,2)

            folder_path = base_path + dataset + "/"
            file_path = folder_path + "parameter_tuning_" + model + "_16_20201115-211523.txt"
            print(file_path)
            file = open(file_path, "r")
            f1 = file.readlines()
            time = []
            recall = []
            recall_result = 0
            color_index = 0

            for x in f1:
                if "Built index " in x:
                    index_conf = x.split("Built index")[1].split("The time usage")[0]
                
                if "This is the record for IMI with nbits =  " in x:
                    index_conf = x.split("This is the record for IMI with nbits =  ")[1]

                if recall_result > 0 and len(x.split(" ")) > 1 and (len(x.split(" ")) < 6 or "Recall" in x):
                    recall.append(float(x.split(" ")[-2]))
                    time.append(float(x.split(" ")[-1].split("\n")[0]))
                
                if recall_result > 0 and len(x.split(" ")) <= 1:
                    if (len(recall) > 0):
                        inds_time = (-np.array(time)).argsort()
                        sorted_time = np.array(time)[inds_time]
                        inds_recall = (-np.array(recall)).argsort()
                        sorted_recall = np.array(recall)[inds_recall]

                        
                        ax[0].plot(list(sorted_time), list(sorted_recall), label = index_conf, color = all_colors[color_index % color_length], linestyle = all_types[int(color_index / color_length)%type_length] , marker = all_nodes[int(color_index / (color_length * type_length))%4])
                        best_point = get_best_point(list(sorted_time), list(sorted_recall))
                        target = ""
                        if (str(target)+" " in index_conf or " "+ str(target) in index_conf):
                            ax[1].plot(best_point[0], best_point[1])
                            ax[1].text(best_point[0], best_point[1], index_conf, fontsize=8)
                            color_index += 1
                            print(color_index, color_index % color_length , int(color_index / color_length)%type_length, int(color_index / (color_length * type_length))%4)

                    time = []
                    recall = []
                    recall_result = 0

                if "The result for recall = "  in x:
                    if float(x.split("The result for recall = ")[1].split("\n")[0]) == recall_test:
                        recall_result = recall_test
                    else:
                        recall_result = 0

            ax[0].set_title(dataset + " " + model)
            ax[0].set_ylabel("Recall@" + str(recall_test))
            ax[0].set_xlabel("Time / ms")
            ax[0].legend(prop={'size':7}, ncol = 3)
            #plt.title(dataset + " " + model)
            #plt.ylabel("Recall@" + str(recall_test))
            #plt.xlabel("Time / ms")
            #plt.legend(prop={'size':7}, ncol = 3)
            plt.show()





