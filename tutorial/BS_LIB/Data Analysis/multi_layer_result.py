import numpy as np 
import matplotlib.pyplot as plt 

recall = []
time = []
iteration = 0
recall_target = 1
recall_record = 0

filepath = "/home/yujian/Desktop/extra/Similarity Search/similarity_search_datasets/models_VQ/SIFT1M/parameter_tuning_inverter_index_16.txt"
file = open(filepath, "r")
f1 = file.readlines()
for x in f1:
    if "This is the record for Inverted Index with record 1000 centroids" in x:
        iteration = float(x.split("This is the record for Inverted Index with record 1000 centroids")[1])
    
    if recall_record > 0 and len(x.split(" ")) <= 1:
        print(time)
        print(recall)
        time_inds = (-np.array(time)).argsort()
        recall_inds = (-np.array(recall)).argsort()

        sorted_recall = np.array(recall)[recall_inds]
        sorted_time = np.array(time)[time_inds]

        plt.plot(list(sorted_time), list(sorted_recall), label = str(int(iteration)))
        recall_record = 0
        recall = []
        time = []
    
    if recall_record > 0 and len(x.split(" ")) > 1:

        recall.append(float(x.split(" ")[-2]))
        time.append(float(x.split(" ")[-1].split("\n")[0]))

    if "The result for recall = " in x:
        if float(x.split(" ")[-1]) == recall_target:
            recall_record = recall_target;
    

plt.xlabel("Time / ms")
plt.ylabel("Recall@1")
plt.legend()
plt.show()





