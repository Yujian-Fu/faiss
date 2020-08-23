import numpy as np 
import matplotlib.pyplot as plt

dataset = "DEEP1M"
title = dataset + " / Parameter Setting - Construction Time"
filenames = ["/reasoning_models_VQ_VQ_10_50_10_50.txt", "/reasoning_models_VQ_VQ_10_50_60_100.txt", 
            "/reasoning_models_VQ_VQ_60_100_10_50.txt", "/reasoning_models_VQ_VQ_60_100_60_100.txt"]

filepaths = ["/home/yujian/Desktop/Recording_Files/VQ_VQ/" + dataset + filenames[i] for i in range(len(filenames))]
distance_ratio = 400
x_centroids = []
sum_time = []
first_layer_time = 0
assign_time = 0
second_layer_time = 0

for filepath in filepaths:
    
    file = open(filepath, "r")
    f1 = file.readlines()

    for x in f1:
        if "n centroids:" in x:
            centroids = x.split("n centroids:")[-1].split(" ")[1:3]
            x_axis = centroids[0] + " " + centroids[1]
            x_centroids.append(x_axis)

        if "Finish 1st layer clustering: The time usage:" in x:
            first_layer_time = float(x.split("Finish 1st layer clustering: The time usage:")[1].split(" ")[1])
        
        if "Finish assigning train vectorsThe time usage:" in x:
            assign_time = float(x.split("Finish assigning train vectorsThe time usage:")[1].split(" ")[1])

        if "Finish 2nd layer clusteringThe time usage:" in x:
            second_layer_time = float(x.split("Finish 2nd layer clusteringThe time usage:")[1].split(" ")[1])
            sum_time.append(first_layer_time + second_layer_time + assign_time)

            first_layer_time = 0
            second_layer_time = 0
            assign_time = 0

inds = (-np.array(sum_time)).argsort()
sorted_x_centroids = np.array(x_centroids)[inds]
sorted_sum_time = np.array(sum_time)[inds]
plt.plot(list(sorted_x_centroids), list(sorted_sum_time))

plt.xticks(rotation=60)
plt.xlabel("Centroid Setting")
plt.ylabel("Time / s")
plt.title(title)
plt.show()