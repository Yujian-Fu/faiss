import numpy as np 
import matplotlib.pyplot as plt

dataset = "SIFT1M"
title = dataset + " / Centroid Setting - Construction Time"

filepaths = ["/home/yujian/Desktop/Recording_Files/VQ/" + dataset + "/reasoning.txt", "/home/yujian/Desktop/Recording_Files/VQ/" + dataset + "/reasoning_6000.txt"]
visited_centroids = []
construction_centroids = []
record_1 = False
record_2 = False
record_3 = False 
position = 0

for filepath in filepaths:
    file = open(filepath, "r")
    f1 = file.readlines()

    for x in f1:
        if "R@1 MC:" in x:
            record_1 = True
            position = 0
        
        if "R@10 MC:" in x:
            record_2 = True
            position = 0

        if "R@100 MC:" in x:
            record_3 = True
            position = 0

        if record_1:
            visited_centroid = float(x.split("Finish clustering: The time usage: ")[1].split(" ")[0])
            .append(clustering_time)
        
        if "Construction parameter: dataset: "+dataset+" train_size: 100000 n_centroids: " in x:
            centroids = float(x.split("Construction parameter: dataset: "+dataset+" train_size: 100000 n_centroids: ")[1].split(" ")[0])
            construction_centroids.append(centroids)

print(construction_centroids)
plt.plot(construction_centroids, construction_time)
plt.xlabel("Centroid Setting")
plt.ylabel("Time / s")
plt.title(title)
plt.show()


