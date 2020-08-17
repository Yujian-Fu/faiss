import numpy as np 
import matplotlib.pyplot as plt

dataset = "DEEP1M"
title = dataset + " / Centroid Setting - Construction Time"

filepaths = ["/home/yujian/Desktop/Recording_Files/VQ/" + dataset + "/reasoning.txt", "/home/yujian/Desktop/Recording_Files/VQ/" + dataset + "/reasoning_6000.txt"]
visited_centroids = []
eighty_visited_centoids = []
construction_centroids = []
record_1 = False
record_2 = False
record_3 = False 

for filepath in filepaths:
    file = open(filepath, "r")
    f1 = file.readlines()

    for x in f1:
        if "R@1 MC:" in x:
            


        if "Finish clustering: The time usage: " in x:
            clustering_time = float(x.split("Finish clustering: The time usage: ")[1].split(" ")[0])
            construction_time.append(clustering_time)
        
        if "Construction parameter: dataset: "+dataset+" train_size: 100000 n_centroids: " in x:
            centroids = float(x.split("Construction parameter: dataset: "+dataset+" train_size: 100000 n_centroids: ")[1].split(" ")[0])
            construction_centroids.append(centroids)

print(construction_centroids)
plt.plot(construction_centroids, construction_time)
plt.xlabel("Centroid Setting")
plt.ylabel("Time / s")
plt.title(title)
plt.show()


