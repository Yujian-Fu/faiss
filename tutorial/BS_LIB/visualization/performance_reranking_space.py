import numpy as np 
import matplotlib.pyplot as plt

dataset = "SIFT1M"

nbits = 8
M = [4, 8]

file_name = "/home/yujian/Desktop/Recording_Files/VQ/" + dataset + "/recording_reranking_space_8_8_.txt"

centroids = []
spaces_1 = []
spaces_10 = []
spaces_100 = []
recording = False
file = open(file_name, "r")
f1 = file.readlines()
plt.figure()
for x in f1:
    if recording:
        reranking_space = x.split(" ")
        print(len(reranking_space))
        avg_1_space = np.mean([float(reranking_space[temp * 3 + 0]) for temp in range(100)])
        avg_10_space = np.mean([float(reranking_space[temp * 3 + 1]) for temp in range(100)])
        avg_100_space = np.mean([float(reranking_space[temp * 3 + 2]) for temp in range(100)])
        spaces_1.append(avg_1_space)
        spaces_10.append(avg_10_space)
        spaces_100.append(avg_100_space)
        recording = False

    if "The VQ centroids: " in x:
        print(x.split("The VQ centroids: "))
        centroids.append(float(x.split("The VQ centroids: ")[1].split("\n")[0]))
        recording = True
    

plt.plot(list(centroids), list(spaces_1))
plt.plot(list(centroids), list(spaces_10))
plt.plot(list(centroids), list(spaces_100))
plt.title("Reranking Spapce Requirement for M: 4 nbits: 8")
plt.xlabel("Centroids")
plt.ylabel("Reranking Space")
plt.show()