import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

filepath = "/home/yujian/Downloads/similarity_search_datasets/ANN_SIFT1M/recording/reranking_space.txt"

file = open(filepath, "r")
r1 = file.readlines()


M = 0
nbits = 0
nc = 0
min_length = 50
plot_flag = False
all_list = []
for x in r1:
    if "The PQ structure:" in x:
        if plot_flag:
            for i in range(len(all_list)):
                all_list[i] = all_list[i][:min_length]
            
            all_list = np.array(all_list)
            print(all_list.shape)
            avg_gt_place = np.mean(all_list, axis = 0)
            print(avg_gt_place.shape)
            plt.figure()
            plt.plot(range(min_length), avg_gt_place)
            plt.xlabel("Visited GT")
            plt.ylabel("Reranking space")
            plt.title("Reranking Space - GT " + str(M) + " " + str(nbits) + " " + str(nc))
            plt.savefig("/home/yujian/Desktop/Recording_Files/VQ/analysis/reranking_sapce "+ str(M) + " " + str(nbits) + " " + str(nc) +".png")
            all_list = []

        M = int(x.split(" ")[4])
        nbits = int(x.split(" ")[6])
        nc = int(x.split(" ")[-1])
        plot_flag = True
        
    
    if x.split(" ")[0] == "1":
        length = int(len(x.split(" ")) / 2)
        if min_length > length:
            min_length = length
        target_list = [x.split(" ")[(2 * index + 1)] for index in range(length)]
        target_list = list(map(float, target_list))
        all_list.append(target_list)
    



