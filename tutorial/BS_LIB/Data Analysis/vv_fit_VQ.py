import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math



filepath = "/home/yujian/Desktop/Recording_Files/VQ/analysis/SIFT_reasoning.txt"

file = open(filepath, "r")
f1 = file.readlines()
position = -1
centroids = 0
recall_num = 100
targte_recall_num = int(recall_num * 0.8)
target_index = 0
num_cluster = 0
max_vv = 0

train_x = []
train_y = []
centroid_instance = []
centroid_cluster = []
avg_cluster = []
for x in f1:
    if position == 1:
        gt_vv = (x.split(" ")[0: -1])
        gt_vv = list(map(int, gt_vv))
        max_vv = gt_vv[target_index]
        centroid_instance.append(max_vv)
        centroid_cluster.append(target_index + 1)
        position = -1
        

    if position == 0:
        gt_vv = (x.split(" ")[0: -1])
        gt_vv = list(map(int, gt_vv))
        for i in range(len(gt_vv)):
            if gt_vv[i] >= targte_recall_num:
                target_index = i
                break
        position += 1
    
    if "Construction parameter: dataset: SIFT 1000000 " in x:
        centroids =  int(x.split(" ")[-3])
        train_x.append(centroids)
        if (len(centroid_instance) == 1000):
            print(centroid_cluster)
            train_y.append(np.mean(centroid_instance))
            avg_cluster.append(np.mean(centroid_cluster))
            print(avg_cluster[-1])
        
        centroid_instance = []
        centroid_cluster = []


    if "R@" + str(recall_num)+" MC: " in x:
        position = 0

train_y.append(np.mean(centroid_instance))
avg_cluster.append(np.mean(centroid_cluster))

train_x = np.array(train_x)
train_y = np.array(train_y)
popt,pcov=curve_fit(lambda x, a1, a2, a3, a4: a1 * x**3 + a2*x**2+a3*x +a4, train_x, train_y)
a1,a2,a3,a4 = popt
print(a1, a2, a3, a4)
y_fitted = a1 * train_x**3 + a2 * train_x**2 + a3*train_x +a4
print(np.average(abs(y_fitted - train_y) / train_y))
plt.figure()
plt.scatter(train_x, train_y , c = 'r', alpha=0.3)
plt.plot(train_x,y_fitted,'-b', label ='Fitted curve')
plt.xlabel("num centroids")
plt.ylabel("vv for " + str(targte_recall_num))
plt.savefig("/home/yujian/Desktop/Recording_Files/VQ/analysis/num_centroid-vv" + str(targte_recall_num)+".png")
plt.show()

avg_cluster = np.array(avg_cluster)
popt,pcov=curve_fit(lambda x, a1, a2, a3, a4: a1 * x**3 + a2*x**2+a3*x +a4, train_x, avg_cluster)
a1,a2,a3,a4 = popt
print(a1, a2, a3, a4)
clus_fitted = a1 * train_x**3 + a2 * train_x**2 + a3*train_x +a4
print(np.average(abs(clus_fitted - avg_cluster) / avg_cluster))
plt.figure()
plt.scatter(train_x, avg_cluster , c = 'r', alpha=0.3)
plt.plot(train_x,clus_fitted,'-b', label ='Fitted curve')
plt.xlabel("num centroids")
plt.ylabel("cluster for " + str(targte_recall_num))
plt.savefig("/home/yujian/Desktop/Recording_Files/VQ/analysis/num_centroid-clus" + str(targte_recall_num)+".png")
plt.show()



# -3.550910050808458e-07 0.0037482834557632464 -13.467477069891013 21663.425807983156