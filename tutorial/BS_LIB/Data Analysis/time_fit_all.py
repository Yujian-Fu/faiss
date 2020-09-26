import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

'''
vq layer time:
upper_nc = 1, upper_space, nc_per_group, nq = 1000
'''
def vq_time(x):
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = [0.000613746547749703, 0.0065334267541012625, -2.411200557584412e-07, -0.008832056526501078, 9.716025157101817e-05, -0.002365846261019691, -0.0003561341410022081, 8.25727852495746e-06, 0.061537666157024946, -0.9152645783466672]
    return a1 * x[1] * x[2] * x[3] + a2 * x[1] * x[2] + a3 * x[0] * x[1] * x[2] * x[3] + a4 * x[3] + a5 * x[0] * x[3]  + a6 * x[1] * x[3] + a8 * x[0] * x[1] * x[3]  + a9 * x[0] / x[1] + a7 * x[0] / x[1] * x[3] + a10

'''
k_min_time:
origin_value, target_value, nq = 1000
'''
def k_min_time(x):
    a1, a2, a3, a4, a5, a6 = [2.622118050778938e-05, 0.0002896727591980262, 1.8526843527901282e-07, 0.0015803155863680122, 0, 0.5363022453685298]
    return a1 * x[0] * x[2] + a2 * x[1] * x[2] + a3 * x[0] * x[1] * x[2] + a4 * x[2] + a5 * x[0] / x[1] * x[2]  + a6

'''
instance time:
M instances
'''
def instance_time(x):
    a1, a2, a3 = [7.302502913355863e-06, 1.5658279777016744e-05, 0.02355779065988822]
    return  a1 * x[1] * x[0] + a2 * x[1] + a3

'''
The vq time, k_min_time is for each query instance
x = [num_query, vq_time, k_min_time_1, instance_time, k_min_time_2, num_centroid, keep_space, M, nbits]
     0          1        2             3               4             5            6           7  8     

'''
def fitFunc(x, a1, a2, a3, a4, a5, a6):
    return x[1] + x[2] + x[3] + x[4] + x[0] * (a1 * x[5] + a2 * x[6] + a3 * x[7] * x[8]) + a4

'''
The parameter:
x = [number of centroids, keep space, visit instance]
'''
def newfitFunc(x, a1, a2, a3, a4, a5):
    return  a1 * x[0] + a2 * x[1]  + a3

def fittedFunc(x):
    a1, a2, a3 = [-0.1985797650767179, 7.8163429691468655, 203.95638247098645]
    return a1 * x[0] + a2 * x[1]  + a3

'''
x is the number of centroid built on SIFT1M
'''
def max_vv(x):
    a1, a2, a3, a4 = [-3.2713423005457553e-07, 0.003252678043728797, -11.069227126312628, 21969.06307876916]
    return int(a1 * x**3 + a2*x**2+a3*x +a4) + 1

def clus_vv(x):
    a1, a2, a3, a4 = [1.8215488947559045e-10, -1.7973911102623078e-06, 0.0104091944499985, 3.368365009190357]
    return int(a1 * x**3 + a2*x**2+a3*x +a4) + 1


'''
x:
centroid_num, recall_num, num_cluster_selected, max_visited_vectors
'''

def fit(x, y):
    query_num = 1000
    M = 16
    nbits = 8
    base_vector_size = 1000000
    update_x = []
    update_y = []
    sum_y_parts = []

    for i in range(len(x)):
        each_x = x[i]
        each_y = y[i]
        centroid_vectors = base_vector_size / each_x[0] * each_x[2]
        visited_vectors =  centroid_vectors if centroid_vectors < each_x[3] else each_x[3]
        Vq_time = vq_time([1, 1, each_x[0], query_num])
        K_min_time_1 = k_min_time([each_x[0], each_x[2], query_num])
        K_min_time_2 = k_min_time([visited_vectors, each_x[1], query_num])
        Instance_time = query_num * instance_time([M, visited_vectors])
        
        print(Vq_time, K_min_time_1, K_min_time_2, Instance_time, each_y)
        sum_parts = Vq_time + K_min_time_1 + Instance_time + K_min_time_2
        sum_y_parts.append(sum_parts)
        update_y.append(each_y - sum_parts)
        update_x.append([each_x[0], each_x[2], visited_vectors])

    x_label = []
    for x in update_x:
        x_label.append("(" + str(int(x[0])) + " " + str(int(x[1])) + " " + str(int(x[2])) + ")")

    update_x = np.array(update_x).T
    update_y = np.array(update_y)

    popt,pcov=curve_fit(newfitFunc,update_x,update_y)
    a1, a2, a3, a4, a5 = popt
    y_fitted = newfitFunc(update_x, a1, a2, a3, a4, a5)

    y_fitted = np.array(y_fitted) + np.array(sum_parts)
    update_y = update_y + np.array(sum_parts)
    update_y[np.where(update_y > y_fitted * 1.5)] = y_fitted[np.where(update_y > y_fitted * 1.5)] 

    print(a1, a2, a3, a4, a5)
    print(np.average(abs(y_fitted - update_y)/update_y))
    
    inds = y_fitted.argsort()
    plt.figure()
    plt.scatter(np.array(x_label)[inds], y_fitted[inds], c = 'b', alpha=0.3, label = "Fitted Points")
    plt.scatter(np.array(x_label)[inds], update_y[inds], c = 'r', alpha=0.3, label = "Original Points")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=5)
    plt.xlabel("Configuration")
    plt.ylabel("Search Time")
    plt.title("Configuration - Search Time")
    plt.legend()
    plt.show()
    #plt.savefig("/home/yujian/Desktop/Recording_Files/VQ/analysis/Config_SearchTime.png")

'''
Code for fit the total search time
'''
'''
query_time = 0
clusters = 0
max_visited_vectors = 0
recall_num = 0
train_x = []
y = []
filepath = "/home/yujian/Downloads/similarity_search_datasets/models_VQ/sift1M/recording_"

for centroid_num in range(500, 3100, 100):
    record_file = filepath  + str(centroid_num) + "qps.txt"
    file = open(record_file, "r")
    f1 = file.readlines()

    for x in f1:
        if "Finish SearchThe time usage: " in x:
            query_time = float(x.split(" ")[4])
        
        if "The recall@" in x:
            recall_num = float(x.split(" ")[1].split("recall@")[1])

        if "The search parameters is" in x:
            clusters = float(x.split(" ")[4])
            max_visited_vectors = float(x.split(" ")[-1])

            train_x.append([centroid_num, recall_num, clusters, max_visited_vectors])
            y.append(query_time * 1000)
        
fit(train_x, y)
'''

prediction = []
for centroid_num in range(500, 3600, 100):
    recall_num = 10
    query_num = 1000
    M = 16
    clus_para = clus_vv(centroid_num)
    max_vv_para = max_vv(centroid_num)
    Vq_layer_time = vq_time([1, 1, centroid_num, query_num])
    K_min_time_1 = k_min_time([centroid_num, clus_para, query_num])
    K_min_time_2 = k_min_time([max_vv_para, recall_num, query_num])
    Instance_time = query_num * instance_time([M, max_vv_para])
    Other_time = fittedFunc([centroid_num, clus_para, max_vv_para])
    time_all = Vq_layer_time + K_min_time_1 + K_min_time_2 + Instance_time
    print(centroid_num, clus_para, max_vv_para, Vq_layer_time, K_min_time_1, K_min_time_2, Instance_time, Other_time)
    prediction.append([centroid_num, time_all])

print(prediction)

plt.figure()
plt.xlabel("centroid num")
plt.ylabel("Search Time / ms")
plt.title("Centroid num - Search Time (8 nn)")
plt.plot(np.array(prediction)[:, 0],np.array(prediction)[:, 1])
for a, b in zip(np.array(prediction)[:, 0], np.array(prediction)[:, 1]):  
    plt.text(a, b, (int(a),int(b)),ha='center', va='bottom', fontsize=8, rotation=90)  
plt.show()




