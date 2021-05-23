import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math
import random

recall1_10_6 = [0.25, 0.295, 0.35, 0.372, 0.412, 0.42, 0.425]
time1_10_6 = [2.3, 2.35, 2.41, 2.51, 2.91, 3.43, 4.8]

recall1_10_10_5 = [0.242, 0.285, 0.331, 0.362, 0.402, 0.420, 0.431]
time1_10_10_5 = [2.13, 2.25, 2.341, 2.481, 2.591, 2.93, 4.52]

recall1_10_2_10_4 = [0.238, 0.291, 0.341, 0.3792, 0.412, 0.425, 0.4421]
time1_10_2_10_4 = [2.1, 2.2, 2.3, 2.4, 2.5, 2.9, 4.5]

recall1_10_3_10_3 = [0.201, 0.245, 0.311, 0.342, 0.382, 0.410, 0.411]
time1_10_3_10_3 = [2.13, 2.25, 2.341, 2.481, 2.591, 2.93, 4.52]
'''
plt.figure()
plt.plot(time1_10_6, recall1_10_6, label = "10^6")
plt.plot(time1_10_10_5, recall1_10_10_5, label = "10 + 10^5")
plt.plot(time1_10_2_10_4, recall1_10_2_10_4, label = "100 + 10^4")
plt.plot(time1_10_3_10_3, recall1_10_3_10_3, label = "1000 + 1000")
plt.xlabel("Time / ms")
plt.ylabel("Recall @1")
plt.title("SIFT1B")
plt.legend()
plt.show()
'''

recall10_10_6 = [0.29, 0.365, 0.45, 0.482, 0.502, 0.512, 0.514]
time10_10_6 = [2.35, 2.43, 2.45, 2.58, 3.01, 3.49, 4.92]

recall10_10_10_5 = [0.31, 0.376, 0.458, 0.490, 0.51, 0.523, 0.528]
time10_10_10_5 = [2.3, 2.35, 2.41, 2.51, 2.91, 3.43, 4.8]

recall10_10_2_10_4 = [0.298, 0.372, 0.441, 0.4792, 0.512, 0.525, 0.532]
time10_10_2_10_4 = [2.1, 2.2, 2.3, 2.4, 2.5, 2.9, 4.5]

recall10_10_3_10_3 = [0.281, 0.345, 0.421, 0.442, 0.502, 0.510, 0.511]
time10_10_3_10_3 = [2.33, 2.45, 2.481, 2.581, 3.091, 3.53, 5.02]
'''
plt.figure()
plt.plot(time10_10_6, recall10_10_6, label = "10^6")
plt.plot(time10_10_10_5, recall10_10_10_5, label = "10 + 10^5")
plt.plot(time10_10_2_10_4, recall10_10_2_10_4, label = "100 + 10^4")
plt.plot(time10_10_3_10_3, recall10_10_3_10_3, label = "1000 + 1000")
plt.xlabel("Time / ms")
plt.ylabel("Recall @10")
plt.title("SIFT1B")
plt.legend()
plt.show()
'''

centroid = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
inverted_index_1 = [0.4558, 0.4559, 0.463, 0.464, 0.465,0.467, 0.470,0.4705, 0.471, 0.472, 0.476, 0.475, 0.476, 0.477, 0.476, 0.478]
kmeans_1 =         [0.450, 0.458, 0.458, 0.456, 0.463, 0.460, 0.474, 0.468, 0.471, 0.466, 0.469,0.476, 0.474, 0.472, 0.476, 0.474, 0.476, 0.475]

inverted_index = [0.55431, 0.55891, 0.56183, 0.56382, 0.56579, 0.56775, 0.57049, 0.5711, 0.57266, 0.57433, 0.57689, 0.57655, 0.5756, 0.57697, 0.57729, 0.57665, 0.5769, 0.57814, 0.57839]

kmeans = [0.55219, 0.55851, 0.5624, 0.56341, 0.56474, 0.5657, 0.57068, 0.56922, 0.56947, 0.57235, 0.57256, 0.57488, 0.57329, 0.57367, 0.5783, 0.57562, 0.57831, 0.57797, 0.57838]

'''
plt.figure()
plt.plot(centroid, inverted_index_1, label = "co-optimization")
plt.plot(centroid, kmeans_1, label = "original kmeans")
plt.ylabel("Recall@1")
plt.xlabel("Number of Centroids")
plt.title("Recall Upper Bound on SIFT1M")
plt.xticks(centroid)
plt.legend()
plt.show()
'''


centroids = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000]
time_x = [1.7, 1.85, 1.80, 1.79, 1.77, 1.75, 1.78, 1.74, 1.75, 1.73, 1.71, 1.68, 1.75, 1.79, 1.81, 1.91, 1.99]
recall1_y = [0.460, 0.470, 0.475, 0.478, 0.480, 0.4815, 0.482, 0.4825, 0.483, 0.485, 0.486, 0.490, 0.491, 0.491, 0.4915, 0.492, 0.492]
#recall10_y = [0.563, 0.570, 0.575, 0.576, 0.578, 0.581, 0.5812, 0.5813, 0.5820, 0.5821, 0.5819, 0.5822, 0.5823, 0.5825, 0.5827, 0.5829, 0.5830]

#centroids = [100,  300,  500,  700,  900,  1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300, 3500, 3700, 3900, 4100, 4300, 4500, 4700, 4900]
#time_x =    [6.56, 6.03, 5.58, 5.10, 4.77, 4.20, 3.98, 4.59, 4.06, 4.98, 4.75, 5.79, 5.83, 5.67, 5.31, 6.15, 6.86, 6.93, 6.67, 6.88, 7.12, 7.54, 7.88, 7.48, 7.90]
#recall1_y = [0.440,0.450,0.455,0.458,0.465,0.469,0.478,0.485,0.487,0.488,0.489,0.4895,0.491,0.491,0.4915,0.492,0.492,0.4921,0.4923,0.4922,0.4923, 0.4925, 0.4927, 0.4928, 0.4923]

#centroids = [100, 1100, 2100, 3100, 1600, 1800]
#time_x = [6.56, 4.20, 4.75, 6.15, 4.36, 4.13]
#recall1_y = [0.44, 0.469, 0.489, 0.492, 0.487, 0.489]

'''
centroids = ["10,6", "10,7", "10,8", "30,6", "30,7", "30,8", "50,6", "50,7", "50,8", "70,6", "70,7", "70,8", "90,6", "90,7", "90,8", "110,6", "110,7", "110,8", "130,6", "130,7", "130,8", "150,6", "150,7", "150,8", "170,6", "170,7", "170,8", "190,6", "190,7", "190,8"]
time_x =    [6.72,     6.89,   6.91,    5.93,    6.15,     6.03,   5.42,    5.86,    6.23,     5.78,    6.13,    6.38,    6.32,    6.94,    7.12,    6.45,    6.87,      7.24,     6.23,     6.98,     7.45,    6.54,     7.14,      7.98,     6.95,     7.34,    8.12,      7.06,     7.69,    8.23]
recall1_y = [0.413,    0.428,   0.431,  0.430,   0.441,    0.449,  0.442,   0.449,   0.451,    0.449,   0.458,   0.472,   0.462,   0.466,   0.470,   0.470,   0.474,     0.476,    0.471,    0.473,    0.477,   0.471,    0.472,     0.475,    0.472,    0.4732,  0.476,     0.473,    0.475,   0.476]
'''
'''
centroids = ["10,6", "10,7", "10,8", "70,6", "70,7", "70,8", "130,6", "130,7", "190,6", "190,7", "100,6", "100,7"]
time_x =    [6.72,    6.89,    6.91,    5.78,    6.13,    6.38,     6.23,    6.98,     7.06,     7.69,     6.39,     6.90]
recall1_y = [0.413,   0.428,   0.431,   0.449,   0.458,   0.472,    0.471,   0.473,    0.473,    0.475,    0.470,    0.473]
'''

'''
plt.scatter(time_x, recall1_y, alpha=0.6, color = 'black', s = 10)
plt.title("DEEP1B VQ VQ")
for idx, time in enumerate(time_x):
    plt.text(time, recall1_y[idx], str(centroids[idx]), fontsize=7)
plt.xlabel("Time / ms")
plt.ylabel("Recall")
plt.show()
'''

centroids = list(range(300, 4000, 100))
nb = 1000000
visited_nb = []
for centroid in centroids:
    visited_nb.append(20 * nb / centroid)


index_time = [i * 10 for i in centroids]
base_time = visited_nb
b_c_dist = []
for centroid in centroids:
    b_c_dist.append(3000 / centroid)

'''
plt.plot(centroids, np.array(index_time)/10000, label = "Index Time", linestyle = "-", color = "black")
plt.plot(centroids, np.array(base_time)/10000,label = "Base Vector Time", linestyle = '--', color = "black")
plt.plot(centroids, (np.array(index_time) + np.array(base_time))/10000, label = "Total Time", linestyle = "-.", color = "black")
#plt.plot(centroids, b_c_dist, label = "b_c_dist", color = "black")
plt.xlabel("Number of Cluster", size = 17)
#plt.ylabel("Average b_c Distance", size = 17)
plt.tick_params(labelsize=15)
plt.ylabel("Time Consumption", size = 17)
plt.legend(prop={'size': 18})
plt.show()
'''


IMI_time = [1.35, 1.48, 1.57, 1.61, 2.31, 2.69, 3.92]
IMI_recall1 = [0.25, 0.295, 0.35, 0.372, 0.412, 0.42, 0.425]
IMI_recall10 = [0.29, 0.365, 0.45, 0.482, 0.502, 0.512, 0.514]

IMI_time_OPT = [1.14, 1.18, 1.31, 1.45, 1.80, 2.11, 3.56]
#IMI_time_OPT = [3.63, 3.75, 3.891, 4.081, 4.71, 4.93,  7.07]
IMI_recall1_OPT = [0.26, 0.32, 0.358, 0.399, 0.432, 0.439, 0.441]
IMI_recall10_OPT = [0.32, 0.39, 0.483, 0.533, 0.544, 0.555, 0.555]

ICI_time = [1.43, 1.55, 1.691, 1.7981, 2.51, 2.73, 3.82]
ICI_recall1 = [0.242, 0.285, 0.331, 0.362, 0.402, 0.415, 0.421]
ICI_recall10 = [0.31, 0.376, 0.458, 0.510, 0.513, 0.518, 0.520]

ICI_time_HNSW = [1.4, 1.5, 1.67, 1.76, 2.5, 2.7, 3.9]
ICI_recall1_HNSW = [0.24, 0.28, 0.33, 0.36, 0.40, 0.415, 0.43]
ICI_recall10_HNSW = [0.3, 0.37, 0.45, 0.503, 0.514, 0.525, 0.54]

ICI_time_OPT = [1.04, 1.1, 1.21, 1.39, 1.80, 2.11, 3.56]
#ICI_time_OPT = [3.75, 3.88, 3.97, 4.071, 4.88, 5.09,  6.72]
ICI_recall1_OPT = [0.26, 0.32, 0.358, 0.419, 0.442, 0.448, 0.449]
ICI_recall10_OPT = [0.335, 0.402, 0.503, 0.523, 0.534, 0.545, 0.547]

ICI_time_OPQ = [1.42, 1.59, 1.67, 1.84, 2.54, 2.83, 4.03]
ICI_recall1_OPQ = [0.257, 0.295, 0.341, 0.368, 0.412, 0.438, 0.445]
ICI_recall10_OPQ = [0.33, 0.396, 0.478, 0.526, 0.543, 0.552, 0.558]

II_time = [1.2, 1.3, 1.5, 1.7, 2.4, 2.6, 3.8]
II_recall1 = [0.238, 0.291, 0.341, 0.3792, 0.422, 0.435, 0.435]
II_recall10 = [0.298, 0.372, 0.441, 0.4792, 0.512, 0.525, 0.532]

II_time_OPT = [1.04, 1.1, 1.21, 1.39, 1.80, 2.11, 3.61]
#II_time_OPT = [3.85, 3.99, 4.09, 4.67, 5.15, 5.84,  7.12]
II_recall1_OPT = [0.248, 0.301, 0.361, 0.422, 0.432, 0.455, 0.458]
II_recall10_OPT = [0.31, 0.382, 0.451, 0.492, 0.52, 0.535, 0.538]

II_time_HNSW = [1.04, 1.1, 1.21, 1.39, 1.80, 2.11, 3.71]
II_recall1_HNSW = [0.236, 0.289, 0.338, 0.375, 0.407, 0.43, 0.43]
II_recall10_HNSW = [0.29, 0.37, 0.44, 0.474, 0.5, 0.52, 0.53]

II_time_OPQ = [1.23, 1.35, 1.58, 1.81, 2.55, 2.78, 3.98]
II_recall1_OPQ = [0.268, 0.317, 0.365, 0.422, 0.440, 0.461, 0.469]
II_recall10_OPQ = [0.31, 0.382, 0.451, 0.492, 0.522, 0.535, 0.542]

VQT_time = [1.33, 1.45, 1.781, 1.91, 2.891, 2.93, 4.02]
VQT_recall1 = [0.201, 0.245, 0.351, 0.382, 0.392, 0.410, 0.411]
VQT_recall10 = [0.281, 0.345, 0.421, 0.432, 0.482, 0.500, 0.501]

VQT_time_OPT = [1.23, 1.35, 1.69, 1.91, 2.891, 2.93, 4.02]
VQT_recall1_OPT = [0.241, 0.345, 0.411, 0.432, 0.442, 0.450, 0.451]
VQT_recall10_OPT = [0.291, 0.375, 0.501, 0.522, 0.535, 0.54, 0.541]

IVFADC_time = [1.48, 1.56, 1.73, 1.84, 2.95, 3.32, 3.98]
IVFADC_recall1 = [0.201, 0.245, 0.311, 0.342, 0.382, 0.410, 0.411]
IVFADC_recall10 = [0.297, 0.312, 0.437, 0.468, 0.532, 0.541, 0.543]

ICI_SIFT1b_time = [3.43, 3.55, 3.691, 4.381, 4.51, 4.73, 5.92, 6.87]
ICI_SIFT1b_recall1 = [0.212, 0.245, 0.311, 0.362, 0.382, 0.410, 0.421, 0.422]
ICI_SIFT1b_recall10 = [0.31, 0.376, 0.458, 0.510, 0.52, 0.54, 0.548, 0.550]

IMI_SIFT1b_time = [3.35, 3.48, 3.57, 4.21, 4.48, 4.69, 5.73, 6.32]
IMI_SIFT1b_recall1 = [0.22, 0.245, 0.32, 0.372, 0.412, 0.42, 0.425, 0.428]
IMI_SIFT1b_recall10 = [0.29, 0.365, 0.45, 0.522, 0.542, 0.548, 0.554, 0.558]

IMI_base_time = [3.65, 3.79, 3.89, 4.47, 4.95, 5.64, 6.87, 7.92]
IMI_base_recall1 = [0.16, 0.195, 0.26, 0.332, 0.372, 0.389, 0.395, 0.398]
IMI_base_recall10 = [0.21, 0.305, 0.38, 0.472, 0.512, 0.52, 0.520, 0.525]

IVFADC_base_time = [3.83, 4.15, 4.31, 4.681, 4.83, 5.23, 6.92, 7.67]
IVFADC_base_recall1 = [0.172, 0.25, 0.274, 0.342, 0.362, 0.40, 0.409, 0.414]
IVFADC_base_recall10 = [0.22, 0.335, 0.41, 0.482, 0.522, 0.52, 0.521, 0.522]

'''
plt.plot(IMI_time, IMI_recall10, label = "IMI")
plt.plot(ICI_time, ICI_recall10, label = "ICI")
plt.plot(II_time, II_recall10, label = "II")
plt.plot(VQT_time, VQT_recall10,label = "VQT")
plt.plot(IVFADC_time, IVFADC_recall10, label = "IVFADC")
plt.title("SIFT1B subset")
plt.xlabel("Search Time / ms")
plt.ylabel("Recall @10")
plt.legend()
plt.show()
'''

'''
plt.plot(ICI_time, ICI_recall1, label = "ICI", linestyle=':', color = 'black', marker = ".")
plt.plot(IMI_time, IMI_recall1, label = "IMI", linestyle='-.', color = 'black', marker = ',')
plt.plot(II_time, II_recall1, label = "II", linestyle='-.', color = 'black', marker = 'v')
plt.plot(VQT_time, VQT_recall1, label = "VQT", linestyle = ":", color = 'black', marker = '^')

plt.plot(ICI_time_OPT, ICI_recall1_OPT, label = "ICI_optimized", linestyle='-', color = 'black', marker = '<')
plt.plot(IMI_time_OPT, IMI_recall1_OPT, label = "IMI_optimized", linestyle='-', color = 'black', marker = '>')
plt.plot(II_time_OPT, II_recall1_OPT, label = "II_optimized", linestyle='--', color = 'black', marker = '1')
plt.plot(VQT_time_OPT, VQT_recall1_OPT, label = "VQT_optimized", linestyle='--', color = 'black', marker = '2')
plt.xlabel("Search Time / ms")
plt.ylabel("Recall")
plt.title("SIFT1B Parameter Optimization")
#plt.legend()
plt.show()
'''

'''
plt.plot(II_time_OPT, II_recall10_OPT, label = "3 layers (100)")
plt.plot(ICI_time_OPT, ICI_recall10_OPT, label = "3 layers (1000)")

plt.plot(IMI_time_OPT, IMI_recall10_OPT, label = "3 layers (10000)")

plt.plot(IMI_base_time, IMI_base_recall10, label = "4 layers (10)")
plt.plot(IVFADC_base_time, IVFADC_base_recall10, label = "4 layers (100)")

plt.xlabel("Search Time / ms")
plt.ylabel("Recall @10")
plt.legend()
plt.show()
'''

LID = [9] * 384 + [10] * 292 + [8] * 201 + [7] * 74 + [11] * 21 + [12] * 8 + [6] * 20
RC = [3.5] * 325 + [3.4] * 221 + [3.3] * 106 + [3.2] * 20 + [3.6] * 189 + [3.7] * 78 + [3.8] * 49 + [3.9] * 12
SIZE = [1000] * 258 + [1100] * 170 + [1200] * 140 + [1300] * 88 + [1400] *  23 + [900] * 245 + [800] * 120 + [700] * 56


LID = [12] * 264 + [13] * 292 + [11] * 201 + [10] * 124 + [14] * 71 + [15] * 8 + [9] * 20 + [16] * 20
RC = [3.8] * 245 + [3.7] * 221 + [3.6] * 136 + [3.5] * 40 + [3.9] * 199 + [4.0] * 98 + [4.1] * 49 + [4.2] * 12
SIZE = [1000] * 278 + [1100] * 150 + [1200] * 160 + [1300] * 68 + [1400] *  43 + [900] * 225 + [800] * 140 + [700] * 36

'''
plt.hist(SIZE, bins=7, normed=0,  edgecolor="black", alpha=0.7)
plt.xlabel("SIZE")
plt.ylabel("Number of Clusters")
plt.title("Cluster Size distribution")
plt.show()
'''

'''
342
mu = 0
sigma = 0.12

time_1 = [1.0, 1.1, 1.5, 1.8, 1.9, 2.2, 2.4, 2.6, 2.8, 3.1, 3.2, 3.4, 3.5, 3.7, 3.8, 4.0, 4.2, 4.6, 4.8, 4.9, 5.12, 5.31, 5.50, 5.62, 5.72, 5.80, 5.98, 6.14, 6.24, 6.38, 6.41,6.50, 6.52, 6.63, 6.75, 6.88, 6.9, 7.2, 7.3, 7.4]
IVF_recall_1 = [18.9, 22.7, 26.5, 28.3, 29.8, 31.0, 32.4, 34.5, 35.6, 36.2, 36.6, 36.8, 36.9, 37.1, 37.6, 37.8, 37.9, 38.0, 38.3, 38.5, 38.6, 38.6, 38.7, 38.7, 38.8, 38.9, 39.0, 39.0, 39.1, 39.0, 39.0, 39.0, 38.9, 38.95, 39.0, 38.88, 38.9, 39.0, 38.95, 39.01]


faiss_time = [1.5, 1.7,1.85, 2.06, 2.15, 2.25, 2.54, 2.68, 2.81, 2.90, 3.05, 3.18, 3.3, 3.31, 3.52, 3.78, 3.79, 3.98, 4.0, 4.34, 4.45, 4.76, 5.0, 5.12, 5.30, 5.42, 5.62, 5.80, 5.98, 6.22, 6.24, 6.28, 6.31,6.35, 6.42]
faiss_recall_1 = [16.7, 17.9, 19.6, 21.7, 23.8, 25.9, 27.4, 28.8, 29.5, 31.8, 33.9, 35.2, 35.4, 35.8, 35.9, 35.9, 36.0, 36.2, 36.2, 36.8, 37.0, 37.10, 37.15, 37.14, 37.16, 37.2, 37.15, 37.12, 37.11, 37.17, 37.15, 37.13, 37.18, 37.20, 37.15]

IMI_time = [ 2.31, 2.45, 2.61, 2.83, 2.93, 3.05, 3.18, 3.31, 3.78, 3.98, 4.34, 4.76, 5.12, 5.42, 5.80, 6.22, 6.28, 6.35, 6.53, 6.89, 7.4, 7.49]
IMI_recall_1 = [11.5, 13.6, 16.7, 19.5, 21.8, 22.9, 24.2, 25.6, 27.5, 28.1, 29.0, 29.5, 29.8, 30.2, 30.9, 31.2, 31.5, 31.7, 31.8, 31.9, 31.8,32.0]



time_1 = [1.0, 1.78, 2.36, 2.89, 3.41, 3.79, 4.14, 4.69, 4.92, 5.46, 5.80, 6.13, 6.61, 6.89, 7.12, 7.4]
IVF_recall_1 = [28.9, 45.7, 58.1, 64.6, 70.8, 73.6, 74.8, 75.9, 76.4, 77.3, 78.9, 78.6, 77.8, 77.6, 77.9, 78.9]

faiss_time = [1.9, 2.32, 2.95, 3.87, 4.69, 5.32, 5.98, 6.42]
faiss_recall_1 = [34.7, 50.8, 59.4, 68.3, 71.6, 71.9, 72.8, 72.5]

IMI_time = [2.31, 3.40, 4.45, 5.38, 6.24, 7.08,  7.49]
IMI_recall_1 = [28.8, 49.6, 58.8, 60.8, 61.8, 62.3, 61.9]

plt.plot(time_1, IVF_recall_1, label = "IVFADC+G+P 16")
plt.plot(faiss_time, faiss_recall_1, label = "Faiss IVFADC 16")
plt.plot(IMI_time, IMI_recall_1, label = "Faiss IMI 16")
plt.legend()
plt.title("Performance Figure")
plt.xlabel("Time / ms")
plt.ylabel("Recall@10")
plt.show()



#16 10
time_16_10 = [0.684, 1.108, 1.804, 2.507, 3.336, 4.287, 4.784, 5.05, 5.57, 6.11, 6.78, 8.84]
recall_16_10 = [0.3925, 0.4348, 0.4541, 0.4606, 0.534, 0.5367, 0.539, 0.5539, 0.555, 0.556, 0.573, 0.573]

#16 1 
time_16_1 = [0.669, 1.084, 1.78, 2.68, 3.52, 4.286, 4.815, 5.292, 6.095, 6.552, 7.198, 8.937]
recall_16_1 = [0.361, 0.391, 0.402, 0.401, 0.453, 0.453, 0.454, 0.475, 0.475, 0.474, 0.492, 0.492]

#8 1 
time_8_1 = [0.68, 1.098, 1.723, 2.41, 3.29, 3.72, 4.45, 5.04, 5.13, 5.80, 6.35, 8.04]
recall_8_1 = [0.225, 0.227, 0.231, 0.232, 0.26, 0.26, 0.264, 0.276, 0.277, 0.277, 0.279, 0.28]

#8 10
time_8_10 = [0.657, 1.039, 1.709, 2.375, 3.039, 3.607, 4.195, 4.615, 5.007, 5.601, 6.553, 7.57]
recall_8_10 = [0.295, 0.315, 0.325, 0.326, 0.368, 0.369, 0.370, 0.379, 0.38, 0.38, 0.388, 0.389]

#32 10
time_32_10 = [0.817, 1.24, 2.146, 2.841, 3.711, 4.557, 5.329, 5.722, 6.658, 7.025, 8.051, 9.684]
recall_32_10 = [0.439, 0.499, 0.532, 0.545, 0.654, 0.661, 0.665, 0.689, 0.693, 0.6934, 0.7155, 0.7171]

#32 1
time_32_1 = [0.788, 1.312, 2.17, 2.96, 3.79, 4.66, 5.47, 6.00, 7.17, 8.16, 9.87]
recall_32_1 = [0.447, 0.49, 0.498, 0.504, 0.598, 0.6, 0.631, 0.632, 0.633, 0.661, 0.663]

#16 10
ICI_time_16_10 = [0.53,  0.94,  1.3504, 2.007, 2.8736, 3.287, 3.884, 4.15, 4.87, 5.91, 6.18, 7.03, 8.12]
ICI_recall_16_10 = [0.3925, 0.4548, 0.4741, 0.5206, 0.554, 0.5567, 0.559, 0.5739, 0.575, 0.586, 0.593, 0.593, 0.594]

#16 1
ICI_time_16_1 = [0.53,   1.3504, 2.007, 2.8736, 3.287, 3.884, 4.15, 4.87, 5.91, 6.18, 7.03, 8.12]
ICI_recall_16_1 = [0.371, 0.411, 0.412, 0.441, 0.483, 0.4873, 0.494, 0.4955, 0.4975, 0.50, 0.51, 0.515]

#8 1
ICI_time_8_1 = [0.53,  0.94,  1.3504, 2.8736, 3.287, 3.884, 4.15, 4.87, 5.91, 6.18, 7.03, 8.12]
ICI_recall_8_1 = [0.235, 0.247, 0.251, 0.24, 0.255, 0.27, 0.273, 0.276, 0.28, 0.283, 0.285, 0.29]

#8 10
ICI_time_8_10 = [0.49,    1.2504, 1.807, 2.0736, 3.587, 3.784, 4.45, 4.47, 5.01, 6.88, 7.93, 8.12]
ICI_recall_8_10 = [0.315, 0.335, 0.345, 0.346, 0.378, 0.379, 0.390, 0.395, 0.398, 0.399, 0.399, 0.401]


#32 10
ICI_time_32_10 = [0.53,  0.94,  1.3504, 2.007, 2.8736,  3.884, 4.15, 4.87, 5.91, 6.18, 7.03, 8.12]
ICI_recall_32_10 = [0.409, 0.539, 0.572, 0.585, 0.694, 0.701, 0.709, 0.713, 0.720, 0.728, 0.735, 0.7471]

#32 1
ICI_time_32_1 = [0.49,    1.2504, 1.807, 2.0736, 3.587, 3.784, 4.45, 5.01, 6.88, 7.93, 8.12]
ICI_recall_32_1 = [0.439, 0.499, 0.532, 0.545, 0.654, 0.661, 0.665, 0.689, 0.693, 0.6934, 0.693]


plt.plot(time_8_10, recall_8_10, label = "8 bytes IVFADC")
plt.plot(ICI_time_8_10, ICI_recall_8_10, label = "8 bytes ICI")
plt.plot(time_16_10, recall_16_10, label = "16 bytes IVFADC")
plt.plot(ICI_time_16_10, ICI_recall_16_10, label = "16 bytes ICI")
plt.plot(time_32_10, recall_32_10, label = "32 bytes IVFADC")
plt.plot(ICI_time_32_10, ICI_recall_32_10, label = "32 bytes ICI")
plt.plot()
plt.legend()
plt.title("SIFT1B Dataset")
plt.xlabel("Time / ms")
plt.ylabel("Recall@1")
plt.show()



x_1 = [2.7, 3.0, 3.5, 3.8, 4.0, 4.2, 4.5, 5.0, 5.2, 5.5]
y_1 = [0.54, 0.56, 0.572, 0.577, 0.579, 0.582, 0.590, 0.592, 0.593, 0.593]

x_2 = [2.7, 3.0, 3.2, 3.5, 3.7, 4.0, 4.5, 4.7, 4.9]
y_2 = [0.56, 0.582, 0.587, 0.591, 0.595, 0.598, 0.603, 0.604, 0.605]

x_3 = [2.7, 3.0, 3.5, 3.7, 4.0, 4.2, 4.5, 5.0]
y_3 = [0.548, 0.568, 0.576, 0.585, 0.587, 0.590, 0.595, 0.5955]


x_1 = [0.36, 0.38, 0.408, 0.422, 0.428, 0.43]
y_1 = [3.5, 4.2, 5.17, 6.67, 7.619, 12.0]

x_2 = [0.328, 0.37, 0.4, 0.41, 0.419, 0.424, 0.425, 0.43, 0.434, 0.435]
y_2 = [0.46, 1.2, 1.49, 2.45, 4.33, 5.5, 6.0, 7.7, 11.06, 12.5 ]

x_3 = [0.326, 0.353, 0.377, 0.398, 0.402, 0.413, 0.418, 0.423, 0.427, 0.433, 0.442, 0.449, 0.45, 0.452, 0.456, 0.458, 0.462, 0.471, 0.472, 0.473, 0.476, 0.477, 0.477, 0.477, 0.4775]
y_3 = [0.273, 0.274, 0.28, 0.29, 0.34, 0.38, 0.46, 0.52, 0.54, 0.58, 0.74, 0.81, 0.87, 0.94, 0.98, 1.10, 1.32, 1.32, 1.49, 2.2, 4, 5, 6, 7, 8]


plt.plot(y_1, x_1, label = "M = 4")
plt.plot(y_2, x_2, label = "M = 6")
plt.plot(y_3, x_3, label = "M = 8")



#plt.plot(x_4, y_4)
plt.title("Inverted Multi-Index on SIFT1M")
plt.xlabel("Time / ms")
plt.ylabel("Recall@1")
plt.legend()
plt.show()




IIG_R1 = []

# M = 7
IMIS_R1 = [0.319, 0.349, 0.371, 0.389, 0.4,   0.413, 0.429, 0.434, 0.447, 0.45, 0.455, 0.458, 0.46, 0.469, 0.473, 0.48]
IMIS_T1 = [0.1,   0.10,  0.127, 0.146, 0.162, 0.188, 0.208, 0.22,  0.248, 0.26, 0.278, 0.303, 0.33, 0.40,  0.43,  0.47]

# M = 8
IMIS_R10 = [0.299, 0.354, 0.3907, 0.4173, 0.4358, 0.4538, 0.4653, 0.4762,
            0.4838, 0.4922, 0.499, 0.50, 0.512,  0.516, 0.519, 0.525, 0.529, 
            0.531, 0.535, 0.539, 0.543, 0.548, 0.551, 0.557, 0.560, 0.5628,
            0.5648, 0.5654, 0.5673, 0.5678, 0.5682, 0.570, 0.571, 0.571]
IMIS_T10 = [0.088, 0.102, 0.11,   0.133,  0.1515, 0.168,  0.183,  0.202,
            0.217,  0.2396, 0.256, 0.272, 0.301, 0.314, 0.331, 0.352, 0.373,
            0.392, 0.425, 0.456, 0.571, 0.698, 0.699, 0.826, 1,     1.20,
            1.39,   1.42,   1.7,    1.83,   1.87,    2.06, 0.223, 0.233]

#M = 7
IMID_R1 = [0.108, 0.113, 0.125, 0.131, 0.133, 0.135, 0.139, 0.141, 0.141]
IMIS_T1 = [0.569, 0.434, 0.67,  0.65,  0.946, 0.848, 1.28,  1.21,  1.588]

# M = 6
IMID_R10 = [0.129, 0.146, 0.150, 0.157, 0.159, 0.162, 0.164, 0.1657, 0.1687,
            0.1706, 0.1709, 0.171, 0.1715, 0.1715]
IMID_T10 = [0.307, 0.416, 0.474, 0.594, 0.654, 0.714, 0.822, 0.93,   1.19,
            1.87,   2.13,   2.16,  2.28,   2.50]

# M = 7
IMIG_R1 = [0.108, 0.118, 0.125, 0.131, 0.133, 0.138, 0.14, 0.141, 0.141]
IMIG_T1 = [0.569, 0.434, 0.67,  0.65,  0.94,  1.40,  1.29, 1.21, 1.58]

# M = 7
IMIG_R10 = [0.115, 0.122, 0.129, 0.140, ]
IMIG_T10 = [0.493, 0.81, 0.60,   0.685, ]
'''

r_1_V_L = [0.25, 0.295, 0.35, 0.372, 0.412, 0.42, 0.425]
t_1_V_L = [1.3, 1.35, 1.41, 1.51, 1.91, 2.43, 3.8]
r_10_V_L = [0.335, 0.402, 0.503, 0.523, 0.534, 0.545, 0.547]
t_10_V_L = [1.3, 1.35, 1.41, 1.51, 1.91, 2.43, 3.8]

r_1_V_P = [0.25, 0.295, 0.35, 0.372, 0.412, 0.42, 0.425]
t_1_V_P = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]
r_10_V_P = [0.29, 0.365, 0.45, 0.482, 0.502, 0.512, 0.514]
t_10_V_P = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]

r_1_V_V = [0.257, 0.295, 0.341, 0.368, 0.412, 0.438, 0.445]
t_1_V_V = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]
r_10_V_V = [0.33, 0.396, 0.478, 0.526, 0.543, 0.552, 0.558]
t_10_V_V = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]

r_1_V_V_L = [0.238, 0.291, 0.341, 0.3792, 0.412, 0.425, 0.4421]
t_1_V_V_L = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]
r_10_V_V_L = [0.31, 0.376, 0.458, 0.510, 0.52, 0.54, 0.542]
t_10_V_V_L = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]

r_1_V_V_P = [0.201, 0.245, 0.311, 0.342, 0.382, 0.410, 0.411]
t_1_V_V_P = [1.35, 1.48, 1.57, 1.61, 2.31, 2.69, 3.92]
r_10_V_V_P = [0.29, 0.365, 0.45, 0.482, 0.502, 0.512, 0.514]
t_10_V_V_P = [1.35, 1.48, 1.57, 1.61, 2.31, 2.69, 3.92]

r_1_V_V_V = [0.26, 0.32, 0.358, 0.399, 0.432, 0.439, 0.441]
t_1_V_V_V = [1.14, 1.18, 1.31, 1.45, 1.80, 2.11, 3.56]
r_10_V_V_V = [0.32, 0.39, 0.483, 0.533, 0.544, 0.555, 0.555]
t_10_V_V_V = [1.14, 1.18, 1.31, 1.45, 1.80, 2.11, 3.56]

r_1_V_V_V_L = [0.24, 0.28, 0.33, 0.36, 0.40, 0.415, 0.43]
t_1_V_V_V_L = [1.43, 1.55, 1.691, 1.7981, 2.51, 2.73, 3.82]
r_10_V_V_V_L = [0.31, 0.376, 0.458, 0.490, 0.51, 0.523, 0.528]
t_10_V_V_V_L = [1.43, 1.55, 1.691, 1.7981, 2.51, 2.73, 3.82]

r_1_V_V_V_P = [0.242, 0.285, 0.331, 0.362, 0.402, 0.415, 0.421]
t_1_V_V_V_P = [1.4, 1.5, 1.67, 1.76, 2.5, 2.7, 3.9]
r_10_V_V_V_P = [0.281, 0.345, 0.421, 0.442, 0.502, 0.510, 0.511]
t_10_V_V_V_P = [1.4, 1.5, 1.67, 1.76, 2.5, 2.7, 3.9]

r_1_V_V_V_V = [0.242, 0.285, 0.331, 0.362, 0.402, 0.420, 0.431] 
t_1_V_V_V_V = [1.42, 1.59, 1.67, 1.84, 2.54, 2.83, 4.03]
r_10_V_V_V_V = [0.298, 0.372, 0.441, 0.4792, 0.512, 0.525, 0.532]
t_10_V_V_V_V = [1.42, 1.59, 1.67, 1.84, 2.54, 2.83, 4.03]

'''
plt.plot(t_10_V_L, r_10_V_L, label = "VQ LQ", marker = '.', color = 'black', linestyle = ':')
plt.plot(t_10_V_P, r_10_V_P, label = "VQ PQ", marker = 'o', color = 'black', linestyle = '--')
plt.plot(t_10_V_V, r_10_V_V, label = "VQ VQ", marker = 'v', color = 'black', linestyle = '-.')
plt.plot(t_10_V_V_L, r_10_V_V_L,label = "VQ VQ LQ", marker = '^', color = 'black', linestyle = '-')
plt.plot(t_10_V_V_P, r_10_V_V_P, label = "VQ VQ PQ", marker = '<', color = 'black', linestyle = "-.")
plt.plot(t_10_V_V_V, r_10_V_V_V, label = "VQ VQ VQ", marker = '>', color = 'black', linestyle = '-')
plt.plot(t_10_V_V_V_L, r_10_V_V_V_L, label = "VQ VQ VQ LQ", marker = '1', color = 'black', linestyle = '-.')
plt.plot(t_10_V_V_V_P, r_10_V_V_V_P, label = "VQ VQ VQ PQ", marker = '2', color = 'black', linestyle = ':')
plt.plot(t_10_V_V_V_V, r_10_V_V_V_V, label = "VQ VQ VQ VQ", marker = '3', color = 'black', linestyle = '--')
plt.title("DEEP1B Dataset")
plt.xlabel("Search Time / ms")
plt.ylabel("Recall")
plt.legend()
plt.show()
'''

t_1_10 = [1.33, 1.45, 1.781, 1.91, 2.891, 2.93, 4.02]
r_1_10 = [0.201, 0.245, 0.311, 0.342, 0.382, 0.410, 0.411]
r_10_10 = [0.281, 0.345, 0.421, 0.432, 0.482, 0.500, 0.501]

t_1_30 = [1.48, 1.56, 1.73, 1.84, 2.95, 3.32, 3.98]
r_1_30 = [0.201, 0.245, 0.311, 0.342, 0.382, 0.410, 0.411]
r_10_30 = [0.29, 0.365, 0.45, 0.482, 0.502, 0.512, 0.514]

t_1_50 = [1.23, 1.35, 1.58, 1.81, 2.55, 2.78, 3.98]
r_1_50 = [0.24, 0.28, 0.33, 0.36, 0.40, 0.415, 0.43]
r_10_50 = [0.29, 0.365, 0.45, 0.482, 0.502, 0.512, 0.514]

t_1_70 = [1.04, 1.1, 1.21, 1.39, 1.80, 2.11, 3.71]
r_1_70 = [0.242, 0.285, 0.331, 0.362, 0.402, 0.415, 0.421]
r_10_70 = [0.29, 0.37, 0.44, 0.474, 0.5, 0.52, 0.53]

t_1_100 = [1.04, 1.1, 1.21, 1.39, 1.80, 2.11, 3.61]
r_1_100 = [0.26, 0.32, 0.358, 0.389, 0.422, 0.428, 0.429]
r_10_100 = [0.31, 0.382, 0.451, 0.492, 0.52, 0.535, 0.538]

t_1_300 = [1.2, 1.3, 1.5, 1.7, 2.4, 2.6, 3.8]
r_1_300 = [0.238, 0.291, 0.341, 0.3792, 0.422, 0.435, 0.435]
r_10_300 = [0.298, 0.372, 0.441, 0.4792, 0.512, 0.525, 0.532]

t_1_500 = [1.42, 1.59, 1.67, 1.84, 2.54, 2.83, 4.03]
r_1_500 = [0.257, 0.295, 0.341, 0.368, 0.412, 0.438, 0.445]
r_10_500 = [0.33, 0.396, 0.478, 0.526, 0.543, 0.552, 0.558]

t_1_700 = [1.04, 1.1, 1.21, 1.39, 1.80, 2.11, 3.56]
r_1_700 = [0.26, 0.32, 0.358, 0.399, 0.432, 0.439, 0.441]
r_10_700 = [0.335, 0.402, 0.503, 0.523, 0.534, 0.545, 0.547]

t_1_1000 = [1.4, 1.5, 1.67, 1.76, 2.5, 2.7, 3.9]
r_1_1000 = [0.268, 0.317, 0.365, 0.422, 0.440, 0.461, 0.469] 
r_10_1000 = [0.3, 0.37, 0.45, 0.503, 0.514, 0.525, 0.54]

t_1_3000 = [1.43, 1.55, 1.691, 1.7981, 2.51, 2.73, 3.82]
r_1_3000 = [0.238, 0.291, 0.341, 0.3792, 0.412, 0.425, 0.4421]
r_10_3000 = [0.31, 0.376, 0.458, 0.510, 0.52, 0.54, 0.548]

t_1_5000 = [1.14, 1.18, 1.31, 1.45, 1.80, 2.11, 3.56]
r_1_5000 = [0.248, 0.301, 0.361, 0.422, 0.432, 0.451, 0.454]
r_10_5000 = [0.32, 0.39, 0.483, 0.533, 0.544, 0.555, 0.555]

t_1_7000 = [1.35, 1.48, 1.57, 1.61, 2.31, 2.69, 3.92]
r_1_7000 = [0.25, 0.295, 0.35, 0.372, 0.412, 0.42, 0.425]
r_10_7000 = [0.297, 0.312, 0.437, 0.468, 0.532, 0.541, 0.543]

t_1_10000 = [1.42, 1.59, 1.67, 1.84, 2.54, 2.83, 4.03]
r_1_10000 = [0.25, 0.295, 0.35, 0.372, 0.412, 0.42, 0.425]
r_10_10000 = [0.31, 0.382, 0.451, 0.492, 0.522, 0.535, 0.542]

t_1_30000 = [1.43, 1.55, 1.691, 1.7981, 2.51, 2.73, 3.82]
r_1_30000 = [0.242, 0.285, 0.331, 0.362, 0.402, 0.420, 0.431]
r_10_30000 = [0.31, 0.376, 0.458, 0.490, 0.51, 0.523, 0.528]

t_1_50000 = [1.3, 1.35, 1.41, 1.51, 1.91, 2.43, 3.8]
r_1_50000 = [0.236, 0.289, 0.338, 0.375, 0.407, 0.43, 0.43]
r_10_50000 = [0.298, 0.372, 0.441, 0.4792, 0.512, 0.525, 0.532]

t_1_70000 = [2.3, 2.35, 2.41, 2.51, 2.91, 3.43, 4.8]
r_1_70000 = [0.242, 0.285, 0.331, 0.362, 0.402, 0.415, 0.421]
r_10_70000 = [0.29, 0.365, 0.45, 0.482, 0.502, 0.512, 0.514]

t_1_100000 = [2.1, 2.2, 2.3, 2.4, 2.5, 2.9, 4.5]
r_1_100000 = [0.201, 0.245, 0.311, 0.342, 0.382, 0.410, 0.411]
r_10_100000 = [0.281, 0.345, 0.421, 0.442, 0.502, 0.510, 0.511]

t_1_300000 = [1.3, 1.35, 1.41, 1.51, 1.91, 2.43, 3.8, 4.5]
r_1_300000 = [0.22, 0.245, 0.32, 0.372, 0.412, 0.42, 0.425, 0.428]
r_10_300000 = [0.29, 0.365, 0.45, 0.522, 0.542, 0.548, 0.554, 0.558]

t_1_500000 = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52, 4.7]
r_1_500000 = [0.16, 0.195, 0.26, 0.332, 0.372, 0.389, 0.395, 0.398]
r_10_500000 = [0.22, 0.335, 0.41, 0.482, 0.522, 0.52, 0.521, 0.522]

t_1_700000 = [1.04, 1.1, 1.21, 1.39, 1.80, 2.11, 3.56, 4.6]
r_1_700000 = [0.16, 0.195, 0.26, 0.332, 0.372, 0.389, 0.395, 0.398]
r_10_700000 = [0.21, 0.305, 0.38, 0.472, 0.512, 0.52, 0.520, 0.525]

'''
plt.plot(t_1_10, r_10_10, label = "10", linestyle = "-", marker = '.', markersize=3, color = 'black')
plt.plot(t_1_30, r_10_30, label = "30", linestyle = "--", marker = ',', markersize=3, color = 'black')
plt.plot(t_1_50, r_10_50, label = "50", linestyle = "-.", marker = 'o',markersize=3, color = 'black')
plt.plot(t_1_70, r_10_70, label = "70", linestyle = ":", marker = 'v',markersize=3, color = 'black')
plt.plot(t_1_100, r_10_100, label = "100", linestyle = "-", marker = '^',markersize=3, color = 'black')
plt.plot(t_1_300, r_10_300, label = "300", linestyle = "--", marker = '<',markersize=3, color = 'black')
plt.plot(t_1_500, r_10_500, label = "500", linestyle = "-.", marker = '>',markersize=3, color = 'black')
plt.plot(t_1_700, r_10_700, label = "700", linestyle = ':', marker = '1',markersize=3, color = 'black')
plt.plot(t_1_1000, r_10_1000, label = "1000", linestyle = "-", marker = '2',markersize=3, color = 'black')
plt.plot(t_1_3000, r_10_3000, label = "3000", linestyle = "--", marker = '3',markersize=3, color = 'black')
plt.plot(t_1_5000, r_10_5000, label = "5000", linestyle = "-.", marker = '4',markersize=3, color = 'black')
plt.plot(t_1_7000, r_10_7000, label = "7000", linestyle = ":", marker = 's',markersize=3, color = 'black')
plt.plot(t_1_10000, r_10_10000, label = "10000", linestyle = "-", marker = 'p',markersize=3, color = 'black')
plt.plot(t_1_30000, r_10_30000, label = "30000", linestyle = "--", marker ='*',markersize=3, color = 'black')
plt.plot(t_1_50000, r_10_50000, label = "50000", linestyle = "-.", marker = 'h',markersize=3, color = 'black')
plt.plot(t_1_70000, r_10_70000, label = "70000", linestyle = ":", marker = 'H',markersize=3, color = 'black')
plt.plot(t_1_100000, r_10_100000, label = "100000", linestyle = "-", marker = '+',markersize=3, color = 'black')
plt.plot(t_1_300000, r_10_300000, label = "300000", linestyle = "--", marker = 'x',markersize=3, color = 'black')
plt.plot(t_1_500000, r_10_500000, label = "500000", linestyle = "-.", marker = 'D',markersize=3, color = 'black')
plt.plot(t_1_700000, r_10_700000, label = "700000", linestyle = ":", marker = 'd',markersize=3, color = 'black')
plt.title("DEEP1B Dataset")
plt.xlabel("Search Time / ms")
plt.ylabel("Recall @10")
plt.legend(ncol = 2)
plt.show()
'''

r_1_V_V = [0.257, 0.295, 0.341, 0.368, 0.412, 0.438, 0.445]
t_1_V_V = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]
r_10_V_V = [0.33, 0.396, 0.478, 0.526, 0.543, 0.552, 0.558]
t_10_V_V = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]

r_1_V_V_V = [0.26, 0.32, 0.358, 0.399, 0.432, 0.439, 0.441]
t_1_V_V_V = [1.14, 1.18, 1.31, 1.45, 1.80, 2.11, 3.56]
r_10_V_V_V = [0.32, 0.39, 0.483, 0.533, 0.544, 0.555, 0.555]
t_10_V_V_V = [1.14, 1.18, 1.31, 1.45, 1.80, 2.11, 3.56]

r_1_V_V_L = [0.238, 0.291, 0.341, 0.3792, 0.412, 0.425, 0.4321]
t_1_V_V_L = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]
r_10_V_V_L = [0.31, 0.376, 0.458, 0.510, 0.52, 0.54, 0.542]
t_10_V_V_L = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]

r_1_V_V_L_OP = [0.238, 0.311, 0.371, 0.3892, 0.432, 0.445, 0.4521]
t_1_V_V_L_OP = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]
r_10_V_V_L_OP = [0.33, 0.396, 0.468, 0.530, 0.545, 0.562, 0.568]
t_10_V_V_L_OP = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]

'''
plt.plot(t_1_V_V, r_1_V_V, label = "VQ VQ", color = "black", marker = '.', linestyle = '-.')
plt.plot(t_1_V_V_V, r_1_V_V_V, label = "VQ VQ VQ", color = "black", marker = ',', linestyle = '--')
plt.plot(t_1_V_V_L, r_1_V_V_L, label = "VQ VQ LQ", color = "black", marker = 'o', linestyle = '-.')
plt.plot(t_1_V_V_L_OP, r_1_V_V_L_OP, label = "VQ VQ LQ_OP", color = 'black', marker = 'v', linestyle = "-")
plt.title("SIFT1B Dataset")
plt.xlabel("Search Time / ms")
plt.ylabel("Recall")
#plt.legend()
plt.show()
'''

r_1_V_L = [0.257, 0.295, 0.341, 0.368, 0.412, 0.438, 0.445]
t_1_V_L = [1.42, 1.59, 1.67, 1.84, 2.54, 2.83, 4.03]
r_10_V_L = [0.33, 0.396, 0.478, 0.526, 0.543, 0.552, 0.558]
t_10_V_L = [1.42, 1.59, 1.67, 1.84, 2.54, 2.83, 4.03]

r_1_V_P = [0.25, 0.295, 0.35, 0.372, 0.412, 0.42, 0.425]
t_1_V_P = [1.33, 1.45, 1.591, 1.6981, 2.31, 2.73, 3.52]
r_10_V_P = [0.29, 0.365, 0.45, 0.482, 0.502, 0.512, 0.514]
t_10_V_P = [1.33, 1.45, 1.591, 1.6981, 2.31, 2.73, 3.52]

r_1_V_V_L_OP = [0.238, 0.311, 0.371, 0.3892, 0.432, 0.445, 0.4521]
t_1_V_V_L_OP = [1.23, 1.32, 1.391, 1.481, 1.591, 1.93, 3.52]
r_10_V_V_L_OP = [0.33, 0.396, 0.468, 0.530, 0.545, 0.562, 0.568]
t_10_V_V_L_OP = [1.23, 1.32, 1.391, 1.481, 1.591, 1.93, 3.52]

r_1_V_V_L_OP_PO = [0.248, 0.331, 0.361, 0.4192, 0.439, 0.455, 0.468]
t_1_V_V_L_OP_PO = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]
r_10_V_V_L_OP_PO = [0.338, 0.416, 0.458, 0.5650, 0.5705, 0.578, 0.582]
t_10_V_V_L_OP_PO = [1.13, 1.25, 1.341, 1.481, 1.591, 1.93, 3.52]

'''
plt.plot(t_10_V_L, r_10_V_L, label = "IVFADC", color = "black", marker = '.', linestyle = ':')
plt.plot(t_10_V_P, r_10_V_P, label = "IMI", color = "black", marker = ',', linestyle = '--')
plt.plot(t_10_V_V_L_OP, r_10_V_V_L_OP, label = "VQ VQ LQ_OP", color = "black", marker = 'o', linestyle = '-.')
plt.plot(t_10_V_V_L_OP_PO, r_10_V_V_L_OP_PO, label = "VQ VQ LQ_OP PO", color = 'black', marker = 'v', linestyle = "-")
plt.title("DEEP1B Dataset")
plt.xlabel("Search Time / ms")
plt.ylabel("Recall")
plt.legend()
plt.show()
'''

#SIFT1B dataset 16 bytes
'''
r_1_IVFADC = 
r_10_IVFADC = [0.45, 0.55, 0.59, 0.62, 0.65, 0.661, 0.661]
t_1_IVFADC = [2.42, 2.59, 2.67, 2.84, 3.54, 3.83, 5.03]

r_1_IMI = 
r_10_IMI = [0.37, 0.48, 0.51, 0.537, 0.540, 0.578, 0.582]
t_1_IMI = [2.33, 2.45, 2.591, 2.6981, 3.31, 3.73, 4.52]

r_1_VV = 
r_10_VV = [0.42, 0.57, 0.63, 0.67, 0.68, 0.693, 0.703]
t_1_VV = [2.4, 2.41, 2.59, 2.970, 3.51, 4.86, 5.9]

r_1_VP = 
r_10_VP = [0.45, 0.54, 0.58, 0.652, 0.671, 0.681, 0.694]
t_1_VP = [2.56, 2.73, 2.84, 3.95, 4.32, 4.98, 5.72]

r_1_VVL = 
r_10_VVL = [0.47, 0.59, 0.63, 0.642, 0.672, 0.683, 0.691]
t_1_VVL = [2.2, 2.3, 2.4, 2.5, 2.9, 4.5, 5.42]

r_1_VVP = 
r_10_VVP = [0.41, 0.49, 0.56, 0.58, 0.64, 0.651, 0.66]
t_1_VVP = [2.35, 2.48, 2.57, 2.61, 3.31, 3.69, 4.92]
'''

#Deep1B dataset

r_1_IVFADC = [0.19, 0.27, 0.34, 0.374, 0.4, 0.42, 0.43]
r_10_IVFADC = [0.38, 0.45, 0.57, 0.68, 0.76, 0.77, 0.779]
t_1_IVFADC = [2.42, 2.59, 2.67, 2.84, 3.54, 3.83, 5.03]

r_1_IMI = [0.19, 0.225, 0.30, 0.342, 0.372, 0.392, 0.404]
r_10_IMI = [0.22, 0.41, 0.52, 0.59, 0.62, 0.64, 0.65]
t_1_IMI = [2.33, 2.45, 2.591, 2.6981, 3.31, 3.73, 4.52]

r_1_VV = [0.27, 0.34, 0.443, 0.463, 0.474, 0.487, 0.489]
r_10_VV = [0.31, 0.44, 0.59, 0.74, 0.78, 0.83, 0.84]
t_1_VV = [2.4, 2.41, 2.59, 2.970, 3.51, 4.86, 5.9]

r_1_VP = [0.22, 0.29, 0.383, 0.433, 0.444, 0.455, 0.455]
r_10_VP = [0.32, 0.39, 0.48, 0.59, 0.64, 0.71, 0.75]
t_1_VP = [2.56, 2.73, 2.84, 3.95, 4.32, 4.98, 5.72]

r_1_VVL = [0.23, 0.296, 0.378, 0.426, 0.443, 0.452, 0.458]
r_10_VVL = [0.37, 0.51, 0.64, 0.70, 0.78, 0.81, 0.83]
t_1_VVL = [2.2, 2.3, 2.4, 2.5, 2.9, 4.5, 5.42]

r_1_VVP = [0.18, 0.246, 0.338, 0.40, 0.404, 0.44, 0.448]
r_10_VVP = [0.28, 0.38, 0.51, 0.64, 0.73, 0.76, 0.79]
t_1_VVP = [2.35, 2.78, 2.87, 3.1, 3.61, 3.99, 4.92]

'''
plt.plot(t_1_IVFADC, r_1_IVFADC, label = 'IVFADCPQ', color = "black", marker = '.', linestyle = ':')
plt.plot(t_1_IMI, r_1_IMI, label = 'IMI', color = "black", marker = ',', linestyle = ':')
plt.plot(t_1_VV, r_1_VV, label = 'VQ VQ', color = "black", marker = 'o', linestyle = '-')
plt.plot(t_1_VP, r_1_VP, label = 'VQ PQ', color = "black", marker = 'o', linestyle = '-.')
plt.plot(t_1_VVL, r_1_VVL, label = 'VQ VQ LQ', color = "black", marker = 'v', linestyle = '-')
plt.plot(t_1_VVP, r_1_VVP, label = 'VQ VQ PQ', color = "black", marker = 'v', linestyle = '-.')
plt.title("DEEP1B Dataset")
plt.xlabel("Search Time / ms")
plt.ylabel("Recall@1")
plt.legend()
plt.show()
'''

VQ_no_samp_t = [2.42, 2.59, 2.67, 2.84, 3.54, 3.83, 5.03]
VQ_no_samp_r = [0.41, 0.48, 0.60, 0.69, 0.77, 0.81, 0.839]

VQ_samp_64_t = [2.42, 2.59, 2.67, 2.84, 3.54, 3.83, 5.03]
VQ_samp_64_r = [0.35, 0.38, 0.46, 0.60, 0.71, 0.744, 0.749]

VQ_samp_32_t = [2.42, 2.59, 2.67, 2.84, 3.54, 3.83, 5.03]
VQ_samp_32_r = [0.11, 0.28, 0.39, 0.49, 0.57, 0.66, 0.73]

VV_no_samp_t = [2.4, 2.41, 2.59, 2.970, 3.51, 4.86, 5.9]
VV_no_samp_r = [0.31, 0.44, 0.59, 0.74, 0.78, 0.83, 0.84]

VV_samp_128_t = [2.13, 2.51, 2.79, 2.970, 3.71, 4.96, 6.2]
VV_samp_128_r = [0.276, 0.41, 0.542, 0.68, 0.73, 0.78, 0.82]

VV_samp_64_t = [2.49, 2.58, 2.73, 3.170, 3.81, 4.96, 6.1]
VV_samp_64_r = [0.235, 0.379, 0.497, 0.576, 0.671, 0.682, 0.703]

VV_samp_32_t = [1.9, 2.61, 2.88, 2.970, 3.81, 5.06, 5.76]
VV_samp_32_r = [0.16, 0.25, 0.36, 0.47, 0.59, 0.64, 0.66]

VVV_no_samp_t = [2.2, 2.3, 2.4, 2.5, 2.9, 4.5, 5.42]
VVV_no_samp_r = [0.37, 0.51, 0.64, 0.70, 0.78, 0.81, 0.83]

VVV_samp_128_t = [2.98, 3.1, 3.25, 3.78, 3.9, 4.8, 5.72]
VVV_samp_128_r = [0.245, 0.39, 0.56, 0.64, 0.73, 0.77, 0.79]

VVV_samp_64_t = [1.89, 2.05, 2.3, 2.9, 3.4, 4.04, 5.97]
VVV_samp_64_r = [0.21, 0.38, 0.47, 0.54, 0.62, 0.69, 0.74]

VVV_samp_32_t= [2.2, 2.37, 2.57, 2.96, 3.67, 4.34, 5.97]
VVV_samp_32_r = [0.14, 0.27, 0.39, 0.51, 0.62, 0.66, 0.71]

plt.plot(VQ_no_samp_t, VQ_no_samp_r, label = 'VQ no sampling', color = "black", marker = '.', linestyle = '-')
plt.plot(VQ_samp_64_t, VQ_samp_64_r, label = 'VQ sampling 64', color = "black", marker = '.', linestyle = '-.')
plt.plot(VQ_samp_32_t, VQ_samp_32_r, label = 'VQ sampling 32', color = "black", marker = '.', linestyle = ':')

plt.plot(VV_no_samp_t, VV_no_samp_r, label = 'VV no sampling', color = "black", marker = ',', linestyle = '-')
plt.plot(VV_samp_128_t, VV_samp_128_r, label = 'VV sampling 128', color = "black", marker = ',', linestyle = '--')
plt.plot(VV_samp_64_t, VV_samp_64_r, label = 'VV sampling 64', color = "black", marker = ',', linestyle = '-.')
plt.plot(VV_samp_32_t, VV_samp_32_r, label = 'VV sampling 32', color = "black", marker = ',', linestyle = ':')

plt.plot(VVV_no_samp_t, VVV_no_samp_r, label = 'VVV no sampling', color = "black", marker = 'o', linestyle = '-')
plt.plot(VVV_samp_128_t, VVV_samp_128_r, label = 'VVV sampling 128', color = "black", marker = 'o', linestyle = '--')
plt.plot(VVV_samp_64_t, VVV_samp_64_r, label = 'VVV sampling 64', color = "black", marker = 'o', linestyle = '-.')
plt.plot(VVV_samp_32_t, VVV_samp_32_r, label = 'VVV sampling 32', color = "black", marker = 'o', linestyle = ':')

plt.title("SIFT1B Dataset")
plt.xlabel("Search Time / ms")
plt.ylabel("Recall@10")
plt.legend()
plt.show()