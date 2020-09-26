import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math

filename = "/home/yujian/Downloads/similarity_search_datasets/models_VQ/sift1M/recording_"

time = []
recall = []

plt.figure()
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
index = 0
for i in range(500, 3600, 200):
    filepath = filename + str(i) + "qps.txt"
    file = open(filepath, "r")

    f1 = file.readlines()
    print(filepath)
    for x in f1:
        if "Finish SearchThe time usage: " in x:
            time.append(float(x.split(" ")[-3]))
        
        if "The recall@100" in x:
            recall.append(float(x.split(" ")[-1]))
    
    time_inds = np.array(time).argsort()
    recall_inds = np.array(recall).argsort()

    print(time)
    plt.plot(np.array(time)[time_inds], np.array(recall)[recall_inds], label = str(i), color = all_colors[index])
    index += 1
    time = []
    recall = []

plt.legend()
plt.xlabel("Latency / ms")
plt.ylabel("Recall@100")
plt.title("Latency - Recall")
plt.show()



