import numpy as np 
import matplotlib.pyplot as plt

# Tutorial
'''
x = np.linspace(-1, 1, 50)
y = 2 * x + 1

plt.figure()
plt.plot(x, y)
plt.show()
'''
'''
basepath = "/home/yujian/Desktop/Recording_Files/VQ/SIFT1M/"
filename = [ "recording_500.txt","recording_1000.txt", "recording_2000.txt", "recording_3000.txt", "recording_4000.txt", "recording_5000.txt"]
legends = ["500", "1000", "2000", "3000", "4000", "5000"]
'''
'''
basepath = "/home/yujian/Desktop/Recording_Files/VQ_VQ/SIFT1M/"
filename = [ "recording_10_100.txt","recording_20_100.txt", "recording_30_100.txt", 
"recording_40_100.txt", "recording_50_100.txt", "recording_100_10.txt", "recording_100_20.txt", 
"recording_100_30.txt", "recording_100_40.txt", "recording_100_50.txt",]
legends = ["10_100", "20_100", "30_100", "40_100", "50_100", "100_10", "100_20", "100_30", "100_40", "100_50"]
'''
'''
basepath = "/home/yujian/Desktop/Recording_Files/VQ_LQ/SIFT1M/"
filename = [ "recording_100_10.txt",  "recording_100_20.txt",  "recording_200_10.txt",  "recording_200_20.txt",  "recording_250_20.txt",  "recording_400_10.txt"]
legends = ["100_10", "100_20", "200_10", "200_20", "250_20", "400_10"]
'''

title = "SIFT1M / VQ_PQ"
metric = "recall@1"

search_time_signal_string = "Finish SearchThe time usage: "
recall_signal_string = "The " + metric + " for 1000 queries in parallel mode is: "



plt.figure()
for i in range(len(filename)):
    filepath = basepath + filename[i]
    file = open(filepath, "r")
    f1 = file.readlines()
    time_sequence = []
    recall_sequence = []
    time = ""
    recall = ""
    for x in f1:
        if search_time_signal_string in x:
            time = x.split(search_time_signal_string)[1].split(" ")[0]
            
        if recall_signal_string in x:
            recall = x.split(recall_signal_string)[1].split('\n')[0]
            print("time: ", time, "recall: ", recall)
            time_sequence.append(float(time))
            recall_sequence.append(float(recall))

    time_sequence.sort()
    plt.plot(time_sequence, recall_sequence)
    
plt.legend(legends)
plt.xlabel("Search Time / ms")
plt.ylabel(metric)
plt.title(title)
plt.show()
        


'''
x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y1)
plt.show()

plt.figure(num=3, figsize=(8, 5),)
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.show()
'''


