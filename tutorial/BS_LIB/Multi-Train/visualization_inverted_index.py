import matplotlib.pyplot as plt 
from visualization import *

record_file = ""


distance_line = False
recall1 = False
recall10 = False
recall100 = False
distance = [0,0]
Kmeans_distance_result = []
pq_distance_result = []
centroid_result = []
recall_1 = 0
recall_10 = 0
recall_100 = 0
recall1_result = []
recall10_result = []
recall100_result = []
record_file = "./inverted_index/record/inverted_index_PQSun Nov 29 13%3A19%3A38 2020.txt"
#get_newest_folder("./inverted_index/record", "inverted_index_PQSun")

print(record_file)
with open(record_file , 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "Kmeans with centroids: " in line:
            recall100 = False
            if recall_100 > 0:
                recall100_result.append(recall_100)
            centroid_result.append(int(line.split("Kmeans with centroids: ")[-1]))
        
        if distance_line and "result" not in line:
            distance[0] = float(line.split(" ")[0])
            distance[1] = float(line.split(" ")[1])

        if recall1 and "result" not in line:
            recall_1 = float(line.split(" ")[-2])
        
        if recall10 and "result" not in line:
            recall_10 = float(line.split(" ")[-2])

        if recall100 and "result" not in line:
            recall_100 = float(line.split(" ")[-2])

        if "Distance computation" in line:
            distance_line = True
        
        elif "result for recall@ 1" in line and "result for recall@ 10" not in line:
            Kmeans_distance_result.append(distance[0])
            pq_distance_result.append(distance[1])
            recall1 = True
            distance_line = False
        
        elif "result for recall@ 10" in line and "result for recall@ 100" not in line:
            recall1_result.append(recall_1)
            recall1 = False
            recall10 = True

        elif "result for recall@ 100" in line:
            recall10_result.append(recall_10)
            recall10 = False
            recall100 = True

    recall100_result.append(float(lines[-1].split(" ")[-2]))
    plt.figure()
    print(centroid_result)
    plt.xticks(centroid_result)
    print(recall10_result)
    #plt.plot(centroid_result, recall1_result, label = "recall@1")
    plt.plot(centroid_result, recall10_result, label = "recall@10")
    #plt.plot(centroid_result, recall100_result, label = "recall@100")
    plt.legend()
    plt.show()
    plt.figure()
    plt.xticks(centroid_result)
    plt.plot(centroid_result, pq_distance_result, label = "pq distance")
    plt.legend()
    plt.show()
    plt.figure()
    plt.xticks(centroid_result)
    plt.plot(centroid_result, Kmeans_distance_result, label = "index distance")
    plt.legend()
    plt.show()
    






















