import numpy as np 
import matplotlib.pyplot as plt

dataset = "DEEP1M"
title = dataset + " / Centroids - Search Time"

legends = ["1", "5", "10", "50", "80", "100"]
recall_performance = [1, 5, 10, 50, 80, 100]
filenames = ["/reasoning_models_VQ_VQ_10_50_10_50.txt", "/reasoning_models_VQ_VQ_10_50_60_100.txt", 
            "/reasoning_models_VQ_VQ_60_100_10_50.txt", "/reasoning_models_VQ_VQ_60_100_60_100.txt"]

filepaths = ["/home/yujian/Desktop/Recording_Files/VQ_VQ/" + dataset + filenames[i] for i in range(len(filenames))]
distance_ratio = 400
x_centroids = []
x_visited_vectors = []
avg_x_visited_vectors = []
position = 0
query_start = False
recording_position = []
recording_visited_vectors = []
first_visited_centroids = []
second_visited_centroids = []
avg_second_visited_centroids = []

plt.figure()

for filepath in filepaths:
    
    file = open(filepath, "r")
    f1 = file.readlines()

    for x in f1:
        position = position + 1
        if "n centroids:" in x:
            centroids = x.split("n centroids:")[-1].split(" ")[1:3]
            x_axis = centroids[0] + " " + centroids[1]
            first_visited_centroids.append(float(centroids[0]))
            x_centroids.append(x_axis)
        
        if "Q:" in x:
            position = 0
            query_start = True
        
        if query_start:
            if position == 4:
                recording_position = [-1] * len(recall_performance)
                second_visited = list(map(float, x.split(" ")[0:-1]))
                sum_visited = 0
                for i in range(len(second_visited)):
                    sum_visited = second_visited[i]
                    for j in range (len(recall_performance)):
                        if recording_position[j] == -1 and sum_visited >= recall_performance[j]:
                            recording_position[j] = i



            if position == 5:
                query_start = False
                second_visited_vectors = list(map(float, x.split(" ")[0:-1]))
                for i in range(len(recall_performance)):
                    assert(recording_position[i] >= 0)
                    second_visited_centroids.append(recording_position[i] + 1)
                    x_visited_vectors.append(second_visited_vectors[recording_position[i]])

        if "Finished result analysis:" in x:
            assert(len(x_visited_vectors) == 1000 * len(recall_performance))
            for i in range(len(recall_performance)):
                avg_visited_vectors = np.mean([x_visited_vectors[temp * len(recall_performance) + i] for temp in range(1000)])
                avg_x_visited_vectors.append(avg_visited_vectors)

                avg_second_visited_centroid = np.mean([second_visited_centroids[temp * len(recall_performance) + i] for temp in range(1000)])
                avg_second_visited_centroids.append(avg_second_visited_centroid)
            x_visited_vectors = []
            second_visited_centroids = []


print(x_centroids, x_centroids[0].split(" "))
sum_centroids = np.array([float(x_centroids[i].split(" ")[0]) * float(x_centroids[i].split(" ")[1]) for i in range(len(x_centroids))])
print(sum_centroids)
inds = (sum_centroids).argsort()

for i in range(len(recall_performance)):

    recall_visited_performance = np.array([avg_x_visited_vectors[temp * len(recall_performance) + i] for temp in range(len(x_centroids))])
    second_visited_centroids_num = np.array([avg_second_visited_centroids[temp * len(recall_performance) + i] for temp in range(len(x_centroids))])
    search_time_performance = recall_visited_performance + distance_ratio * (np.array(first_visited_centroids) + second_visited_centroids_num)
    sorted_x_centroids = np.array(x_centroids)[inds]
    sorted_recall_visited_performance = recall_visited_performance[inds]
    search_time_performance = search_time_performance[inds]
    plt.plot(list(sorted_x_centroids), list(search_time_performance))


#my_y_ticks = np.arange(0, 120000, 10000)
plt.xlabel("Centroid Setting")
plt.ylabel("Search Time")
#plt.yticks(my_y_ticks)
plt.xticks(rotation=60)
plt.legend(legends)
plt.title(title)
plt.show()


            



                



            
