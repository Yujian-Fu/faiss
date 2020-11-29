import matplotlib.pyplot as plt 

inverted_index_file = "./kmeans/record/kmeans_PQSun_Nov_29_13_16_37_2020.txt"
kmeans_file = "./kmeans/record/kmeans_PQSun_Nov_29_13_35_12_2020.txt"

name1 = "inverted index"
name2 = "kmeans"

recall1 = False
recall10 = False
recall100 = False

centroid_result1 = []
centroid_result2 = []

visit_1_result1 = []
visit_10_result1 = []
visit_100_result1 = []

recall1_result1 = []
recall10_result1 = []
recall100_result1 = []

visit_1_result2 = []
visit_10_result2 = []
visit_100_result2 = []

recall1_result2 = []
recall10_result2 = []
recall100_result2 = []

index = -1

with open(inverted_index_file, 'r') as f1:
    with open(kmeans_file, 'r') as f2:
        rl1 = f1.readlines()
        rl2 = f2.readlines()
        
        for idx, rl in enumerate([rl1, rl2]):
            if idx == 0:
                visit_1_result = visit_1_result1
                visit_10_result = visit_10_result1
                visit_100_result = visit_100_result1

                recall1_result = recall1_result1
                recall10_result = recall10_result1
                recall100_result = recall100_result1
                centroid_result = centroid_result1
            else:
                visit_1_result = visit_1_result2
                visit_10_result = visit_10_result2
                visit_100_result = visit_100_result2

                recall1_result = recall1_result2
                recall10_result = recall10_result2
                recall100_result = recall100_result2
                centroid_result = centroid_result2


            for line in rl:
                if "Kmeans with centroids: " in line:
                    centroid_result.append(int(line.split("Kmeans with centroids: ")[-1]))
            
                if recall1 and index == 1:
                    recall1_result.append(float(x) for x in line.split(" ")[0:-1])
                    index = -1
                    recall1 = False

                if recall1 and index == 0:
                    visit_1_result.append(float(x) for x in line.split(" ")[0:-1])
                    index = 1
                
                if recall10 and index == 1:
                    recall10_result.append(float(x) for x in line.split(" ")[0:-1])
                    index = -1
                    recall10 = False

                if recall10 and index == 0:
                    visit_10_result.append(float(x) for x in line.split(" ")[0:-1])
                    index = 1
                
                if recall100 and index == 1:
                    recall100_result.append(float(x) for x in line.split(" ")[0:-1])
                    index = -1
                    recall100 = False

                if recall100 and index == 0:
                    visit_100_result.append(float(x) for x in line.split(" ")[0:-1])
                    index = 1
                

                
                if "result for recall@ 1" in line and "result for recall@ 10" not in line:
                    recall1 = True
                    index = 0
                
                if "result for recall@ 10" in line and "result for recall@ 100" not in line:
                    index = 0
                    recall10 = True
                
                if "result for recall@ 100" in line:
                    index = 0
                    recall100 = True

for idx, centroid in enumerate(centroid_result1):
    assert(centroid_result1[idx] == centroid_result2[idx])
    plt.figure()

    plt.semilogx(list(visit_1_result1[idx]), list(recall1_result1[idx]), label = name1 + " recall 1")
    plt.semilogx(list(visit_1_result2[idx]), list(recall1_result2[idx]), label = name2 + " recall 1")
    #plt.xlim((0, 300000))

    plt.title("centroid: " + str(centroid))
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogx(list(visit_10_result1[idx]), list(recall10_result1[idx]), label = name1 + " recall 10")
    plt.semilogx(list(visit_10_result2[idx]), list(recall10_result2[idx]), label = name2 + " recall 10")
    #plt.xlim((0, 300000))
    plt.title("centroid: " + str(centroid))
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogx(list(visit_100_result1[idx]), list(recall100_result1[idx]), label = name1 + " recall 100")
    plt.semilogx(list(visit_100_result2[idx]), list(recall100_result2[idx]), label = name2 + " recall 100")
    #plt.xlim((0, 300000))
    plt.title("centroid: " + str(centroid))
    plt.legend()
    plt.show()


