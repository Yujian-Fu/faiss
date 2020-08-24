import numpy as np 
import networkx as nx
import faiss 
import os
import utils
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew

k_result = 5
path_list = utils.get_dataset_path_gaussian()
feature_path = "/home/y/yujianfu/ivf-hnsw/data/analysis/LID_entropy_record_gaussian.txt"
feature_file = open(feature_path, "w")


feature_file.write("Name   LID   Entropy\n")
for dataset_path in path_list:
    print("Loading dataset from ", dataset_path)
    dataset = utils.fvecs_read(dataset_path)
    print("Computing LID and entropy")
    entropy = utils.compute_entropy(dataset)
    LID = utils.compute_LID(dataset)
    dataset_name = dataset_path.split(".")[0].split("/")[-1]
    print(dataset_name + "   " + str(LID) + "   " + str(entropy) + "\n")
    feature_file.write(dataset_name + "   " + str(LID) + "   " + str(entropy) + "\n")


'''
property_string = "Name   dist_kurtosis   dist_skew   AC   DSC   DC_mean   DC_median   DC_std   TS\n"
feature_file.write(property_string)

for dataset_path in path_list:
    #Load the dataset
    dataset_name = dataset_path.split(".")[0].split("/")[-1]
    dist_path = dataset_path.split(".")[0] + "_dist.npy"
    label_path = dataset_path.split(".")[0] + "_labels.npy"


    print("Load dataset from ", dataset_path)
    dataset = utils.fvecs_read(dataset_path)
    dimension = dataset.shape[1]

    #Compute the NN graph
    print("Computing the neighbors")
    if os.path.exists(dist_path):
        distance = np.load(dist_path)
        labels = np.load(label_path)

    else:
        index = faiss.IndexFlatL2(dimension)
        index.add(dataset)
        distance, labels = index.search(dataset, k_result + 1)
        np.save(dist_path, distance)
        np.save(label_path, labels)

    
    #Draw the distance distribution of the NN graph
    
    print("Computing the skewness and the kurtosis")
    x = []
    for i in range(dataset.shape[0]):
        for j in range(k_result):
            x.append(distance[i][j + 1])
    x = np.array(x)
    dist_kurtosis = kurtosis(x)
    dist_skew = skew(x)

    print("Generating the figure")
    sns.kdeplot(x, shade=True)
    sns.rugplot(x)
    plt.xlabel("Distance")
    plt.ylabel("Proportion")
    plt.title(dataset_name)
    fig_path = dataset_path.split(".")[0] + "_distance.png"
    plt.savefig(fig_path)

    
    #Compute the LID and entropy of the dataset
    
    #print("Computing the LID and entropy")
    #LID = utils.compute_LID(dataset)
    #entropy = utils.compute_entropy(dataset)

    #Analyze the property
    print("Generating Directed Graph")
    DG = nx.DiGraph()
    DG.add_nodes_from(range(dataset.shape[0]))

    for i in range(dataset.shape[0]):
        for j in range(k_result):
            DG.add_weighted_edges_from([(i, labels[i][j + 1], distance[i][j + 1])])


    
    #G = nx.Graph()
    #G.add_nodes_from(range(dataset.shape[0]))

    #for i in range(dataset.shape[0]):
    #    for j in range(k_result):
    #        G.add_edges_from([(i, labels[j + 1])])
    

    
    #Properties from networkX
    #Re-check the requirement of those features 
    

    print("Computing the graph features")
    AC = nx.average_clustering(DG)

    DSC = nx.degree_assortativity_coefficient(DG)


    DC = np.array(list(nx.degree_centrality(DG)))
    DC_mean = np.mean(DC)
    DC_median = np.median(DC)
    DC_std = np.std(DC)

    #IDC = nx.in_degree_centrality(DG)

    #ODC = nx.out_degree_centrality(DG)

    TS = nx.transitivity(DG)

    #DT = nx.diameter(DG)

    #RD = nx.radius(DG)

    #Not implemented for directed type
    #GE = nx.global_efficiency(DG)

    #Only support when all nodes are connected
    #ASPL = nx.average_shortest_path_length(DG)

    #Not implemented for directed type
    #SIGMA = nx.sigma(DG)

    #Not implemented for directed type
    #OMEGA = nx.omega(DG)

    result_string = dataset_name +"   "+str(dist_kurtosis)+"   "+str(dist_skew)+"   "+str(AC)+"   "+str(DSC)+"   "+str(DC_mean)+"   "+str(DC_median)+"   "+str(DC_std)+"   "+str(TS)+"\n"
    print(result_string)
    feature_file.write(result_string)
'''














