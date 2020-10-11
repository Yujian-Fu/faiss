import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans


points = np.random.rand(100, 2)
center = KMeans(n_clusters=10)
center.fit(points)
centroids = center.cluster_centers_
vor = Voronoi(centroids)
voronoi_plot_2d(vor)

plt.show()

