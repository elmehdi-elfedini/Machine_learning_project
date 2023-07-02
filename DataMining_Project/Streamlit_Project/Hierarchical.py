import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
plt.style.use("fivethirtyeight")


# Define your data for clustering (replace with your actual data)
X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]]
              )

# Perform hierarchical clustering
Z = linkage(X, method='ward')
# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


