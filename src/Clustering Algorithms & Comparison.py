#  Importing Required Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler 

#  Function to read a single file
def read_input_file(file_path):
    points = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Node:"):
            try:
                x = float(lines[i+1].strip())
                y = float(lines[i+2].strip())
                z = float(lines[i+3].strip())
                intensity = float(lines[i+4].strip())
                points.append([x, y, z, intensity])
                i += 5
            except:
                i += 1
        else:
            i += 1
    return points

# Function to process all files and combine data.
def load_all_data(folder_path):
    filenames = [
        "LLL_datapoints.exnode",
        "LUL_datapoints.exnode",
        "RLL_datapoints.exnode",
        "RML_datapoints.exnode",
        "RUL_datapoints.exnode"
    ]
    all_data = []
    for fname in filenames:
        file_path = os.path.join(folder_path, fname)
        all_data.extend(read_input_file(file_path))
    return np.array(all_data)

# Set my folder path here:
folder_path = "C:/Users/rahim/New folder/Code_task_Aref/inputs"
all_data_np = load_all_data(folder_path)  # Loads all data points into a NumPy array: all_data_np.

# Normalize data : Ensures x, y, z, and intensity are all treated equally.
scaler = StandardScaler()
features = scaler.fit_transform(all_data_np)

# Kmeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(features)

# DBSCAN Parameter Tuning
print("\n-> DBSCAN Parameter Tuning:")
best_eps = 0
best_clusters = 0

for eps in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    db = DBSCAN(eps=eps, min_samples=10).fit(features)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  eps={eps:.1f} -> Clusters: {n_clusters}, Noise Points: {n_noise}")
    if n_clusters > best_clusters:
        best_clusters = n_clusters
        best_eps = eps

print(f" Best eps found: {best_eps} with {best_clusters} clusters")

# DBSCAN clustering
dbscan = DBSCAN(best_eps, min_samples=10)
dbscan_labels = dbscan.fit_predict(features)

# Agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=5)
agglo_labels = agglo.fit_predict(features)

# Visualize clusters in 3D
def plot_clusters(data, labels, title, filename): 
    fig = plt.figure(figsize=(10, 7))     
    ax = fig.add_subplot(111, projection='3d')  
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab20', s=10, alpha=0.8)
    # Plots each point at (x, y, z).
    ax.set_xlabel('X Coordinate')  
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(title)
    plt.colorbar(scatter, label="Cluster") # Adds labels and colorbar.
    plt.savefig(filename) # saves the figure to a PNG file.
    plt.show()

# Run and Show Result
plot_clusters(all_data_np, kmeans_labels, "KMeans Clustering (k=5)", "KMeans_Clusters.png")
plot_clusters(all_data_np, dbscan_labels, "DBSCAN Clustering", "DBSCAN_Clusters.png")
plot_clusters(all_data_np, agglo_labels, "Agglomerative Clustering (k=5)", "Agglomerative_Clusters.png")

# Visual Comparison of Clustering Algorithms (3D Plot)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'}) # creates a row of 3 subplots (side-by-side).
titles = ['KMeans Clustering (k=5)', 'DBSCAN Clustering', 'Agglomerative Clustering (k=5)'] # Stores the titles for each of the 3 subplots in a list.
label_sets = [kmeans_labels, dbscan_labels, agglo_labels] # This list contains the cluster labels from each algorithm.

for ax, labels, title in zip(axes, label_sets, titles): # groups each axis, its labels, and its title.
    ax.scatter(all_data_np[:, 0], all_data_np[:, 1], all_data_np[:, 2], c=labels, cmap='tab20', s=3, alpha=0.7)
    # Plots all points as 3D scatter dots.
    ax.set_title(title) # Adds the specific clustering title.
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

plt.suptitle("Visual Comparison of KMeans, DBSCAN, and Agglomerative Clustering", fontsize=16) # Adds one big title above all three subplots.
plt.tight_layout() # Automatically adjusts spacing so plots don’t overlap or get cut off.
plt.savefig("Visual_Comparison_Clusters_3D.png") 
plt.show()

# Print summary: Shows how many unique clusters were created by each algorithms.
print("-> Clustering Summary:")
print(f"KMeans Clusters: {len(set(kmeans_labels))}")
print(f"DBSCAN Clusters (excluding noise): {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
print(f"DBSCAN Noise Points: {list(dbscan_labels).count(-1)}")
print(f"Agglomerative Clusters: {len(set(agglo_labels))}")
