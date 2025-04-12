README.txt

Project: Lung CT Data Visualization and Clustering

Files:
1. main.py
   - Reads CT scan data points (x, y, z, intensity) from `.exnode` files.
   - Visualizes the 3D distribution of all points using color-coded intensity.

2. Clustering Algorithms & Comparison.py
   - Loads and normalizes the same CT data.
   - Applies three clustering algorithms: KMeans, DBSCAN, and Agglomerative Clustering.
   - Displays and saves 3D plots of each clustering result, plus a side-by-side comparison.
   - Prints a clustering summary.

Usage:
- Make sure the input folder contains the 5 `.exnode` files: LLL, LUL, RLL, RML, and RUL.
- Update the `folder_path` or `input_folder` in each script with your actual directory path.
- Run `main.py` to view data, or `Clustering Algorithms & Comparison.py` to perform and visualize clustering.
