#  Importing Required Libraries
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

#  Function to read a single file
def read_input_file(file_path): # defines a function to open and read one file.
    all_points = [] # store all 4 values from each node in a list like: [x, y, z, intensity]
    with open(file_path, 'r') as file: # opens the file in read mode ('r')
        lines = file.readlines() # reads all lines into a list

    i = 0 # Start reading from the first line.
    while i < len(lines): # loop iterates through each line of the file.
        line = lines[i].strip() #.strip() removes any extra whitespace or newline characters.
        if line.startswith("Node:"): # Looks for a line starting with "Node:", which indicates a new point.
            try:                    # Reads the next 4 lines after "Node:" Each line contains a single number: x, y, z, or intensity.
                x = float(lines[i + 1].strip())
                y = float(lines[i + 2].strip())
                z = float(lines[i + 3].strip())
                intensity = float(lines[i + 4].strip())
                all_points.append([x, y, z, intensity]) # Puts [x, y, z, intensity] into the all_points list.
                i += 5                               # Skip to the next Node: (5 lines later).
            except (ValueError, IndexError):
                i += 1                              # Skips to the next line safely.
        else:
            i += 1                                 # If current line isn’t "Node:", just move to the next one.
    return all_points

#  Function to process all files and combine data.
def read_all_files(input_folder):
    filenames = [
        "LLL_datapoints.exnode",
        "LUL_datapoints.exnode",
        "RLL_datapoints.exnode",
        "RML_datapoints.exnode",
        "RUL_datapoints.exnode"
    ]
    all_data = []   # will store all points from all files in a single list.
    for fname in filenames:  # loop processes each filename one by one.
        path = os.path.join(input_folder, fname)  # Builds the full file path like.
        print(f"Reading file: {fname}")  # shows which file is being read. 
        points = read_input_file(path)
        print(f" -> Loaded {len(points)} points")  # shows how many points it loaded.
        all_data.extend(points)  # adds all the points from this file to the master list.
    return np.array(all_data)   # Converts the list of lists into a NumPy array for plotting.

#  Function to visualize in 3D
def visualize_points(points, save_path="All_Lobes_3D.png"):  # Function to display the points in 3D color plot.
    if points.ndim != 2 or points.shape[1] != 4:  # Checks that the points array is 2D with 4 columns.
        raise ValueError("Expected a 2D array with 4 columns (x, y, z, intensity)")
    
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]  # Extracts each column of the NumPy array.

    fig = plt.figure(figsize=(10, 7))  # Sets up a 3D plot window.
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=intensity, cmap='hot', s=10, alpha=0.8) # Plots each point in space.

    # Sets labels and title.
    ax.set_xlabel("X Coordinate") 
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    fig.colorbar(scatter, label=" Intensity")  # Shows colorbar that matches intensity to color.
    plt.title("3D Visualization of the Points from All Lobes")  # Displays the 3D plot.
    plt.savefig(save_path)
    plt.show()



#  Main execution
if __name__ == "__main__":
    input_folder = "C:/Users/New folder/Code_task_Aref/inputs" # Sets my folder path to load data from.
    all_points = read_all_files(input_folder) # Calls the read function to get all data.
    
    print(f"Total Combined Points: {len(all_points)}") # Prints how many points were loaded from the files.
    if all_points.size > 0:
        visualize_points(all_points, save_path="All_Lobes_3D.png") # If points were successfully loaded, it shows the 3D plot.
    else:
        print("No points loaded. Check file paths or contents.")
