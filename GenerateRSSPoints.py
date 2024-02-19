import os
import numpy as np
folder1_path = "C:\/Users\/ZAIMEN\OneDrive\/Bureau\/resultatsRSS\/resultats_simulation_WI_stage1\/complexe_t21"  # Replace with the path to your first folder
folder2_path = "C:\/Users\/ZAIMEN\OneDrive\/Bureau\/resultatsRSS\/resultats_simulation_WI_stage2\/complexe_t21"  # Replace with the path to your second folder

x_origin=50
y_origin=0


rss_points = []

# Iterate through files in both folders
for file1, file2 in zip(os.listdir(folder1_path), os.listdir(folder2_path)):
    file1_path = os.path.join(folder1_path, file1)
    file2_path = os.path.join(folder2_path, file2)

    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        # Read the last lines of each file
        lines_file1 = f1.readlines()
        lines_file2 = f2.readlines()

        # Extract x, y coordinates from the last line of file1
        coordinates = lines_file1[-1].strip().split()  # Assuming x is at position 1 and y is at position 2
        x_coord = float(coordinates[1])  # Assuming x is at position 1
        y_coord = float(coordinates[2])  # Assuming y is at position 2
        # Extract rss value from the last line of file2
        rss_value = float(lines_file2[-1].strip().split()[5])  # Assuming rss is the last value

        # Append x, y, rss to rss_points
        rss_points.append([x_coord - x_origin, y_coord - y_origin, rss_value])

# Convert rss_points to a NumPy array
##rss_points = np.array(rss_points)

# Print or use the rss_points array as needed
print("RSS Points Array:")
print(list(rss_points))
