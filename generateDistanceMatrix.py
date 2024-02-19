import numpy as np

# Size of the matrix
matrix_size = 400

# Creating a matrix of zeros
matrix = np.zeros((matrix_size, matrix_size))

# Coordinates of the center
center_x, center_y = matrix_size // 2, matrix_size // 2

# Calculating distances for each cell
for i in range(matrix_size):
    for j in range(matrix_size):
        distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
        matrix[i, j] = distance

import matplotlib.pyplot as plt

def visualize_matrix(matrix):
    plt.figure(figsize=(8, 8))  # Adjust the figure size if needed
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Indoor Environment')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    #plt.grid(visible=True, color='black', linestyle='-', linewidth=0.5)
    plt.grid(visible=False)
    plt.show()

visualize_matrix(matrix)

import os
# Printing or using the resulting matrix
#print(matrix)
folder_path = 'C:\/Users\/ZAIMEN\/PycharmProjects\/GenerateCNNinput\/computedTensors\/complexe_t24'
file_path = os.path.join(folder_path, 'distance.csv')
#np.savetxt(file_path, matrix, delimiter=',', fmt='%f' )