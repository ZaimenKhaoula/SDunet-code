import numpy as np
import matplotlib.pyplot as plt
import generateLayoutMatrix


def generate_line_of_sight(matrix):
    tx_position = (200, 200)
    line_of_sight = np.zeros_like(matrix)
    # Mark obstacle cells directly
    line_of_sight[matrix !=0] = 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0 and (i != tx_position[0] or j != tx_position[1]):
                x, y = tx_position
                dx = i - x
                dy = j - y
                steps = max(abs(dx), abs(dy))
                step_x = dx / steps
                step_y = dy / steps
                obstacle_hit = False

                for step in range(steps + 1):
                    if matrix[int(round(x)), int(round(y))] != 0:
                        obstacle_hit = True
                        break
                    x += step_x
                    y += step_y

                if not obstacle_hit:
                    line_of_sight[i, j] = 1
    indices = np.where(line_of_sight == 2)
    line_of_sight[indices] = 0

    return line_of_sight



obstacle_matrix = generateLayoutMatrix.generate_obstacle_matrix(generateLayoutMatrix.complexe_t24_obstacles_list)
generateLayoutMatrix.visualize_matrix(obstacle_matrix)
import os

folder_path = 'C:\/Users\/ZAIMEN\/PycharmProjects\/GenerateCNNinput\/computedTensors\/complexe_t24'
# Generate line of sight matrix
line_of_sight_matrix = generate_line_of_sight(obstacle_matrix)
file_path = os.path.join(folder_path, 'Los.csv')
#np.savetxt(file_path, line_of_sight_matrix, delimiter=',', fmt='%f' )
print(line_of_sight_matrix.shape)
generateLayoutMatrix.visualize_matrix(line_of_sight_matrix)
"""
# Visualize the matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original matrix
axes[0].imshow(matrix, cmap='gray', interpolation='none')
axes[0].set_title('Original Matrix')

# Plot the line of sight matrix
axes[1].imshow(line_of_sight_matrix, cmap='viridis', interpolation='none')
axes[1].set_title('Line of Sight Matrix')

plt.tight_layout()
plt.show()
"""