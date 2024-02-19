import numpy as np
import matplotlib.pyplot as plt
import os


def draw_line(matrix, start_x_cell, start_y_cell, end_x_cell, end_y_cell, value, thickness, matrix_size):
    x1= start_x_cell
    y1 = start_y_cell
    x2, y2 = end_x_cell , end_y_cell
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if x1 < x2:
        sx = 1
    else:
        sx = -1

    if y1 < y2:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while True:
        # Set the primary line
        matrix[y1][x1] = value

        # Set additional thickness above the line
        for i in range(-thickness // 2, thickness // 2 ):
            matrix[max(0, min(y1 + i, len(matrix) - 1))][x1] = value

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx

        if e2 < dx:
            err += dx
            y1 += sy


def draw_line_anc(matrix, start_x_cell, start_y_cell, end_x_cell, end_y_cell, value, thickness):
    x1, y1 = start_x_cell, start_y_cell
    x2, y2 = end_x_cell, end_y_cell
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if x1 < x2:
        sx = 1
    else:
        sx = -1

    if y1 < y2:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while True:
        # Set the primary line
        matrix[y1][x1] = value

        # Set additional thickness above the line
        for i in range(-thickness // 2, thickness // 2-1):
            matrix[max(0, min(y1 + i, len(matrix) - 1))][x1] = value

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx

        if e2 < dx:
            err += dx
            y1 += sy


def generate_obstacle_matrix(obstacles):
    # Create a 400x400 matrix initialized with zeros
    matrix_size = 400
    indoor_matrix = np.zeros((matrix_size, matrix_size))
    # Define constants
    cell_size = 5  # Each cell represents 5cm * 5cm

    # Convert width from centimeters to cell units
    def width_to_cells(width_cm):
        a = int(width_cm / cell_size)
        if a==0:
            a=1
        return a

    # Convert coordinates from Euclidean plane to matrix indices
    def convert_to_matrix_indices(x, y):
        matrix_x = int((100 * x) / cell_size)
        matrix_y = int((100 * y) / cell_size)
        return matrix_size - matrix_y - 1, matrix_x

    # Iterate through each obstacle
    for obstacle in obstacles:
        start_x, start_y = obstacle[0]
        end_x, end_y = obstacle[1]
        width_cm = obstacle[2]
        permittivity = obstacle[3]
        # Convert coordinates from Euclidean plane to matrix indices
        start_x_cell, start_y_cell = convert_to_matrix_indices(start_x, start_y)
        end_x_cell, end_y_cell = convert_to_matrix_indices(end_x, end_y)
        # Ensure coordinates stay within matrix bounds
        start_x_cell = max(min(start_x_cell, matrix_size - 1), 0)
        start_y_cell = max(min(start_y_cell, matrix_size - 1), 0)
        end_x_cell = max(min(end_x_cell, matrix_size - 1), 0)
        end_y_cell = max(min(end_y_cell, matrix_size - 1), 0)
        # Calculate obstacle width in cells
        width_cells = width_to_cells(width_cm)
        # Adjust horizontal line position to fit within matrix boundaries
        if start_y == end_y:
            if start_x_cell - width_cells < 0:
                start_x_cell = width_cells-1

            # Fill the matrix with obstacle permittivity values for horizontal lines
            for i in range(start_x_cell - width_cells+1, start_x_cell+1):
                for j in range(min(start_y_cell, end_y_cell), max(start_y_cell, end_y_cell) + 1):
                    indoor_matrix[i, j] = permittivity
        else:
            if start_x == end_x:  # For vertical lines
                if start_y_cell + width_cells > matrix_size-1:
                    start_y_cell = matrix_size - width_cells - 1
                # Fill the matrix with obstacle permittivity values for vertical lines
                for i in range(min(end_x_cell,start_x_cell) , max(end_x_cell,start_x_cell) + 1):
                    if start_y_cell + width_cells+1 == 400:
                        j_start=start_y_cell+1
                        j_end = start_y_cell + width_cells+1
                    else:
                        j_start = start_y_cell
                        j_end = start_y_cell + width_cells
                    for j in range(j_start, j_end):
                        indoor_matrix[i, j] = permittivity
            else:
                draw_line(indoor_matrix, max(min(start_x*20, matrix_size - 1), 0), max(min(matrix_size-1-start_y*20, matrix_size - 1), 0) , max(min(end_x*20, matrix_size - 1), 0), max(min(matrix_size-1-end_y*20, matrix_size - 1), 0),  permittivity, width_cells,matrix_size)

    return indoor_matrix


def generate_obstacle_matrix_new(obstacles):
    # Create a 400x400 matrix initialized with zeros
    matrix_size = 400
    indoor_matrix = np.zeros((matrix_size, matrix_size))
    # Define constants
    cell_size = 5  # Each cell represents 5cm * 5cm

    # Convert width from centimeters to cell units
    def width_to_cells(width_cm):
        a = int(width_cm / cell_size)
        if a==0:
            a=1
        return a

    # Convert coordinates from Euclidean plane to matrix indices
    def convert_to_matrix_indices(x, y):
        matrix_x = int((100 * x) / cell_size)
        matrix_y = int((100 * y) / cell_size)
        #return matrix_size - matrix_y - 1, matrix_x
        return matrix_size - matrix_y -1, matrix_x

    # Iterate through each obstacle
    for obstacle in obstacles:
        start_x, start_y = obstacle[0]
        end_x, end_y = obstacle[1]
        width_cm = obstacle[2]
        permittivity = obstacle[3]
        # Convert coordinates from Euclidean plane to matrix indices
        start_x_cell, start_y_cell = convert_to_matrix_indices(start_x, start_y)
        end_x_cell, end_y_cell = convert_to_matrix_indices(end_x, end_y)
        # Ensure coordinates stay within matrix bounds
        start_x_cell = max(min(start_x_cell, matrix_size - 1), 0)
        start_y_cell = max(min(start_y_cell, matrix_size - 1), 0)
        end_x_cell = max(min(end_x_cell, matrix_size - 1), 0)
        end_y_cell = max(min(end_y_cell, matrix_size - 1), 0)
        # Calculate obstacle width in cells
        width_cells = width_to_cells(width_cm)
        # Adjust horizontal line position to fit within matrix boundaries
        if start_y == end_y:
            if start_x_cell - width_cells < 0:
                start_x_cell = width_cells-1
            # Fill the matrix with obstacle permittivity values for horizontal lines
            for i in range(start_x_cell - width_cells, start_x_cell):
                for j in range(min(start_y_cell, end_y_cell), max(start_y_cell, end_y_cell) + 1):
                    indoor_matrix[i, j] = permittivity
        else:
            if start_x == end_x:  # For vertical lines
                if start_y_cell + width_cells > matrix_size-1:
                    #start_y_cell = matrix_size - width_cells
                    start_y_cell = matrix_size - width_cells - 1

                # Fill the matrix with obstacle permittivity values for vertical lines
                for i in range(min(end_x_cell,start_x_cell) , max(end_x_cell,start_x_cell) + 1):
                    for j in range(start_y_cell, start_y_cell + width_cells):
                        indoor_matrix[i, j] = permittivity
            else:
                draw_line(indoor_matrix, max(min(start_x*20, matrix_size - 1), 0), max(min(matrix_size-1-start_y*20, matrix_size - 1), 0) , max(min(end_x*20, matrix_size - 1), 0), max(min(matrix_size-1-end_y*20, matrix_size - 1), 0),  permittivity, width_cells,matrix_size)

    return indoor_matrix


# Rest of the code remains the same for visualization...


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



def compress_matrix(matrix, target_size):
    # Calculate compression ratio
    compression_ratio = matrix.shape[0] // target_size

    # Initialize a compressed matrix
    compressed_matrix = np.zeros((target_size, target_size))

    # Iterate over the original matrix and compress
    for i in range(target_size):
        for j in range(target_size):
            # Calculate the average value within the compression window
            window = matrix[i * compression_ratio: (i + 1) * compression_ratio, j * compression_ratio: (j + 1) * compression_ratio]
            compressed_matrix[i, j] = np.mean(window)

    return compressed_matrix

# Example obstacles (format: ((start_x, start_y), (end_x, end_y), width_cm, permittivity))
permittivity_concrete = 7
#permittivity_concrete =0.015
conductivity_concrete = 0.015
thickness_concrete = 30


permittivity_wood = 5
#permittivity_wood = 0
conductivity_wood = 0
thickness_wood = 5



permittivity_glass = 2.4
#permittivity_glass = 0
conductivity_glass =0
thickness_glass =0.5

permittivity_brick = 4.440
#permittivity_brick = 0.01
conductivity_brick = 0.01
thickness_brick = 12.5


permittivity_plaster = 2.5
#permittivity_plaster = 0.03
conductivity_plaster = 0.03
thickness_plaster = 20



#obstacle : starting point, ending point, thikness, permittivity




#complexe-t0
complexe_t0_obstacles_list = [
((0, 0), (5, 0), thickness_plaster, permittivity_plaster),
((5, 0), (6, 0), thickness_glass, permittivity_glass),
((6, 0), (12.5, 0), thickness_plaster, permittivity_plaster),
((12.5, 0), (13.5, 0), thickness_glass, permittivity_glass),
((13.5, 0), (16, 0), thickness_plaster, permittivity_plaster),
((16, 0), (17, 0), thickness_glass, permittivity_glass),
((17, 0), (20, 0), thickness_plaster, permittivity_plaster),
((20, 0), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (15, 20), thickness_plaster, permittivity_plaster),
((14, 20), (15, 20), thickness_glass, permittivity_glass),
((7, 20), (14, 20), thickness_plaster, permittivity_plaster),
((5.5, 20), (7, 20), thickness_glass, permittivity_glass),
((0, 20), (5.5, 20), thickness_plaster, permittivity_plaster),
((0, 0), (0, 7.5), thickness_plaster, permittivity_plaster),
((0, 7.5), (0, 8.5), thickness_glass, permittivity_glass),
((0, 8.5), (0, 16), thickness_plaster, permittivity_plaster),
((0,16), (0, 17), thickness_glass, permittivity_glass),
((0, 17), (0, 20), thickness_plaster, permittivity_plaster),
((3, 0), (3, 4), thickness_plaster, permittivity_plaster),
((2, 4), (6, 4), thickness_plaster, permittivity_plaster),
((10, 0), (10, 4), thickness_plaster, permittivity_plaster),
((10, 4), (14, 4), thickness_plaster, permittivity_plaster),
((16, 4), (20, 4), thickness_plaster, permittivity_plaster),
((0,10), (4, 10), thickness_plaster, permittivity_plaster),
((4, 10), (4, 14), thickness_plaster, permittivity_plaster),
((4, 16), (4, 20), thickness_plaster, permittivity_plaster),
((4, 17), (7, 17), thickness_plaster, permittivity_plaster),
((9, 17), (14,17), thickness_plaster, permittivity_plaster),
((16, 17), (20, 17), thickness_plaster, permittivity_plaster),
((11, 17), (11, 20), thickness_plaster, permittivity_plaster),

]

complexe_t1_obstacles_list = [
    ((0, 0), (20, 0), thickness_plaster, permittivity_plaster),
    ((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
    ((20, 0), (20, 4), thickness_plaster, permittivity_plaster),
    ((0, 0), (5, 0), thickness_plaster, permittivity_plaster),
    ((20, 5), (20, 20), thickness_plaster, permittivity_plaster),
    ((0, 2.5), (0, 0), thickness_plaster, permittivity_plaster),
    ((0, 4), (0, 11.5), thickness_plaster, permittivity_plaster),
    ((0, 12.5), (0, 17.5), thickness_plaster, permittivity_plaster),
    ((0, 18.5), (0, 20), thickness_plaster, permittivity_plaster),
    ((0,2.5), (0, 4), thickness_glass, permittivity_glass),
    ((0,11.5), (0, 12.5), thickness_glass, permittivity_glass),
    ((0,17.5), (0, 18.5), thickness_glass, permittivity_glass),
    ((20,4), (20, 5), thickness_glass, permittivity_glass),
    ((0, 6), (3,6), thickness_plaster, permittivity_plaster),
    ((5, 6), (10, 6), thickness_plaster, permittivity_plaster),
    ((10, 6), (10, 0), thickness_plaster, permittivity_plaster),
    ((14, 0), (14, 2), thickness_plaster, permittivity_plaster),
    ((14, 4), (14, 7), thickness_plaster, permittivity_plaster),
    ((14, 7), (16, 7), thickness_plaster, permittivity_plaster),
    ((18, 7), (20, 7), thickness_plaster, permittivity_plaster),
((0, 8), (3, 8), thickness_plaster, permittivity_plaster),
((3, 8), (3, 10), thickness_plaster, permittivity_plaster),
((3, 12), (3, 16), thickness_plaster, permittivity_plaster),
((0, 14), (3, 14), thickness_plaster, permittivity_plaster),
((3, 17), (3, 20), thickness_plaster, permittivity_plaster),
((14, 9), (14, 13), thickness_plaster, permittivity_plaster),
((14, 9), (16.5, 9), thickness_plaster, permittivity_plaster),
((14, 13), (20, 13), thickness_plaster, permittivity_plaster),
((14, 15), (20, 15), thickness_plaster, permittivity_plaster),
((14, 15), (14, 18), thickness_plaster, permittivity_plaster),

]

complexe_t2_obstacles_list=[

((1, 0), (2.5,0 ), thickness_glass, permittivity_glass),
((8, 0), (9.5,0 ), thickness_glass, permittivity_glass),
((3.5, 0), (5, 0), thickness_glass, permittivity_glass),
((11.5,0 ), (13,0 ), thickness_glass, permittivity_glass),
((14.5,0 ), (16, 0), thickness_glass, permittivity_glass),
(( 18, 0), (19, 0), thickness_glass, permittivity_glass),
((0,0 ), (1,0 ), thickness_brick, permittivity_brick),
((2.5, 0), (3.5, 0), thickness_brick, permittivity_brick),
((5, 0), (8, 0), thickness_brick, permittivity_brick),
((9.5, 0), (11.5,0 ), thickness_brick, permittivity_brick),
((16,0 ), (18,0 ), thickness_brick, permittivity_brick),
((19 ,0 ), (20, 0 ), thickness_brick, permittivity_brick),
((13, 0), (14.5, 0), thickness_brick, permittivity_brick),
((0, 1), (0, 2), thickness_glass, permittivity_glass),
((0, 4), (0,5.5), thickness_glass, permittivity_glass),
((0, 10), (0, 12), thickness_glass, permittivity_glass),
((0, 16), (0, 18), thickness_glass, permittivity_glass),
((0, 0), (0, 1), thickness_brick, permittivity_brick),
((0, 2), (0, 4), thickness_brick, permittivity_brick),
((0, 5.5), (0, 10), thickness_brick, permittivity_brick),
((0, 12), (0, 16), thickness_brick, permittivity_brick),
((0, 18), (0, 20), thickness_brick, permittivity_brick),

((6.5, 0), (6.5, 3), thickness_plaster, permittivity_plaster),
((0, 3), (3.5, 3), thickness_plaster, permittivity_plaster),
((5, 3), (12,3 ), thickness_plaster, permittivity_plaster),
(( 13.5, 3), (20, 3), thickness_plaster, permittivity_plaster),
((16, 3), (16, 10), thickness_plaster, permittivity_plaster),
((16, 11.5), (16, 16.5), thickness_plaster, permittivity_plaster),
((16, 18), (16,20 ), thickness_plaster, permittivity_plaster),
((16,18.5 ), (14,18.5 ), thickness_plaster, permittivity_plaster),
((12.5, 18.5), (10.5, 18.5), thickness_plaster, permittivity_plaster),
((10.5, 18.5), (10.5, 20), thickness_plaster, permittivity_plaster),
((8, 18), (8,20 ), thickness_plaster, permittivity_plaster),
((8,16.5 ), (8, 6.5), thickness_plaster, permittivity_plaster),
((8, 6.5), (5, 6.5), thickness_plaster, permittivity_plaster),
((0, 6.5), (3.5, 6.5), thickness_plaster, permittivity_plaster),
((0,15 ), (8, 15), thickness_plaster, permittivity_plaster),
((17.5, 0), (17.5, 2), thickness_plaster, permittivity_plaster),
((20,12.5 ), (16, 12.5), thickness_plaster, permittivity_plaster),

((20, 4), (20, 6), thickness_glass, permittivity_glass),
((20, 9), (20, 11), thickness_glass, permittivity_glass),
((20,16 ), (20, 18), thickness_glass, permittivity_glass),

((20, 0), (20, 4), thickness_brick, permittivity_brick),
((20, 6), (20, 9), thickness_brick, permittivity_brick),
((20, 11), (20, 16), thickness_brick, permittivity_brick),
((20, 18), (20,20 ), thickness_brick, permittivity_brick),

((1.5, 20), (3, 20), thickness_glass, permittivity_glass),
((4.5, 20), (6, 20), thickness_glass, permittivity_glass),
((12,20 ), (13.5, 20), thickness_glass, permittivity_glass),
((17, 20), (18.5, 20), thickness_glass, permittivity_glass),

((0, 20), (1.5, 20), thickness_brick, permittivity_brick),
((3, 20), (4.5,20 ), thickness_brick, permittivity_brick),
((6, 20), (12,20 ), thickness_brick, permittivity_brick),
((13.5, 20), (17,20 ), thickness_brick, permittivity_brick),
((18.5, 20), (20, 20), thickness_brick, permittivity_brick),

]

complexe_t3_obstacles_list=[

((4.5 , 0), (6, 0), thickness_glass, permittivity_glass),
((9, 0), (11,0 ), thickness_glass, permittivity_glass),
((17,0 ), (19,0 ), thickness_glass, permittivity_glass),
((2, 20), (4, 20), thickness_glass, permittivity_glass),
((7, 20), (9, 20), thickness_glass, permittivity_glass),
((12,20 ), (14, 20), thickness_glass, permittivity_glass),
((17, 20), (18, 20), thickness_glass, permittivity_glass),

(( 0, 0), (4.5,0 ), thickness_brick, permittivity_brick),
((6, 0), (9, 0), thickness_brick, permittivity_brick),
((11,0 ), (17, 0), thickness_brick, permittivity_brick),
((19, 0), (20, 0), thickness_brick, permittivity_brick),
((0, 20), (2, 20), thickness_brick, permittivity_brick),
((4, 20), (7, 20), thickness_brick, permittivity_brick),
((9, 20), (12, 20), thickness_brick, permittivity_brick),
((14,20 ), (17, 20), thickness_brick, permittivity_brick),
((18, 20), (20, 20), thickness_brick, permittivity_brick),

(( 0, 2), (0, 3), thickness_glass, permittivity_glass),
((0, 5), (0, 7), thickness_glass, permittivity_glass),
((0, 10), (0, 11), thickness_glass, permittivity_glass),
((0, 13), (0, 15), thickness_glass, permittivity_glass),
((0, 18), (0, 19), thickness_glass, permittivity_glass),

((0, 0), (0, 2), thickness_brick, permittivity_brick),
((0, 3), (0, 5), thickness_brick, permittivity_brick),
((0, 7), (0, 10), thickness_brick, permittivity_brick),
((0, 11), (0, 13), thickness_brick, permittivity_brick),
((0, 15), (0, 18), thickness_brick, permittivity_brick),
((0, 19), (0, 20), thickness_brick, permittivity_brick),


((20, 1), (20, 3), thickness_glass, permittivity_glass),
((20, 6), (20, 8), thickness_glass, permittivity_glass),
((20, 11), (20, 13), thickness_glass, permittivity_glass),
((20, 16), (20, 18), thickness_glass, permittivity_glass),

((20, 0), (20, 1), thickness_brick, permittivity_brick),
((20, 3), (20, 6), thickness_brick, permittivity_brick),
((20, 8), (20, 11 ), thickness_brick, permittivity_brick),
((20, 13), (20, 16), thickness_brick, permittivity_brick),
((20, 18), (20 , 20), thickness_brick, permittivity_brick),


((7, 0), (7, 5), thickness_plaster, permittivity_plaster),
((7, 4), (8.5, 4), thickness_plaster, permittivity_plaster),
((10, 4), (13.5, 4), thickness_plaster, permittivity_plaster),
((13.5, 4), (13.5, 0), thickness_plaster, permittivity_plaster),

((8.5, 4), (10, 4), thickness_wood, permittivity_wood),
((15.5, 16.5), (15.5, 18), thickness_wood, permittivity_wood),

((15.5, 0), (15.5, 4.5), thickness_plaster, permittivity_plaster),
((15.5, 1.5), (18.5, 1.5), thickness_plaster, permittivity_plaster),
((15.5, 6.5), (15.5, 11.5), thickness_plaster, permittivity_plaster),
((15.5, 9.5), (20, 9.5), thickness_plaster, permittivity_plaster),
((15.5, 13), (15.5, 16.55), thickness_plaster, permittivity_plaster),
((15.5, 14.5), (20, 14.5), thickness_plaster, permittivity_plaster),
((15.5, 18), (15.5, 20), thickness_plaster, permittivity_plaster),

((3.5, 12), (3.5, 13.5), thickness_wood, permittivity_wood),
((5, 17), (6.5, 17), thickness_wood, permittivity_wood),

((3.5, 4), (3.5, 12), thickness_plaster, permittivity_plaster),
((3.5, 9), (0, 9), thickness_plaster, permittivity_plaster),
((3.5, 13.5), (3.5, 17), thickness_plaster, permittivity_plaster),
((0, 17), (5, 17), thickness_plaster, permittivity_plaster),
((6.5, 17), (11,17 ), thickness_plaster, permittivity_plaster),
((11, 17), (11, 20), thickness_plaster, permittivity_plaster),


]


complexe_t4_obstacles_list=[

(( 7.5, 0), (10, 0), thickness_glass, permittivity_glass),
((16, 0), (18,0 ), thickness_glass, permittivity_glass),
((0, 2), (0, 4), thickness_glass, permittivity_glass),
((0, 8), (0, 9), thickness_glass, permittivity_glass),
((0, 11), (0, 12), thickness_glass, permittivity_glass),
((0, 14), (0, 15), thickness_glass, permittivity_glass),
((0, 18), (0, 19), thickness_glass, permittivity_glass),
((1, 20), (3, 20), thickness_glass, permittivity_glass),
((7, 20), (9, 20), thickness_glass, permittivity_glass),
((15, 20), (17, 20), thickness_glass, permittivity_glass),
((20, 3), (20, 5), thickness_glass, permittivity_glass),
((20, 8), (20, 10), thickness_glass, permittivity_glass),
((20, 13), (20, 15), thickness_glass, permittivity_glass),
((20, 16), (20, 18), thickness_glass, permittivity_glass),


((0, 0), (7.5, 0), thickness_plaster, permittivity_plaster),
((10, 0), (16, 0), thickness_plaster, permittivity_plaster),
((18, 0), (20, 0), thickness_plaster, permittivity_plaster),
((0,0 ), (0, 2), thickness_plaster, permittivity_plaster),
((0, 4), (0, 8), thickness_plaster, permittivity_plaster),
((0, 9), (0, 11), thickness_plaster, permittivity_plaster),
((0, 12), (0, 14), thickness_plaster, permittivity_plaster),
((0, 15), (0, 18), thickness_plaster, permittivity_plaster),
((0, 19), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (1, 20), thickness_plaster, permittivity_plaster),
((3, 20), (7, 20), thickness_plaster, permittivity_plaster),
((9, 20), (15, 20), thickness_plaster, permittivity_plaster),
((17,20 ), (20, 20), thickness_plaster, permittivity_plaster),
((20, 0), (20, 3), thickness_plaster, permittivity_plaster),
((20, 5), (20, 8), thickness_plaster, permittivity_plaster),
((20, 10), (20, 13), thickness_plaster, permittivity_plaster),
((20, 15), (20, 16), thickness_plaster, permittivity_plaster),
((20,18 ), (20, 20), thickness_plaster, permittivity_plaster),

((6, 13), (6, 14.5), thickness_wood, permittivity_wood),

((13.5, 0), (13.5,2 ), thickness_plaster, permittivity_plaster),
((13.5, 4), (13.5, 9), thickness_plaster, permittivity_plaster),
((13.5, 7), (20, 7), thickness_plaster, permittivity_plaster),
((13.5, 10.5), (13.5,16.5 ), thickness_plaster, permittivity_plaster),
((13.5, 12), (20, 12), thickness_plaster, permittivity_plaster),
((13.5, 18), (13.5, 20), thickness_plaster, permittivity_plaster),
((6, 4), (6, 7), thickness_plaster, permittivity_plaster),
((6, 7), (0, 7), thickness_plaster, permittivity_plaster),
((6, 10), (0, 10), thickness_plaster, permittivity_plaster),
((6, 10), (6, 13), thickness_plaster, permittivity_plaster),
((6, 14.5), (6, 18.5), thickness_plaster, permittivity_plaster),
((6, 16.5), (0, 16.5), thickness_plaster, permittivity_plaster),
]

complexe_t5_obstacles_list=[

((7.5, 0), (9, 0), thickness_glass, permittivity_glass),
((13, 0), (14,0 ), thickness_glass, permittivity_glass),
((16,0 ), (18, 0), thickness_glass, permittivity_glass),
((20, 3), (20, 5), thickness_glass, permittivity_glass),
((20, 8), (20, 10), thickness_glass, permittivity_glass),
((20, 13), (20, 15), thickness_glass, permittivity_glass),
((20, 18), (20, 19), thickness_glass, permittivity_glass),
((0, 2.5), (0, 4), thickness_glass, permittivity_glass),
((0, 5.5), (0, 7), thickness_glass, permittivity_glass),
((0, 12), (0, 14), thickness_glass, permittivity_glass),
((0, 17.5), (0, 19), thickness_glass, permittivity_glass),
((2, 20), (4, 20), thickness_glass, permittivity_glass),
((11, 20), (13,20 ), thickness_glass, permittivity_glass),
((16, 20), (18,20 ), thickness_glass, permittivity_glass),

((0, 0), (7.5,0 ), thickness_plaster, permittivity_plaster),
((9,0 ), (13, 0), thickness_plaster, permittivity_plaster),
((14, 0), (16, 0), thickness_plaster, permittivity_plaster),
((18, 0), (20, 0), thickness_plaster, permittivity_plaster),
((20, 0), (20, 3), thickness_plaster, permittivity_plaster),
((20, 5), (20, 8), thickness_plaster, permittivity_plaster),
((20, 10), (20, 13), thickness_plaster, permittivity_plaster),
((20, 15), (20, 18), thickness_plaster, permittivity_plaster),
((20, 19), (20, 20), thickness_plaster, permittivity_plaster),
((0, 0), (0, 2.5), thickness_plaster, permittivity_plaster),
((0, 4), (0, 5.5), thickness_plaster, permittivity_plaster),
((0, 7), (0, 12), thickness_plaster, permittivity_plaster),
((0, 14), (0, 17.5), thickness_plaster, permittivity_plaster),
((0, 19), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (2, 20), thickness_plaster, permittivity_plaster),
((4, 20), (11,20 ), thickness_plaster, permittivity_plaster),
((13, 20), (16,20 ), thickness_plaster, permittivity_plaster),
((18,20 ), (20, 20), thickness_plaster, permittivity_plaster),

((5, 1.5), (0, 1.5), thickness_plaster, permittivity_plaster),
((5, 1.5), (5, 4.5), thickness_plaster, permittivity_plaster),
((5, 6), (5, 10.5), thickness_plaster, permittivity_plaster),
((5, 8.5), (0, 8.5), thickness_plaster, permittivity_plaster),
((5, 12), (5, 16.5), thickness_plaster, permittivity_plaster),
((7.5, 16.5), (0, 16.5), thickness_plaster, permittivity_plaster),
((7.5, 16.5), (7.5, 17.5), thickness_plaster, permittivity_plaster),
((7.5, 19), (7.5, 20), thickness_plaster, permittivity_plaster),

((5, 4.5), (5, 6), thickness_wood, permittivity_wood),
((5, 10.5), (5, 12), thickness_wood, permittivity_wood),
((7.5, 17.5), (7.5, 19), thickness_wood, permittivity_wood),
((9.5, 17.5), (9.5, 20), thickness_wood, permittivity_wood),
((18, 16.5), (16.5, 16.5), thickness_wood, permittivity_wood),
((11.5, 0), (11.5, 6.5), thickness_plaster, permittivity_plaster),
((15, 6.5), (20, 6.5), thickness_plaster, permittivity_plaster),
((15, 6.5), (15, 12.5), thickness_plaster, permittivity_plaster),
((20, 16.5), (18, 16.5), thickness_plaster, permittivity_plaster),
((16.5, 16.5), (14.5, 16.5), thickness_plaster, permittivity_plaster),
((14.5, 16.5), (14.5, 20), thickness_plaster, permittivity_plaster),
]

complexe_t6_obstacles_list=[

((2,0 ), (4,0 ), thickness_glass, permittivity_glass),
((8, 0), (9,0 ), thickness_glass, permittivity_glass),
((12, 0), (15, 0), thickness_glass, permittivity_glass),
((0, 2), (0, 4), thickness_glass, permittivity_glass),
((0, 7), (0, 8), thickness_glass, permittivity_glass),
((0, 11), (0,12 ), thickness_glass, permittivity_glass),
((0, 16), (0,18 ), thickness_glass, permittivity_glass),
((1, 20), (2,20 ), thickness_glass, permittivity_glass),
((5, 20), (7, 20), thickness_glass, permittivity_glass),
((11, 20), (13, 20), thickness_glass, permittivity_glass),
((17,20 ), (18,20 ), thickness_glass, permittivity_glass),
((20, 2), (20, 4), thickness_glass, permittivity_glass),
((20,6 ), (20, 7), thickness_glass, permittivity_glass),
((20,10 ), (20, 12), thickness_glass, permittivity_glass),
((20, 16), (20, 18), thickness_glass, permittivity_glass),

((0, 0), (2, 0), thickness_plaster, permittivity_plaster),
((4, 0), (8, 0), thickness_plaster, permittivity_plaster),
((9, 0), (12, 0), thickness_plaster, permittivity_plaster),
((15, 0), (20,0 ), thickness_plaster, permittivity_plaster),
((0, 0), (0, 2), thickness_plaster, permittivity_plaster),
((0, 4), (0, 7), thickness_plaster, permittivity_plaster),
((0, 8), (0, 11), thickness_plaster, permittivity_plaster),
((0, 12), (0, 16), thickness_plaster, permittivity_plaster),
((0, 18), (0,20 ), thickness_plaster, permittivity_plaster),
((0,20 ), (1, 20), thickness_plaster, permittivity_plaster),
((2, 20), (5, 20), thickness_plaster, permittivity_plaster),
((7, 20), (11,20 ), thickness_plaster, permittivity_plaster),
((13, 20), (17, 20), thickness_plaster, permittivity_plaster),
((18,20 ), (20,20 ), thickness_plaster, permittivity_plaster),
((20, 0), (20, 2), thickness_plaster, permittivity_plaster),
((20, 4), (20, 6), thickness_plaster, permittivity_plaster),
((20, 7), (20, 10), thickness_plaster, permittivity_plaster),
((20, 12), (20, 16), thickness_plaster, permittivity_plaster),
((20,18 ), (20, 20), thickness_plaster, permittivity_plaster),

((7.5, 0), (7.5, 5), thickness_plaster, permittivity_plaster),
((7.5, 5), (2.5, 5), thickness_plaster, permittivity_plaster),
((10.5, 0), (10.5, 1.5), thickness_plaster, permittivity_plaster),
((10.5, 3), (10.5, 5), thickness_plaster, permittivity_plaster),
((10.5, 5), (17.5, 5), thickness_plaster, permittivity_plaster),

((15, 5), (15, 10.5), thickness_plaster, permittivity_plaster),
((15, 8), (20, 8), thickness_plaster, permittivity_plaster),
((15, 12), (15, 14), thickness_plaster, permittivity_plaster),
((15, 14), (20, 14), thickness_plaster, permittivity_plaster),
((15, 16), (18, 16), thickness_plaster, permittivity_plaster),
((15, 16), (15, 20), thickness_plaster, permittivity_plaster),
((3, 20), (3, 15.5), thickness_plaster, permittivity_plaster),
((3, 15.5), (2, 15.5), thickness_plaster, permittivity_plaster),

((4, 7), (4, 8.5), thickness_wood, permittivity_wood),
((4, 10.5), (4, 12), thickness_wood, permittivity_wood),
((15, 10.5), (15, 12), thickness_wood, permittivity_wood),

((0, 6), (4, 6), thickness_plaster, permittivity_plaster),
((4, 6), (4,7 ), thickness_plaster, permittivity_plaster),
((4, 8.5), (4, 9.5), thickness_plaster, permittivity_plaster),
((4, 9.5), (0, 9.5), thickness_plaster, permittivity_plaster),
((4, 9.5), (4, 10.5), thickness_plaster, permittivity_plaster),
((4, 12), (4, 13.5), thickness_plaster, permittivity_plaster),
((4, 13.5), (0,13.5), thickness_plaster, permittivity_plaster),
]

complexe_t7_obstacles_list=[

((1, 0), (2, 0), thickness_glass, permittivity_glass),
((5, 0), (7, 0), thickness_glass, permittivity_glass),
((9, 0), (11, 0), thickness_glass, permittivity_glass),
((15, 0), (17, 0), thickness_glass, permittivity_glass),
((0,0 ), (1,0 ), thickness_plaster, permittivity_plaster),
((2,0 ), (5, 0), thickness_plaster, permittivity_plaster),
((7, 0), (9, 0), thickness_plaster, permittivity_plaster),
((11,0 ), (15, 0), thickness_plaster, permittivity_plaster),
((17,0 ), (20,0 ), thickness_plaster, permittivity_plaster),

((20, 1), (20, 3), thickness_glass, permittivity_glass),
((20, 6), (20, 8), thickness_glass, permittivity_glass),
((20, 12), (20, 14), thickness_glass, permittivity_glass),
((20, 18), (20, 19), thickness_glass, permittivity_glass),

((20,0 ), (20, 1), thickness_plaster, permittivity_plaster),
((20, 3), (20, 6), thickness_plaster, permittivity_plaster),
((20, 8), (20, 12), thickness_plaster, permittivity_plaster),
((20, 14), (20, 18), thickness_plaster, permittivity_plaster),
((20,19 ), (20, 20), thickness_plaster, permittivity_plaster),

((0, 3), (0,5 ), thickness_wood, permittivity_wood),
((0, 9), (0, 11), thickness_wood, permittivity_wood),
((0, 16), (0, 18), thickness_wood, permittivity_wood),

((0, 0), (0, 3), thickness_plaster, permittivity_plaster),
((0, 5), (0, 9), thickness_plaster, permittivity_plaster),
((0,11 ), (0, 16), thickness_plaster, permittivity_plaster),
((0, 18), (0, 20), thickness_plaster, permittivity_plaster),

((6, 20), (8, 20), thickness_glass, permittivity_glass),
((13, 20), (14,20 ), thickness_glass, permittivity_glass),
((16, 20), (18, 20), thickness_glass, permittivity_glass),

((0, 20), (6, 20), thickness_plaster, permittivity_plaster),
((8, 20), (13, 20), thickness_plaster, permittivity_plaster),
((14, 20), (16, 20), thickness_plaster, permittivity_plaster),
((18, 20), (20, 20), thickness_plaster, permittivity_plaster),

((4,3 ), (4, 4.5), thickness_wood, permittivity_wood),
((4, 9), (4, 10.5), thickness_wood, permittivity_wood),

((4,0 ), (4, 3), thickness_plaster, permittivity_plaster),
((4, 4.5), (4, 7), thickness_plaster, permittivity_plaster),
((4, 7), (0, 7), thickness_plaster, permittivity_plaster),
((4, 7), (4, 9), thickness_plaster, permittivity_plaster),
((4, 10.5), (4, 12.5), thickness_plaster, permittivity_plaster),
((4, 12.5), (0, 12.5), thickness_plaster, permittivity_plaster),

((2.5, 14.5), (2.5, 17.5), thickness_plaster, permittivity_plaster),
((2.5, 14.5), (10.5, 14.5), thickness_plaster, permittivity_plaster),
((10.5, 14.5), (10.5, 20), thickness_plaster, permittivity_plaster),


((12, 16.5), (20, 16.5), thickness_plaster, permittivity_plaster),
((13.5, 16.5), (13.5, 15), thickness_plaster, permittivity_plaster),
((13.5, 13), (13.5, 7.5), thickness_plaster, permittivity_plaster),
((13.5, 5.5), (13.5, 1.5), thickness_plaster, permittivity_plaster),
((13.5, 4.5), (20, 4.5), thickness_plaster, permittivity_plaster),
((13.5, 11), (17, 11), thickness_plaster, permittivity_plaster),

((13.5, 7.5), (13.5, 5.5), thickness_wood, permittivity_wood),
((13.5, 15), (13.5, 13), thickness_wood, permittivity_wood),

]


complexe_t8_obstacles_list =[

((15, 6), (17,6 ), thickness_wood, permittivity_wood),
((2,0 ), (4, 0), thickness_glass, permittivity_glass),
((7.5, 0), (9.5, 0), thickness_glass, permittivity_glass),
((14,0 ), (16, 0), thickness_glass, permittivity_glass),

((0, 0), (2, 0), thickness_plaster, permittivity_plaster),
((4, 0), (7.5, 0), thickness_plaster, permittivity_plaster),
((9.5, 0), (14, 0), thickness_plaster, permittivity_plaster),
((16,0 ), (20, 0), thickness_plaster, permittivity_plaster),
((6, 0), (6,4 ), thickness_plaster, permittivity_plaster),
((11,0 ), (11,6 ), thickness_plaster, permittivity_plaster),
((2.5, 6), (15,6 ), thickness_plaster, permittivity_plaster),
((17,6 ), (20,6 ), thickness_plaster, permittivity_plaster),

((2, 20), (4, 20), thickness_wood, permittivity_wood),

((0, 7), (0, 9.5), thickness_glass, permittivity_glass),
((0, 16), (0, 18), thickness_glass, permittivity_glass),
((7.5, 20), (10, 20), thickness_glass, permittivity_glass),
((13.5, 20), (17, 20), thickness_glass, permittivity_glass),

((0,0 ), (0, 7), thickness_plaster, permittivity_plaster),
((0, 9.5), (0, 16), thickness_plaster, permittivity_plaster),
((0, 18), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (2, 20), thickness_plaster, permittivity_plaster),
((4, 20), (7.5, 20), thickness_plaster, permittivity_plaster),
((10, 20), (13.5, 20), thickness_plaster, permittivity_plaster),
((17, 20), (20, 20), thickness_plaster, permittivity_plaster),

((6, 20), (6, 14), thickness_plaster, permittivity_plaster),
((6, 14), (2.5, 14), thickness_plaster, permittivity_plaster),
((11.5, 20), (11.5, 14), thickness_plaster, permittivity_plaster),
((8, 14), (17.5, 14), thickness_plaster, permittivity_plaster),


((20, 2), (20, 4), thickness_glass, permittivity_glass),
((20, 8), (20, 12), thickness_glass, permittivity_glass),
((20,17 ), (20, 19), thickness_glass, permittivity_glass),

((20, 0), (20, 2), thickness_plaster, permittivity_plaster),
((20, 4), (20, 8), thickness_plaster, permittivity_plaster),
((20, 12), (20, 17), thickness_plaster, permittivity_plaster),
((20, 19), (20, 20), thickness_plaster, permittivity_plaster),

]


complexe_t9_obstacles_list=[

((17, 9.5), (17, 11), thickness_wood, permittivity_wood),
((17, 2.5), (17, 4.5), thickness_wood, permittivity_wood),

((3.5, 0), (6.5,0 ), thickness_glass, permittivity_glass),
((10.5, 0), (13, 0), thickness_glass, permittivity_glass),

((0, 0), (3.5, 0), thickness_plaster, permittivity_plaster),
((6.5, 0), (10.5, 0), thickness_plaster, permittivity_plaster),
((13,0 ), (20, 0), thickness_plaster, permittivity_plaster),
((17, 0), (17, 2.5), thickness_plaster, permittivity_plaster),
((17, 4.5), (17, 9.5), thickness_plaster, permittivity_plaster),
((17, 7.5), (20, 7.5), thickness_plaster, permittivity_plaster),
((17, 11), (17, 15), thickness_plaster, permittivity_plaster),

((0, 2), (0, 4), thickness_glass, permittivity_glass),
((0, 7.5), (0, 10), thickness_glass, permittivity_glass),
((0, 12), (0, 14), thickness_glass, permittivity_glass),

((0, 0), (0, 2), thickness_plaster, permittivity_plaster),
((0, 4), (0, 7.5), thickness_plaster, permittivity_plaster),
((0, 10), (0, 12), thickness_plaster, permittivity_plaster),
((0, 14), (0, 20), thickness_plaster, permittivity_plaster),

((20, 4), (20, 6), thickness_glass, permittivity_glass),
((20, 11), (20, 13), thickness_glass, permittivity_glass),
((20, 17), (20, 19), thickness_glass, permittivity_glass),

((20, 0), (20, 4), thickness_plaster, permittivity_plaster),
((20, 6), (20, 11), thickness_plaster, permittivity_plaster),
((20, 13), (20, 17), thickness_plaster, permittivity_plaster),
((20, 19), (20, 20), thickness_plaster, permittivity_plaster),

((16, 15), (14, 15), thickness_wood, permittivity_wood),
((10, 17.5), (10, 19), thickness_wood, permittivity_wood),


((20, 15), (16, 15), thickness_plaster, permittivity_plaster),
((14, 15), (10, 15), thickness_plaster, permittivity_plaster),
((10, 15), (10, 17.5), thickness_plaster, permittivity_plaster),
((10 ,19 ), (10, 20), thickness_plaster, permittivity_plaster),


((2.5, 20), (4.5, 20), thickness_glass, permittivity_glass),
((11.5, 20), (13, 20), thickness_glass, permittivity_glass),
((16, 20), (17.5, 20), thickness_glass, permittivity_glass),

((0, 20), (2.5, 20), thickness_plaster, permittivity_plaster),
((4.5, 20), (11.5, 20), thickness_plaster, permittivity_plaster),
((11.5, 20), (13,20 ), thickness_plaster, permittivity_plaster),
((17.5, 20), (20, 20), thickness_plaster, permittivity_plaster),

((0, 6), (5, 6), thickness_plaster, permittivity_plaster),
((5, 6), (5, 9), thickness_plaster, permittivity_plaster),
((5,9 ), (7, 9), thickness_plaster, permittivity_plaster),
((7, 9), (7, 10), thickness_plaster, permittivity_plaster),
((7, 11.5), (7, 12), thickness_plaster, permittivity_plaster),
((7, 12), (5, 12), thickness_plaster, permittivity_plaster),
((5,12 ), (5, 15), thickness_plaster, permittivity_plaster),
((0,15 ), (8, 15), thickness_plaster, permittivity_plaster),
((8, 15), (8, 17.5), thickness_plaster, permittivity_plaster),

]

complexe_t10_obstacles_list=[

((1, 0), (3, 0), thickness_glass, permittivity_glass),
((6.5, 0), (8.5, 0), thickness_glass, permittivity_glass),
((13.5, 0), (15.5, 0), thickness_glass, permittivity_glass),

((0,0 ), (1,0 ), thickness_plaster, permittivity_plaster),
((3,0 ), (6.5,0 ), thickness_plaster, permittivity_plaster),
((8.5, 0), (13.5,0 ), thickness_plaster, permittivity_plaster),
((15.5, 0), (20,0 ), thickness_plaster, permittivity_plaster),

((4.5, 0), (4.5,5 ), thickness_plaster, permittivity_plaster),
((2, 5), (8, 5), thickness_plaster, permittivity_plaster),
((10.5, 0), (10.5, 5), thickness_plaster, permittivity_plaster),
((17, 0), (17,5 ), thickness_plaster, permittivity_plaster),
((17, 5), (14, 5), thickness_plaster, permittivity_plaster),

((20, 3.5), (20, 5), thickness_glass, permittivity_glass),
((20, 9.5), (20, 11.5), thickness_glass, permittivity_glass),
((20,14 ), (20, 15.5), thickness_glass, permittivity_glass),
((20,18 ), (20, 19), thickness_glass, permittivity_glass),

((20,0), (20, 3.5), thickness_plaster, permittivity_plaster),
((20,5), (20, 9.5), thickness_plaster, permittivity_plaster),
((20, 11.5), (20,14 ), thickness_plaster, permittivity_plaster),
((20, 15.5), (20, 18), thickness_plaster, permittivity_plaster),
((20, 19), (20,20 ), thickness_plaster, permittivity_plaster),

((6,20 ), (6, 15), thickness_plaster, permittivity_plaster),
((2, 15), (8.5, 15), thickness_plaster, permittivity_plaster),
((20, 17), (12, 17), thickness_plaster, permittivity_plaster),
((13.5, 17), (13.5, 11.5), thickness_plaster, permittivity_plaster),
((13.5, 7), (13.5,10 ), thickness_plaster, permittivity_plaster),
((13.5, 7), (20, 7), thickness_plaster, permittivity_plaster),

((0, 2), (0, 3.5), thickness_glass, permittivity_glass),
((0, 7), (0,9 ), thickness_glass, permittivity_glass),
((0, 12), (0, 13.5), thickness_glass, permittivity_glass),
((0, 16.5), (0, 18), thickness_glass, permittivity_glass),

((0, 0), (0, 2), thickness_plaster, permittivity_plaster),
((0, 3.5), (0, 7), thickness_plaster, permittivity_plaster),
((0, 9), (0, 12), thickness_plaster, permittivity_plaster),
((0, 13.5), (0, 16.5), thickness_plaster, permittivity_plaster),
((0, 18), (0, 20), thickness_plaster, permittivity_plaster),

((1.5, 20), (3, 20), thickness_glass, permittivity_glass),
((7.5, 20), (9, 20), thickness_glass, permittivity_glass),
((13.5, 20), (15.5, 20), thickness_glass, permittivity_glass),

((0, 20), (1.5, 20), thickness_plaster, permittivity_plaster),
((3, 20), (7.5, 20), thickness_plaster, permittivity_plaster),
((9, 20), (13.5, 20), thickness_plaster, permittivity_plaster),
((15.5, 20), (20, 20), thickness_plaster, permittivity_plaster),

]
complexe_t11_obstacles_list=[


((2, 0), (4, 0), thickness_glass, permittivity_glass),
((6, 0), (8,0 ), thickness_glass, permittivity_glass),
((13, 0), (14,0 ), thickness_glass, permittivity_glass),
((16,0 ), (17, 0), thickness_glass, permittivity_glass),
((0,0 ), (2, 0), thickness_plaster, permittivity_plaster),
((4, 0), (6, 0), thickness_plaster, permittivity_plaster),
((8, 0), (13,0 ), thickness_plaster, permittivity_plaster),
((14,0 ), (16,0 ), thickness_plaster, permittivity_plaster),
((17, 0), (20, 0), thickness_plaster, permittivity_plaster),


((0, 2), (0, 3), thickness_glass, permittivity_glass),
((0, 11), (0, 13), thickness_glass, permittivity_glass),
((0, 16), (0, 18), thickness_glass, permittivity_glass),
((0, 0), (0,2 ), thickness_plaster, permittivity_plaster),
((0, 3), (0, 11), thickness_plaster, permittivity_plaster),
((0, 13), (0, 16), thickness_plaster, permittivity_plaster),
((0, 18), (0, 20), thickness_plaster, permittivity_plaster),

((20,1 ), (20, 2), thickness_glass, permittivity_glass),
((20, 5), (20, 6), thickness_glass, permittivity_glass),
((20, 11), (20, 13), thickness_glass, permittivity_glass),
((20, 15), (20, 17), thickness_glass, permittivity_glass),
((20,0 ), (20,1 ), thickness_plaster, permittivity_plaster),
((20, 2), (20, 5), thickness_plaster, permittivity_plaster),
((20, 6), (20, 11), thickness_plaster, permittivity_plaster),
((20,13 ), (20, 15), thickness_plaster, permittivity_plaster),
((20, 17), (20, 20), thickness_plaster, permittivity_plaster),

((2,20 ), (4, 20), thickness_glass, permittivity_glass),
((8, 20), (10, 20), thickness_glass, permittivity_glass),
((0, 20), (2, 20), thickness_plaster, permittivity_plaster),
((4, 20), (8, 20), thickness_plaster, permittivity_plaster),
((10, 20), (20, 20), thickness_plaster, permittivity_plaster),

((9, 7), (9, 9), thickness_wood, permittivity_wood),
((15, 15), (15, 16), thickness_wood, permittivity_wood),
((0, 5), (9,5 ), thickness_plaster, permittivity_plaster),
((9, 5), (9, 7), thickness_plaster, permittivity_plaster),
((9, 9), (9,12 ), thickness_plaster, permittivity_plaster),
((9,12 ), (15,12 ), thickness_plaster, permittivity_plaster),
((15, 12), (15, 15), thickness_plaster, permittivity_plaster),
((15, 16), (15, 20), thickness_plaster, permittivity_plaster),

((12, 1), (12, 2), thickness_wood, permittivity_wood),
((16,9 ), (17, 9), thickness_wood, permittivity_wood),

((12, 0), (12, 1), thickness_plaster, permittivity_plaster),
((12, 2), (12,4 ), thickness_plaster, permittivity_plaster),
((12,4 ), (14, 4), thickness_plaster, permittivity_plaster),
((14, 4), (14, 9), thickness_plaster, permittivity_plaster),
((14, 9), (16, 9), thickness_plaster, permittivity_plaster),
((17, 9), (20, 9), thickness_plaster, permittivity_plaster),
]


complexe_t12_obstacles_list=[


((1, 0), (3, 0), thickness_glass, permittivity_glass),
((7, 0), (9, 0), thickness_glass, permittivity_glass),
((14, 0), (16, 0), thickness_glass, permittivity_glass),
((0,0 ), (1, 0), thickness_plaster, permittivity_plaster),
((3, 0), (7, 0), thickness_plaster, permittivity_plaster),
((9, 0), (14, 0), thickness_plaster, permittivity_plaster),
((16,0 ), (20, 0), thickness_plaster, permittivity_plaster),

((20, 2), (20, 4), thickness_glass, permittivity_glass),
((20, 9), (20, 11), thickness_glass, permittivity_glass),
((20, 17), (20, 18), thickness_glass, permittivity_glass),
((20, 0), (20, 2), thickness_plaster, permittivity_plaster),
((20, 4), (20,9 ), thickness_plaster, permittivity_plaster),
((20, 11), (20, 17), thickness_plaster, permittivity_plaster),
((20,18 ), (20,20 ), thickness_plaster, permittivity_plaster),

((0, 3), (0, 4), thickness_glass, permittivity_glass),
((0, 9), (0, 10), thickness_glass, permittivity_glass),
((0, 13), (0, 14), thickness_glass, permittivity_glass),
((0, 17), (0, 18), thickness_glass, permittivity_glass),
((3, 20), (5, 20), thickness_glass, permittivity_glass),
((8, 20), (10,20 ), thickness_glass, permittivity_glass),
((12, 20), (14, 20), thickness_glass, permittivity_glass),
((17, 20), (18, 20), thickness_glass, permittivity_glass),

((0, 0), (0, 3), thickness_plaster, permittivity_plaster),
((0, 4), (0, 9), thickness_plaster, permittivity_plaster),
((0, 10), (0, 13), thickness_plaster, permittivity_plaster),
((0, 14), (0, 17), thickness_plaster, permittivity_plaster),
((0, 18), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (3, 20), thickness_plaster, permittivity_plaster),
((5, 20), (8, 20), thickness_plaster, permittivity_plaster),
((10, 20), (12, 20), thickness_plaster, permittivity_plaster),
((14, 20), (17, 20), thickness_plaster, permittivity_plaster),
((18, 20), (20, 20), thickness_plaster, permittivity_plaster),

((5, 3), (5, 5), thickness_wood, permittivity_wood),
((5, 0), (5, 3), thickness_plaster, permittivity_plaster),
((5, 5), (5, 7), thickness_plaster, permittivity_plaster),
((5, 7), (0, 7), thickness_plaster, permittivity_plaster),
((5, 7), (5, 9), thickness_plaster, permittivity_plaster),
((5, 11), (5, 17), thickness_plaster, permittivity_plaster),
((5, 17), (7, 17), thickness_plaster, permittivity_plaster),
((7, 17), (7,20 ), thickness_plaster, permittivity_plaster),


((12, 2), (12, 4), thickness_wood, permittivity_wood),
((12, 11), (12, 13), thickness_wood, permittivity_wood),
((15, 17), (15, 18), thickness_wood, permittivity_wood),

((12, 0), (12, 2), thickness_plaster, permittivity_plaster),
((12, 4), (12, 7), thickness_plaster, permittivity_plaster),
((12, 7), (20, 7), thickness_plaster, permittivity_plaster),
((12, 7), (12, 11), thickness_plaster, permittivity_plaster),
((12, 13), (12, 15), thickness_plaster, permittivity_plaster),
((12, 15), (20, 15), thickness_plaster, permittivity_plaster),
((15, 15), (15, 17), thickness_plaster, permittivity_plaster),
((15, 18), (15, 20), thickness_plaster, permittivity_plaster),

]

complexe_t14_obstacles_list=[

((2, 0), (4, 0), thickness_glass, permittivity_glass),
((8, 0), (10, 0), thickness_glass, permittivity_glass),
((14, 0), (17, 0), thickness_glass, permittivity_glass),
((0,2 ), (0, 3), thickness_glass, permittivity_glass),
((0, 7), (0, 8), thickness_glass, permittivity_glass),
((0, 10), (0,11 ), thickness_glass, permittivity_glass),
((0, 14), (0, 15), thickness_glass, permittivity_glass),

((0,0 ), (2, 0), thickness_plaster, permittivity_plaster),
((4, 0), (8, 0), thickness_plaster, permittivity_plaster),
((10, 0), (14, 0), thickness_plaster, permittivity_plaster),
((17, 0), (20, 0), thickness_plaster, permittivity_plaster),
((0, 0), (0, 2), thickness_plaster, permittivity_plaster),
((0, 3), (0, 7), thickness_plaster, permittivity_plaster),
((0, 8), (0, 10), thickness_plaster, permittivity_plaster),
((0, 11), (0, 14), thickness_plaster, permittivity_plaster),
((0, 15), (0, 20), thickness_plaster, permittivity_plaster),

((7, 2), (7, 4), thickness_wood, permittivity_wood),
((7, 7), (7, 8), thickness_wood, permittivity_wood),
((7, 10), (7, 11), thickness_wood, permittivity_wood),
((7,13 ), (7, 14), thickness_wood, permittivity_wood),

((7, 0), (7, 2), thickness_plaster, permittivity_plaster),
((7, 4), (7, 5), thickness_plaster, permittivity_plaster),
((7, 5), (0, 5), thickness_plaster, permittivity_plaster),
((7, 5), (7, 7), thickness_plaster, permittivity_plaster),
((7, 8), (7, 9), thickness_plaster, permittivity_plaster),
((7, 9), (0, 9), thickness_plaster, permittivity_plaster),
((7, 9), (7, 10), thickness_plaster, permittivity_plaster),
((7, 11), (7, 12), thickness_plaster, permittivity_plaster),
((7, 12), (0, 12), thickness_plaster, permittivity_plaster),
((7, 12), (7, 13), thickness_plaster, permittivity_plaster),
((7, 14), (7, 16), thickness_plaster, permittivity_plaster),
((7, 16), (0, 16), thickness_plaster, permittivity_plaster),

((12, 2), (12, 4), thickness_wood, permittivity_wood),
((12, 8), (12, 10), thickness_wood, permittivity_wood),
((15,12 ), (17, 12), thickness_wood, permittivity_wood),

((12, 0), (12, 2), thickness_plaster, permittivity_plaster),
((12, 4),(12,6 ), thickness_plaster, permittivity_plaster),
((12, 6), (20, 6), thickness_plaster, permittivity_plaster),
((12, 6), (12, 8), thickness_plaster, permittivity_plaster),
((12, 10), (12, 12), thickness_plaster, permittivity_plaster),
((12, 12), (15, 12), thickness_plaster, permittivity_plaster),
((17, 12), (20, 12), thickness_plaster, permittivity_plaster),

((4, 20), (6, 20), thickness_glass, permittivity_glass),
((9, 20), (11, 20), thickness_glass, permittivity_glass),
((16, 20), (18, 20), thickness_glass, permittivity_glass),
((0, 20), (4, 20), thickness_plaster, permittivity_plaster),
((6, 20), (9, 20), thickness_plaster, permittivity_plaster),
((11, 20), (16, 20), thickness_plaster, permittivity_plaster),
((18, 20), (20, 20), thickness_plaster, permittivity_plaster),

((20, 2), (20, 4), thickness_glass, permittivity_glass),
((20, 8), (20, 10), thickness_glass, permittivity_glass),
((20, 13), (20, 14), thickness_glass, permittivity_glass),
((20, 16), (20, 18), thickness_glass, permittivity_glass),

((20, 0), (20, 2), thickness_plaster, permittivity_plaster),
((20, 4), (20, 8), thickness_plaster, permittivity_plaster),
((20, 10), (20, 13), thickness_plaster, permittivity_plaster),
((20, 14), (20, 16), thickness_plaster, permittivity_plaster),
((20, 18), (20, 20), thickness_plaster, permittivity_plaster),

((13, 17), (13, 18), thickness_wood, permittivity_wood),
((20, 15), (13, 15), thickness_plaster, permittivity_plaster),
((13, 15), (13, 17), thickness_plaster, permittivity_plaster),
((13, 18), (13, 20), thickness_plaster, permittivity_plaster),

]
complexe_t15_obstacles_list=[

((4, 0), (6, 0), thickness_glass, permittivity_glass),
((10, 0), (11,0 ), thickness_glass, permittivity_glass),
((15, 0), (17, 0), thickness_glass, permittivity_glass),
((7, 4), (6, 5), thickness_wood, permittivity_wood),
((0, 0), (4,0 ), thickness_plaster, permittivity_plaster),
((6, 0), (10, 0), thickness_plaster, permittivity_plaster),
((11,0 ), (15, 0), thickness_plaster, permittivity_plaster),
((17, 0), (20, 0), thickness_plaster, permittivity_plaster),
((8, 0), (8, 3), thickness_plaster, permittivity_plaster),
((8, 3), (7, 4), thickness_plaster, permittivity_plaster),
((6, 5), (5, 6), thickness_plaster, permittivity_plaster),
((5, 6), (0, 6), thickness_plaster, permittivity_plaster),

((5, 8), (5, 9), thickness_wood, permittivity_wood),
((5, 13), (5, 14), thickness_wood, permittivity_wood),
((6, 15), (7, 15), thickness_wood, permittivity_wood),


((5, 6), (5, 8), thickness_plaster, permittivity_plaster),
((5, 9), (5, 11), thickness_plaster, permittivity_plaster),
((5, 11), (0, 11), thickness_plaster, permittivity_plaster),
((5, 11), (5, 13), thickness_plaster, permittivity_plaster),
((5, 14), (5, 15), thickness_plaster, permittivity_plaster),
((0, 15), (6, 15), thickness_plaster, permittivity_plaster),
((7, 15), (8, 15), thickness_plaster, permittivity_plaster),
((8, 15), (8, 20), thickness_plaster, permittivity_plaster),

((12, 4), (12, 6), thickness_wood, permittivity_wood),
((16, 12), (16, 13), thickness_wood, permittivity_wood),
((14, 15), (15, 15), thickness_wood, permittivity_wood),

((12, 0), (12, 4), thickness_plaster, permittivity_plaster),
((12, 6), (12, 11), thickness_plaster, permittivity_plaster),
((12, 11), (20, 11), thickness_plaster, permittivity_plaster),
((16, 11), (16, 12), thickness_plaster, permittivity_plaster),
((16, 13), (16, 15), thickness_plaster, permittivity_plaster),
((13, 15), (14, 15), thickness_plaster, permittivity_plaster),
((15, 15), (20, 15), thickness_plaster, permittivity_plaster),
((13, 15), (13, 20), thickness_plaster, permittivity_plaster),

((2, 20), (4, 20), thickness_glass, permittivity_glass),
((10, 20), (12, 20), thickness_glass, permittivity_glass),
((16, 20), (18, 20), thickness_glass, permittivity_glass),


((0, 20), (2, 20), thickness_plaster, permittivity_plaster),
((4, 20), (10, 20), thickness_plaster, permittivity_plaster),
((12, 20), (16, 20), thickness_plaster, permittivity_plaster),
((18, 20), (20, 20), thickness_plaster, permittivity_plaster),

((0, 1), (0, 3), thickness_glass, permittivity_glass),
((0, 8), (0,9 ), thickness_glass, permittivity_glass),
((0, 12), (0, 13), thickness_glass, permittivity_glass),
((0, 17), (0, 18), thickness_glass, permittivity_glass),

((0, 0), (0, 1), thickness_plaster, permittivity_plaster),
((0, 3), (0, 8), thickness_plaster, permittivity_plaster),
((0, 9), (0, 12), thickness_plaster, permittivity_plaster),
((0, 13), (0, 17), thickness_plaster, permittivity_plaster),

((20, 2), (20, 4), thickness_glass, permittivity_glass),
((20, 7), (20, 9), thickness_glass, permittivity_glass),
((20, 13), (20, 14), thickness_glass, permittivity_glass),
((20, 17), (20, 18), thickness_glass, permittivity_glass),
((20, 0), (20, 2), thickness_plaster, permittivity_plaster),
((20, 4), (20, 7), thickness_plaster, permittivity_plaster),
((20, 9), (20, 13), thickness_plaster, permittivity_plaster),
((20, 14), (20, 17), thickness_plaster, permittivity_plaster),
((20, 18), (20, 20), thickness_plaster, permittivity_plaster),

]

complexe_t16_obstacles_list=[
((3, 0), (5, 0), thickness_glass, permittivity_glass),
((8, 0), (10, 0), thickness_glass, permittivity_glass),
((14,0 ), (16, 0), thickness_glass, permittivity_glass),
((0, 4), (0, 6), thickness_glass, permittivity_glass),
((0, 9), (0, 10), thickness_glass, permittivity_glass),
((0, 12), (0, 13), thickness_glass, permittivity_glass),
((0, 16), (0, 17), thickness_glass, permittivity_glass),

((0, 0), (3, 0), thickness_plaster, permittivity_plaster),
((5, 0), (8, 0), thickness_plaster, permittivity_plaster),
((10, 0), (14, 0), thickness_plaster, permittivity_plaster),
((16, 0), (20, 0), thickness_plaster, permittivity_plaster),
((0, 0), (0, 4), thickness_plaster, permittivity_plaster),
((0, 6), (0, 9), thickness_plaster, permittivity_plaster),
((0, 10), (0,12 ), thickness_plaster, permittivity_plaster),
((0, 13), (0, 16), thickness_plaster, permittivity_plaster),
((0, 17), (0, 20), thickness_plaster, permittivity_plaster),

((4,  20), (6, 20), thickness_glass, permittivity_glass),
((9, 20), (11, 20), thickness_glass, permittivity_glass),
((16, 20), (18, 20), thickness_glass, permittivity_glass),
((20, 2), (20, 4), thickness_glass, permittivity_glass),
((20, 8), (20, 10), thickness_glass, permittivity_glass),
((20, 14), (20, 16), thickness_glass, permittivity_glass),

((0, 20), (4, 20), thickness_plaster, permittivity_plaster),
((6, 20), (9, 20), thickness_plaster, permittivity_plaster),
((11, 20), (16, 20), thickness_plaster, permittivity_plaster),
((18, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 0), (20, 2), thickness_plaster, permittivity_plaster),
((20, 4), (20, 8), thickness_plaster, permittivity_plaster),
((20, 10), (20, 14), thickness_plaster, permittivity_plaster),
((20, 16), (20, 20), thickness_plaster, permittivity_plaster),

((15, 6), (17, 6), thickness_wood, permittivity_wood),
((12, 0), (12, 6), thickness_plaster, permittivity_plaster),
((12, 6), (15, 6), thickness_plaster, permittivity_plaster),
((17, 6), (20, 6), thickness_plaster, permittivity_plaster),

((6, 14), (6, 15), thickness_wood, permittivity_wood),
((16, 12), (17, 12), thickness_wood, permittivity_wood),
((13, 16), (13, 17), thickness_wood, permittivity_wood),

((0, 8), (4, 8), thickness_plaster, permittivity_plaster),
((4, 8), (4, 12), thickness_plaster, permittivity_plaster),
((4, 12), (6, 12), thickness_plaster, permittivity_plaster),
((6, 12), (6, 14), thickness_plaster, permittivity_plaster),
((6, 15), (6, 16), thickness_plaster, permittivity_plaster),
((6, 16), (7, 16), thickness_plaster, permittivity_plaster),
((7, 16), (7, 20), thickness_plaster, permittivity_plaster),
((20, 12), (17,12), thickness_plaster, permittivity_plaster),
((16, 12), (13, 12), thickness_plaster, permittivity_plaster),
((13, 12), (13, 16), thickness_plaster, permittivity_plaster),
((13, 17), (13,20 ), thickness_plaster, permittivity_plaster),

]

complexe_t17_obstacles_list=[

((1, 0), (2, 0), thickness_glass, permittivity_glass),
((3, 0), (4, 0), thickness_glass, permittivity_glass),
((6, 0), (7, 0), thickness_glass, permittivity_glass),
((10, 0), (11, 0), thickness_glass, permittivity_glass),
((15, 0), (17, 0), thickness_glass, permittivity_glass),

((0, 0), (1,0 ), thickness_plaster, permittivity_plaster),
((2, 0), (3, 0), thickness_plaster, permittivity_plaster),
((4, 0), (6, 0), thickness_plaster, permittivity_plaster),
((7, 0), (10, 0), thickness_plaster, permittivity_plaster),
((11,0 ), (15, 0), thickness_plaster, permittivity_plaster),
((17, 0), (20, 0), thickness_plaster, permittivity_plaster),

((0, 6), (0, 8), thickness_glass, permittivity_glass),
((0, 10), (0, 12), thickness_glass, permittivity_glass),
((0, 16), (0, 18), thickness_glass, permittivity_glass),
((0, 0), (0, 6), thickness_plaster, permittivity_plaster),
((0, 8), (0, 10), thickness_plaster, permittivity_plaster),
((0, 12), (0, 16), thickness_plaster, permittivity_plaster),
((0, 18), (0, 20), thickness_plaster, permittivity_plaster),

((2, 20), (4, 20), thickness_glass, permittivity_glass),
((12, 20), (14, 20), thickness_glass, permittivity_glass),
((7, 20), (9, 20), thickness_glass, permittivity_glass),
((17, 20), (19, 20), thickness_glass, permittivity_glass),

((0, 20), (2, 20), thickness_plaster, permittivity_plaster),
((4, 20), (7, 20), thickness_plaster, permittivity_plaster),
((9, 20), (12, 20), thickness_plaster, permittivity_plaster),
((14, 20), (17, 20), thickness_plaster, permittivity_plaster),
((19, 20), (20, 20), thickness_plaster, permittivity_plaster),

((6, 20), (6,12 ), thickness_glass, permittivity_glass),
((2,4 ), (3, 4), thickness_wood, permittivity_wood),
((6, 4), (7, 4), thickness_wood, permittivity_wood),
((10, 4), (11, 4), thickness_wood, permittivity_wood),

((5, 0), (5, 4), thickness_plaster, permittivity_plaster),
((0, 4), (2, 4), thickness_plaster, permittivity_plaster),
((3, 4), (6, 4), thickness_plaster, permittivity_plaster),
((7, 4), (10, 4), thickness_plaster, permittivity_plaster),
((11, 4), (13, 4), thickness_plaster, permittivity_plaster),
((13, 4), (13, 0), thickness_plaster, permittivity_plaster),
((9, 4), (9, 0), thickness_plaster, permittivity_plaster),

((10, 18), (10, 16), thickness_wood, permittivity_wood),
((10, 20), (10, 18), thickness_plaster, permittivity_plaster),
((10, 16), (10, 13), thickness_plaster, permittivity_plaster),
((10, 13), (20, 13), thickness_plaster, permittivity_plaster),

((15, 7), (14, 6), thickness_wood, permittivity_wood),
((20, 2), (20, 4), thickness_glass, permittivity_glass),
((20, 9), (20, 11), thickness_glass, permittivity_glass),
((20, 14), (20, 16), thickness_glass, permittivity_glass),
((20, 0), (20,2 ), thickness_plaster, permittivity_plaster),
((20, 4), (20, 9), thickness_plaster, permittivity_plaster),
((20, 8), (16, 8), thickness_plaster, permittivity_plaster),
((16, 8), (15, 7), thickness_plaster, permittivity_plaster),
((14, 6), (13, 4), thickness_plaster, permittivity_plaster),
((20, 11), (20, 14), thickness_plaster, permittivity_plaster),
((20, 16), (20, 20), thickness_plaster, permittivity_plaster),

]


complexe_t18_obstacles_list=[
((7, 0), (9, 0), thickness_glass, permittivity_glass),
((14, 0), (16, 0), thickness_glass, permittivity_glass),
((0, 2), (0, 3), thickness_glass, permittivity_glass),
((0, 6), (0, 7), thickness_glass, permittivity_glass),
((0, 9), (0, 11), thickness_glass, permittivity_glass),
((0, 16), (0, 18), thickness_glass, permittivity_glass),
((0, 0), (7, 0), thickness_plaster, permittivity_plaster),
((9, 0), (14, 0), thickness_plaster, permittivity_plaster),
((16, 0), (20, 0), thickness_plaster, permittivity_plaster),
((0, 0), (0, 2), thickness_plaster, permittivity_plaster),
((0, 3), (0, 6), thickness_plaster, permittivity_plaster),
((0, 7), (0, 9), thickness_plaster, permittivity_plaster),
((0, 11), (0, 16), thickness_plaster, permittivity_plaster),
((0, 18), (0, 20), thickness_plaster, permittivity_plaster),
((20, 5), (20, 7), thickness_glass, permittivity_glass),
((20,9 ), (20, 11), thickness_glass, permittivity_glass),
((20, 15), (20, 17), thickness_glass, permittivity_glass),
((20, 0), (20, 5), thickness_plaster, permittivity_plaster),
((20, 7), (20, 9), thickness_plaster, permittivity_plaster),
((20, 11), (20, 15), thickness_plaster, permittivity_plaster),
((20, 17), (20, 20), thickness_plaster, permittivity_plaster),
((1, 20), (3, 20), thickness_glass, permittivity_glass),
((7, 20), (9, 20), thickness_glass, permittivity_glass),
((13, 20), (15, 20), thickness_glass, permittivity_glass),
((0, 20), (1, 20), thickness_plaster, permittivity_plaster),
((3, 20), (7, 20), thickness_plaster, permittivity_plaster),
((9, 20), (13, 20), thickness_plaster, permittivity_plaster),
((15, 20), (20, 20), thickness_plaster, permittivity_plaster),

((4, 3), (4, 4), thickness_wood, permittivity_wood),
((4, 6), (4, 7), thickness_wood, permittivity_wood),
((4, 9), (4, 10), thickness_wood, permittivity_wood),
((4, 13), (4,15 ), thickness_wood, permittivity_wood),

((4, 0), (4, 3), thickness_plaster, permittivity_plaster),
((4, 4), (4, 5), thickness_plaster, permittivity_plaster),
((4, 5), (0, 5), thickness_plaster, permittivity_plaster),
((4, 5), (4, 6), thickness_plaster, permittivity_plaster),
((4, 7), (4, 8), thickness_plaster, permittivity_plaster),
((4, 8), (0,8 ), thickness_plaster, permittivity_plaster),
((4, 8), (4, 9), thickness_plaster, permittivity_plaster),
((4, 10), (4, 12), thickness_plaster, permittivity_plaster),
((4, 12), (0, 12), thickness_plaster, permittivity_plaster),
((4, 12), (4, 13), thickness_plaster, permittivity_plaster),
((4, 15), (4, 20), thickness_plaster, permittivity_plaster),

((8, 4), (9, 4), thickness_wood, permittivity_wood),
((6, 0), (6, 4), thickness_plaster, permittivity_plaster),
((6, 4), (8, 4), thickness_plaster, permittivity_plaster),
((9, 4), (11, 4), thickness_plaster, permittivity_plaster),
((11, 4), (11,0 ), thickness_plaster, permittivity_plaster),
((13, 2), (13, 3), thickness_wood, permittivity_wood),
((13, 0), (13, 2), thickness_plaster, permittivity_plaster),
((13, 3), (13, 4), thickness_plaster, permittivity_plaster),
((13, 4), (20, 4), thickness_plaster, permittivity_plaster),

((16, 17), (16, 15), thickness_wood, permittivity_wood),
((16, 9), (16, 7), thickness_wood, permittivity_wood),

((16, 20), (16, 17), thickness_plaster, permittivity_plaster),
((16, 15), (16, 9), thickness_plaster, permittivity_plaster),
((16, 7), (16, 4), thickness_plaster, permittivity_plaster),
((16, 12), (20, 12), thickness_plaster, permittivity_plaster),

]
complexe_t19_obstacles_list=[

((11,1 ), (11, 3), thickness_wood, permittivity_wood),
((2, 0), (4, 0), thickness_glass, permittivity_glass),
((7, 0), (9, 0), thickness_glass, permittivity_glass),
((14, 0), (16, 0), thickness_glass, permittivity_glass),

((0, 0), (2, 0), thickness_plaster, permittivity_plaster),
((4, 0), (7, 0), thickness_plaster, permittivity_plaster),
((9, 0), (14, 0), thickness_plaster, permittivity_plaster),
((16, 0), (20, 0), thickness_plaster, permittivity_plaster),
((11,0 ), (11,1 ), thickness_plaster, permittivity_plaster),
((11, 3), (11, 5), thickness_plaster, permittivity_plaster),
((11, 5), (5, 5), thickness_plaster, permittivity_plaster),
((5, 5), (5, 7), thickness_plaster, permittivity_plaster),
((5, 7), (0, 7), thickness_plaster, permittivity_plaster),

((15, 10), (15, 12), thickness_wood, permittivity_wood),
((20, 4), (20, 6), thickness_glass, permittivity_glass),
((20, 9), (20, 11), thickness_glass, permittivity_glass),

((20, 0), (20, 4), thickness_plaster, permittivity_plaster),
((20, 6), (20, 9), thickness_plaster, permittivity_plaster),
((20, 11), (20, 20), thickness_plaster, permittivity_plaster),
((20, 8), (15, 8), thickness_plaster, permittivity_plaster),
((15, 8), (15, 10), thickness_plaster, permittivity_plaster),
((15, 12), (15, 14), thickness_plaster, permittivity_plaster),
((15, 14), (13, 14), thickness_plaster, permittivity_plaster),
((13, 14), (13, 20), thickness_plaster, permittivity_plaster),

((11, 1), (11, 3), thickness_wood, permittivity_wood),
((11, 0), (11, 1), thickness_plaster, permittivity_plaster),
((11, 3), (11, 5), thickness_plaster, permittivity_plaster),
((11, 5), (5, 5), thickness_plaster, permittivity_plaster),
((5, 5), (5, 7), thickness_plaster, permittivity_plaster),
((5, 7), (0, 7), thickness_plaster, permittivity_plaster),

((5, 13), (7, 14), thickness_wood, permittivity_wood),
((0, 13), (5, 13), thickness_plaster, permittivity_plaster),
((7, 14), (7, 16), thickness_plaster, permittivity_plaster),
((7, 16), (8, 16), thickness_plaster, permittivity_plaster),
((8, 16), (8, 20), thickness_plaster, permittivity_plaster),

((0, 3), (0, 5), thickness_glass, permittivity_glass),
((0, 10), (0, 12), thickness_glass, permittivity_glass),
((0, 16), (0, 18), thickness_glass, permittivity_glass),

((0, 0), (0, 3), thickness_plaster, permittivity_plaster),
((0, 5), (0, 10), thickness_plaster, permittivity_plaster),
((0, 12), (0, 16), thickness_plaster, permittivity_plaster),
((0, 18), (0, 20), thickness_plaster, permittivity_plaster),

((4, 20), (6, 20), thickness_glass, permittivity_glass),
((9, 20), (11, 20), thickness_glass, permittivity_glass),
((15, 20), (17, 20), thickness_glass, permittivity_glass),

((0, 20), (4, 20), thickness_plaster, permittivity_plaster),
((6, 20), (9, 20), thickness_plaster, permittivity_plaster),
((11, 20), (15, 20), thickness_plaster, permittivity_plaster),
((17, 20), (20, 20), thickness_plaster, permittivity_plaster),

]

complexe_t20_obstacles_list=[

((2, 0), (4,0 ), thickness_glass, permittivity_glass),
((8,0 ), (10, 0), thickness_glass, permittivity_glass),
((14, 0), (16, 0), thickness_glass, permittivity_glass),
((0, 0), (2,0 ), thickness_plaster, permittivity_plaster),
((4, 0), (8, 0), thickness_plaster, permittivity_plaster),
((10, 0), (14, 0), thickness_plaster, permittivity_plaster),
((16, 0), (20, 0), thickness_plaster, permittivity_plaster),

((0, 4), (0, 6), thickness_glass, permittivity_glass),
((0, 10), (0, 12), thickness_glass, permittivity_glass),
((0, 15), (0, 17), thickness_glass, permittivity_glass),
((0, 0), (0, 4), thickness_plaster, permittivity_plaster),
((0, 6), (0, 10), thickness_plaster, permittivity_plaster),
((0, 12), (0, 15), thickness_plaster, permittivity_plaster),
((0, 17), (0, 20), thickness_plaster, permittivity_plaster),

((20, 3), (20, 4), thickness_glass, permittivity_glass),
((20, 7), (20, 9), thickness_glass, permittivity_glass),
((20, 13), (20, 15), thickness_glass, permittivity_glass),

((20, 0), (20, 3), thickness_plaster, permittivity_plaster),
((20, 4), (20, 7), thickness_plaster, permittivity_plaster),
((20, 9), (20, 13), thickness_plaster, permittivity_plaster),
((20, 15), (20, 20), thickness_plaster, permittivity_plaster),

((5, 20), (7, 20), thickness_glass, permittivity_glass),
((10, 20), (12, 20), thickness_glass, permittivity_glass),
((15, 20), (17, 20), thickness_glass, permittivity_glass),

((0, 20), (5, 20), thickness_plaster, permittivity_plaster),
((7, 20), (10, 20), thickness_plaster, permittivity_plaster),
((12, 20), (15, 20), thickness_plaster, permittivity_plaster),
((17, 20), (20, 20), thickness_plaster, permittivity_plaster),

((10, 4), (8, 4), thickness_wood, permittivity_wood),
((5, 4), (3,4 ), thickness_wood, permittivity_wood),

((20, 5), (17, 5), thickness_plaster, permittivity_plaster),
((12,5 ), (15, 5), thickness_plaster, permittivity_plaster),
((12, 5), (12, 0), thickness_plaster, permittivity_plaster),
((12, 4), (10, 4), thickness_plaster, permittivity_plaster),
((8, 4), (6, 4), thickness_plaster, permittivity_plaster),
((6, 4), (6, 0), thickness_plaster, permittivity_plaster),
((6, 4), (5, 4), thickness_plaster, permittivity_plaster),
((3, 4), (2, 3), thickness_plaster, permittivity_plaster),
((2, 3), (0, 3), thickness_plaster, permittivity_plaster),
((20, 10), (16, 10), thickness_plaster, permittivity_plaster),
((16, 10), (16, 17), thickness_plaster, permittivity_plaster),

((3, 13), (5, 13), thickness_wood, permittivity_wood),
((13, 17), (13, 19), thickness_wood, permittivity_wood),
((0, 13), (3, 13), thickness_plaster, permittivity_plaster),
((5, 13), (8, 13), thickness_plaster, permittivity_plaster),
((8,13 ), (8, 15), thickness_plaster, permittivity_plaster),
((8, 15), (13, 15), thickness_plaster, permittivity_plaster),
((13, 15), (13, 17), thickness_plaster, permittivity_plaster),
((13, 19), (13, 20), thickness_plaster, permittivity_plaster),

]

complexe_t21_obstacles_list=[

((3, 0), (5, 0), thickness_glass, permittivity_glass),
((9, 0), (11, 0), thickness_glass, permittivity_glass),
((14, 0), (16, 0), thickness_glass, permittivity_glass),
((20, 4), (20, 6), thickness_glass, permittivity_glass),
((20, 10), (20,12 ), thickness_glass, permittivity_glass),
((20, 15), (20, 17), thickness_glass, permittivity_glass),

((0,0 ), (3, 0), thickness_plaster, permittivity_plaster),
((5, 0), (9, 0), thickness_plaster, permittivity_plaster),
((11, 0), (14, 0), thickness_plaster, permittivity_plaster),
((16, 0), (20, 0), thickness_plaster, permittivity_plaster),
((20, 0), (20, 4), thickness_plaster, permittivity_plaster),
((20, 6), (20, 10), thickness_plaster, permittivity_plaster),
((20, 12), (20, 15), thickness_plaster, permittivity_plaster),
((20, 17), (20, 20), thickness_plaster, permittivity_plaster),

((0, 3), (0, 5), thickness_glass, permittivity_glass),
((0, 9), (0,11 ), thickness_glass, permittivity_glass),
((0,15 ), (0, 17), thickness_glass, permittivity_glass),

((0, 0), (0, 3), thickness_plaster, permittivity_plaster),
((0, 5), (0, 9), thickness_plaster, permittivity_plaster),
((0, 11), (0, 15), thickness_plaster, permittivity_plaster),
((0, 17), (0,20), thickness_plaster, permittivity_plaster),

((3, 20), (5, 20), thickness_glass, permittivity_glass),
((8, 20), (10, 20), thickness_glass, permittivity_glass),
((14, 20), (16, 20), thickness_glass, permittivity_glass),
((0, 20), (3, 20), thickness_plaster, permittivity_plaster),
((5, 20), (8, 20), thickness_plaster, permittivity_plaster),
((10, 20), (14, 20), thickness_plaster, permittivity_plaster),
((16, 20), (20, 20), thickness_plaster, permittivity_plaster),

((12, 4), (12, 6), thickness_wood, permittivity_wood),
((7, 15), (7, 17), thickness_wood, permittivity_wood),

((7, 0), (7, 3), thickness_plaster, permittivity_plaster),
((7, 3), (12, 3), thickness_plaster, permittivity_plaster),
((12, 3), (12, 4), thickness_plaster, permittivity_plaster),
((12, 6), (12, 8), thickness_plaster, permittivity_plaster),
((12, 8), (20, 8), thickness_plaster, permittivity_plaster),
((0, 7), (7, 7), thickness_plaster, permittivity_plaster),
((7, 7), (7, 9), thickness_plaster, permittivity_plaster),
((7, 9), (8, 9), thickness_plaster, permittivity_plaster),
((7, 11), (8, 11), thickness_plaster, permittivity_plaster),
((7, 11), (7, 15), thickness_plaster, permittivity_plaster),
((7, 13), (0, 13), thickness_plaster, permittivity_plaster),
((7, 17), (7, 20), thickness_plaster, permittivity_plaster),

((11, 16), (12, 15), thickness_wood, permittivity_wood),
((12, 13), (20, 13), thickness_plaster, permittivity_plaster),
((12, 13), (12, 15), thickness_plaster, permittivity_plaster),
((11, 16), (11,20 ), thickness_plaster, permittivity_plaster),

]

complexe_t22_obstacles_list=[

((3, 0), (5, 0), thickness_glass, permittivity_glass),
((9, 0), (11, 0), thickness_glass, permittivity_glass),
((14, 0), (16, 0), thickness_glass, permittivity_glass),
((20, 3), (20, 5), thickness_glass, permittivity_glass),
((20, 9), (20, 11), thickness_glass, permittivity_glass),
((20, 15), (20, 17), thickness_glass, permittivity_glass),
((0, 0), (3,0 ), thickness_plaster, permittivity_plaster),
((5, 0), (9, 0), thickness_plaster, permittivity_plaster),
((11, 0), (14, 0), thickness_plaster, permittivity_plaster),
((16, 0), (20, 0), thickness_plaster, permittivity_plaster),
((20, 0), (20,3 ), thickness_plaster, permittivity_plaster),
((20, 5), (20, 9), thickness_plaster, permittivity_plaster),
((20, 11), (20, 15), thickness_plaster, permittivity_plaster),
((20, 17), (20, 20), thickness_plaster, permittivity_plaster),

((0, 5), (0, 7), thickness_glass, permittivity_glass),
((0, 10), (0, 12), thickness_glass, permittivity_glass),
((0, 15), (0, 17), thickness_glass, permittivity_glass),
((3, 20), (5, 20), thickness_glass, permittivity_glass),
((8, 20), (10, 20), thickness_glass, permittivity_glass),
((14, 20), (16, 20), thickness_glass, permittivity_glass),


((0, 0), (0, 5), thickness_plaster, permittivity_plaster),
((0,7 ), (0, 10), thickness_plaster, permittivity_plaster),
((0, 12), (0, 15), thickness_plaster, permittivity_plaster),
((0, 17), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (3, 20), thickness_plaster, permittivity_plaster),
((5, 20), (8, 20), thickness_plaster, permittivity_plaster),
((10, 20), (14, 20), thickness_plaster, permittivity_plaster),
((16, 20), (20, 20), thickness_plaster, permittivity_plaster),

((7, 6), (7, 8), thickness_wood, permittivity_wood),
((6, 16), (6, 18), thickness_wood, permittivity_wood),
((7, 0), (7, 6), thickness_plaster, permittivity_plaster),
((7, 8), (7, 12), thickness_plaster, permittivity_plaster),
((7, 12), (6, 13), thickness_plaster, permittivity_plaster),
((6, 13), (6, 16), thickness_plaster, permittivity_plaster),
((6, 14), (0, 14), thickness_plaster, permittivity_plaster),
((6, 18), (6, 20), thickness_plaster, permittivity_plaster),


((12, 3), (12, 5), thickness_wood, permittivity_wood),
((12, 8), (12, 10), thickness_wood, permittivity_wood),
((15, 13), (17, 13), thickness_wood, permittivity_wood),

((12, 0), (12, 3), thickness_plaster, permittivity_plaster),
((12, 5), (12, 8), thickness_plaster, permittivity_plaster),
((12, 10), (12, 13), thickness_plaster, permittivity_plaster),
((12, 13), (15, 13), thickness_plaster, permittivity_plaster),
((17, 13), (20, 13), thickness_plaster, permittivity_plaster),

]


complexe_t23_obstacles_list=[

((0, 3), (0, 5), thickness_glass, permittivity_glass),
((12, 0), (14, 0), thickness_glass, permittivity_glass),
((0, 11), (0, 13), thickness_glass, permittivity_glass),
((0, 16), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (10, 20), thickness_glass, permittivity_glass),
((15, 20), (17, 20), thickness_glass, permittivity_glass),

((0, 0), (12,0 ), thickness_plaster, permittivity_plaster),
((14, 0), (20, 0), thickness_plaster, permittivity_plaster),
((0, 0), (0, 3), thickness_plaster, permittivity_plaster),
((0, 5), (0, 11), thickness_plaster, permittivity_plaster),
((0, 13), (0, 16), thickness_plaster, permittivity_plaster),
((10, 20), (15, 20), thickness_plaster, permittivity_plaster),
((17, 20), (20, 20), thickness_plaster, permittivity_plaster),

((20, 4), (20, 6), thickness_glass, permittivity_glass),
((20, 11), (20, 13), thickness_glass, permittivity_glass),
((20, 17), (20, 19), thickness_glass, permittivity_glass),
((20, 0), (20, 4), thickness_plaster, permittivity_plaster),
((20, 6), (20, 11), thickness_plaster, permittivity_plaster),
((20, 13), (20, 17), thickness_plaster, permittivity_plaster),
((20, 19), (20, 20), thickness_plaster, permittivity_plaster),

((4, 4), (4, 6), thickness_wood, permittivity_wood),
((5, 0), (5,3), thickness_plaster, permittivity_plaster),
((5, 3), (4, 4), thickness_plaster, permittivity_plaster),
((4, 6), (0, 6), thickness_plaster, permittivity_plaster),

((6, 14), (8, 14), thickness_wood, permittivity_wood),
((0, 14), (6, 14), thickness_plaster, permittivity_plaster),
((8, 14), (10, 14), thickness_plaster, permittivity_plaster),
((10, 14), (10, 20), thickness_plaster, permittivity_plaster),

((10, 7), (12, 7), thickness_wood, permittivity_wood),
((13, 10), (13, 12), thickness_wood, permittivity_wood),
((13, 15), (13, 17), thickness_wood, permittivity_wood),

((8, 0), (8, 7), thickness_plaster, permittivity_plaster),
((8, 7), (10,7 ), thickness_plaster, permittivity_plaster),
((12, 7), (13, 8), thickness_plaster, permittivity_plaster),
((13, 8), (20, 8), thickness_plaster, permittivity_plaster),
((13, 8), (13, 10), thickness_plaster, permittivity_plaster),
((13, 12), (13, 15), thickness_plaster, permittivity_plaster),
((13, 17), (13, 20), thickness_plaster, permittivity_plaster),
]

complexe_t24_obstacles_list=[

((2, 0), (4, 0), thickness_glass, permittivity_glass),
((10, 0), (12, 0), thickness_glass, permittivity_glass),
((0, 7), (0, 9), thickness_glass, permittivity_glass),
((0, 15), (0, 17), thickness_glass, permittivity_glass),
((7, 20), (9, 20), thickness_glass, permittivity_glass),
((14, 20), (16, 20), thickness_glass, permittivity_glass),

((0, 0), (2, 0), thickness_plaster, permittivity_plaster),
((4, 0), (10, 0), thickness_plaster, permittivity_plaster),
((12, 0), (20, 0), thickness_plaster, permittivity_plaster),
((0,0 ), (0,7 ), thickness_plaster, permittivity_plaster),
((0, 9), (0, 15), thickness_plaster, permittivity_plaster),
((0, 17), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (7, 20), thickness_plaster, permittivity_plaster),
((9, 20), (14, 20), thickness_plaster, permittivity_plaster),
((16, 20), (20, 20), thickness_plaster, permittivity_plaster),

((20, 4), (20, 6), thickness_glass, permittivity_glass),
((20, 12), (20, 14), thickness_glass, permittivity_glass),
((20, 0), (20, 4), thickness_plaster, permittivity_plaster),
((20, 6), (20, 12), thickness_plaster, permittivity_plaster),
((20, 14), (20, 20), thickness_plaster, permittivity_plaster),

((7, 0), (7, 6), thickness_glass, permittivity_glass),
((7, 6), (3, 6), thickness_glass, permittivity_glass),

((13, 4), (13, 6), thickness_wood, permittivity_wood),
((9, 0), (9, 3), thickness_plaster, permittivity_plaster),
((9, 3), (13, 3), thickness_plaster, permittivity_plaster),
((13, 3), (13, 4), thickness_plaster, permittivity_plaster),
((13, 6), (13, 7), thickness_plaster, permittivity_plaster),
((13, 7), (20,7 ), thickness_plaster, permittivity_plaster),

((14, 14), (14, 16), thickness_wood, permittivity_wood),
((20, 10), (14, 10), thickness_plaster, permittivity_plaster),
((14, 10), (14, 14), thickness_plaster, permittivity_plaster),
((14, 16), (14, 20), thickness_plaster, permittivity_plaster),

((8, 13), (8, 15), thickness_wood, permittivity_wood),
((10, 17), (10, 19), thickness_wood, permittivity_wood),
((0, 12), (8, 12), thickness_plaster, permittivity_plaster),
((8, 12), (8, 13), thickness_plaster, permittivity_plaster),
((8, 15), (10, 15), thickness_plaster, permittivity_plaster),
((10, 15), (10, 17), thickness_plaster, permittivity_plaster),
((10, 19), (10, 20), thickness_plaster, permittivity_plaster),

]

basic12_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),
((7,0 ), (7, 20), thickness_concrete, permittivity_concrete),
((9, 0), (9, 20), thickness_concrete, permittivity_concrete),
((12,0 ), (12, 20), thickness_concrete, permittivity_concrete),
((14, 0), (14, 20), thickness_concrete, permittivity_concrete),
]

basic12_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((7,0 ), (7, 20), thickness_wood, permittivity_wood),
((9, 0), (9, 20), thickness_wood, permittivity_wood),
((12,0 ), (12, 20), thickness_wood, permittivity_wood),
((14, 0), (14, 20), thickness_wood, permittivity_wood),
]

basic12_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),
((7,0 ), (7, 20), thickness_glass, permittivity_glass),
((9, 0), (9, 20), thickness_glass, permittivity_glass),
((12,0 ), (12, 20), thickness_glass, permittivity_glass),
((14, 0), (14, 20), thickness_glass, permittivity_glass),
]

basic12_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
((7,0 ), (7, 20), thickness_brick, permittivity_brick),
((9, 0), (9, 20), thickness_brick, permittivity_brick),
((12,0 ), (12, 20), thickness_brick, permittivity_brick),
((14, 0), (14, 20), thickness_brick, permittivity_brick),
]



basic12_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),

((0, 0), (20,0 ), thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((7,0 ), (7, 20), thickness_plaster, permittivity_plaster),
((9, 0), (9, 20), thickness_plaster, permittivity_plaster),
((12,0 ), (12, 20), thickness_plaster, permittivity_plaster),
((14, 0), (14, 20), thickness_plaster, permittivity_plaster),
]

basic11_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),
((6,6 ), (14, 6), thickness_concrete, permittivity_concrete),
((6, 6), (6, 14), thickness_concrete, permittivity_concrete),
((6, 14), (14, 14), thickness_concrete, permittivity_concrete),
((14, 6), (14, 14), thickness_concrete, permittivity_concrete),
((8, 8), (12, 8), thickness_concrete, permittivity_concrete),
((8, 8), (8, 12), thickness_concrete, permittivity_concrete),
((8, 12), (12, 12), thickness_concrete, permittivity_concrete),
((12, 12), (12, 8), thickness_concrete, permittivity_concrete),
]

basic11_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((6,6 ), (14, 6), thickness_wood, permittivity_wood),
((6, 6), (6, 14), thickness_wood, permittivity_wood),
((6, 14), (14, 14), thickness_wood, permittivity_wood),
((14, 6), (14, 14), thickness_wood, permittivity_wood),
((8, 8), (12, 8), thickness_wood, permittivity_wood),
((8, 8), (8, 12), thickness_wood, permittivity_wood),
((8, 12), (12, 12), thickness_wood, permittivity_wood),
((12, 12), (12, 8), thickness_wood, permittivity_wood),
]

basic11_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),
((6,6 ), (14, 6), thickness_glass, permittivity_glass),
((6, 6), (6, 14), thickness_glass, permittivity_glass),
((6, 14), (14, 14), thickness_glass, permittivity_glass),
((14, 6), (14, 14), thickness_glass, permittivity_glass),
((8, 8), (12, 8), thickness_glass, permittivity_glass),
((8, 8), (8, 12), thickness_glass, permittivity_glass),
((8, 12), (12, 12), thickness_glass, permittivity_glass),
((12, 12), (12, 8), thickness_glass, permittivity_glass),
]


basic11_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
((6,6 ), (14, 6), thickness_brick, permittivity_brick),
((6, 6), (6, 14), thickness_brick, permittivity_brick),
((6, 14), (14, 14), thickness_brick, permittivity_brick),
((14, 6), (14, 14), thickness_brick, permittivity_brick),
((8, 8), (12, 8), thickness_brick, permittivity_brick),
((8, 8), (8, 12), thickness_brick, permittivity_brick),
((8, 12), (12, 12), thickness_brick, permittivity_brick),
((12, 12), (12, 8), thickness_brick, permittivity_brick),
]


basic11_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((6,6 ), (14, 6), thickness_plaster, permittivity_plaster),
((6, 6), (6, 14), thickness_plaster, permittivity_plaster),
((6, 14), (14, 14), thickness_plaster, permittivity_plaster),
((14, 6), (14, 14), thickness_plaster, permittivity_plaster),
((8, 8), (12, 8), thickness_plaster, permittivity_plaster),
((8, 8), (8, 12), thickness_plaster, permittivity_plaster),
((8, 12), (12, 12), thickness_plaster, permittivity_plaster),
((12, 12), (12, 8), thickness_plaster, permittivity_plaster),

]


basic10_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((10, 0), (10, 1), thickness_plaster, permittivity_plaster),
((10, 1), (8, 1), thickness_plaster, permittivity_plaster),
((8,1 ), (8, 2), thickness_plaster, permittivity_plaster),
((8, 2), (6, 2), thickness_plaster, permittivity_plaster),
((6, 2), (6, 4), thickness_plaster, permittivity_plaster),
((6, 4), (4,  4), thickness_plaster, permittivity_plaster),
((4, 4), (4, 8), thickness_plaster, permittivity_plaster),
((4, 8), (2, 8), thickness_plaster, permittivity_plaster),
((2, 8), (2, 10), thickness_plaster, permittivity_plaster),
((2, 10), (0, 10), thickness_plaster, permittivity_plaster),
((12, 0), (12, 3), thickness_plaster, permittivity_plaster),
((12, 3), (17, 3), thickness_plaster, permittivity_plaster),
((17, 3), (17, 7), thickness_plaster, permittivity_plaster),
((0, 12), (14, 12), thickness_plaster, permittivity_plaster),
((14,12 ), (14, 16), thickness_plaster, permittivity_plaster),
((0, 18), (10, 18), thickness_plaster, permittivity_plaster),
((10, 18), (10, 15), thickness_plaster, permittivity_plaster),
]


basic10_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((10, 0), (10, 1), thickness_wood, permittivity_wood),
((10, 1), (8, 1), thickness_wood, permittivity_wood),
((8,1 ), (8, 2), thickness_wood, permittivity_wood),
((8, 2), (6, 2), thickness_wood, permittivity_wood),
((6, 2), (6, 4), thickness_wood, permittivity_wood),
((6, 4), (4,  4), thickness_wood, permittivity_wood),
((4, 4), (4, 8), thickness_wood, permittivity_wood),
((4, 8), (2, 8), thickness_wood, permittivity_wood),
((2, 8), (2, 10), thickness_wood, permittivity_wood),
((2, 10), (0, 10), thickness_wood, permittivity_wood),
((12, 0), (12, 3), thickness_wood, permittivity_wood),
((12, 3), (17, 3), thickness_wood, permittivity_wood),
((17, 3), (17, 7), thickness_wood, permittivity_wood),
((0, 12), (14, 12), thickness_wood, permittivity_wood),
((14,12 ), (14, 16), thickness_wood, permittivity_wood),
((0, 18), (10, 18), thickness_wood, permittivity_wood),
((10, 18), (10, 15), thickness_wood, permittivity_wood),
]


basic10_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),
((10, 0), (10, 1), thickness_glass, permittivity_glass),
((10, 1), (8, 1), thickness_glass, permittivity_glass),
((8,1 ), (8, 2), thickness_glass, permittivity_glass),
((8, 2), (6, 2), thickness_glass, permittivity_glass),
((6, 2), (6, 4), thickness_glass, permittivity_glass),
((6, 4), (4,  4), thickness_glass, permittivity_glass),
((4, 4), (4, 8), thickness_glass, permittivity_glass),
((4, 8), (2, 8), thickness_glass, permittivity_glass),
((2, 8), (2, 10), thickness_glass, permittivity_glass),
((2, 10), (0, 10), thickness_glass, permittivity_glass),
((12, 0), (12, 3), thickness_glass, permittivity_glass),
((12, 3), (17, 3), thickness_glass, permittivity_glass),
((17, 3), (17, 7), thickness_glass, permittivity_glass),
((0, 12), (14, 12), thickness_glass, permittivity_glass),
((14,12 ), (14, 16), thickness_glass, permittivity_glass),
((0, 18), (10, 18), thickness_glass, permittivity_glass),
((10, 18), (10, 15), thickness_glass, permittivity_glass),
]



basic10_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
((10, 0), (10, 1), thickness_brick, permittivity_brick),
((10, 1), (8, 1), thickness_brick, permittivity_brick),
((8,1 ), (8, 2), thickness_brick, permittivity_brick),
((8, 2), (6, 2), thickness_brick, permittivity_brick),
((6, 2), (6, 4), thickness_brick, permittivity_brick),
((6, 4), (4,  4), thickness_brick, permittivity_brick),
((4, 4), (4, 8), thickness_brick, permittivity_brick),
((4, 8), (2, 8), thickness_brick, permittivity_brick),
((2, 8), (2, 10), thickness_brick, permittivity_brick),
((2, 10), (0, 10), thickness_brick, permittivity_brick),
((12, 0), (12, 3), thickness_brick, permittivity_brick),
((12, 3), (17, 3), thickness_brick, permittivity_brick),
((17, 3), (17, 7), thickness_brick, permittivity_brick),
((0, 12), (14, 12), thickness_brick, permittivity_brick),
((14,12 ), (14, 16), thickness_brick, permittivity_brick),
((0, 18), (10, 18), thickness_brick, permittivity_brick),
((10, 18), (10, 15), thickness_brick, permittivity_brick),
]



basic9_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((0, 9), (17, 9), thickness_plaster, permittivity_plaster),
((0, 12), (20, 12), thickness_plaster, permittivity_plaster),
((0, 18), (6, 18), thickness_plaster, permittivity_plaster),
((5,16 ), (16, 16), thickness_plaster, permittivity_plaster),
((17, 9), (17, 2), thickness_plaster, permittivity_plaster),
((18, 20), (18, 14), thickness_plaster, permittivity_plaster),

]



basic9_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((0, 9), (17, 9), thickness_wood, permittivity_wood),
((0, 12), (20, 12), thickness_wood, permittivity_wood),
((0, 18), (6, 18), thickness_wood, permittivity_wood),
((5,16 ), (16, 16), thickness_wood, permittivity_wood),
((17, 9), (17, 2), thickness_wood, permittivity_wood),
((18, 20), (18, 14), thickness_wood, permittivity_wood),
]



basic9_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),
((0, 9), (17, 9), thickness_glass, permittivity_glass),
((0, 12), (20, 12), thickness_glass, permittivity_glass),
((0, 18), (6, 18), thickness_glass, permittivity_glass),
((5,16 ), (16, 16), thickness_glass, permittivity_glass),
((17, 9), (17, 2), thickness_glass, permittivity_glass),
((18, 20), (18, 14), thickness_glass, permittivity_glass),
]


basic9_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
((0, 9), (17, 9), thickness_brick, permittivity_brick),
((0, 12), (20, 12), thickness_brick, permittivity_brick),
((0, 18), (6, 18), thickness_brick, permittivity_brick),
((5,16 ), (16, 16), thickness_brick, permittivity_brick),
((17, 9), (17, 2), thickness_brick, permittivity_brick),
((18, 20), (18, 14), thickness_brick, permittivity_brick),
]


basic8_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),
((5, 4), (5, 20), thickness_concrete, permittivity_concrete),
((7, 0), (7, 17), thickness_concrete, permittivity_concrete),
((11, 1), (11,19 ), thickness_concrete, permittivity_concrete),
]

basic8_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((5, 4), (5, 20), thickness_wood, permittivity_wood),
((7, 0), (7, 17), thickness_wood, permittivity_wood),
((11, 1), (11,19 ), thickness_wood, permittivity_wood),
]


basic8_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),
((5, 4), (5, 20), thickness_glass, permittivity_glass),
((7, 0), (7, 17), thickness_glass, permittivity_glass),
((11, 1), (11,19 ), thickness_glass, permittivity_glass),
]

basic8_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
((5, 4), (5, 20), thickness_brick, permittivity_brick),
((7, 0), (7, 17), thickness_brick, permittivity_brick),
((11, 1), (11,19 ), thickness_brick, permittivity_brick),
]


basic8_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((5, 4), (5, 20), thickness_plaster, permittivity_plaster),
((7, 0), (7, 17), thickness_plaster, permittivity_plaster),
((11, 1), (11,19 ), thickness_plaster, permittivity_plaster),
]

basic7_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),
((7, 0), (7, 10), thickness_concrete, permittivity_concrete),
((13, 11), (13, 20), thickness_concrete, permittivity_concrete),
]

basic7_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((7, 0), (7, 10), thickness_wood, permittivity_wood),
((13, 11), (13, 20), thickness_wood, permittivity_wood),
]



basic7_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),
((7, 0), (7, 10), thickness_glass, permittivity_glass),
((13, 11), (13, 20), thickness_glass, permittivity_glass),
]

basic7_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
((7, 0), (7, 10), thickness_brick, permittivity_brick),
((13, 11), (13, 20), thickness_brick, permittivity_brick),
]

basic7_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((7, 0), (7, 10), thickness_plaster, permittivity_plaster),
((13, 11), (13, 20), thickness_plaster, permittivity_plaster),
]


basic6_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),
((1, 7), (10 ,7), thickness_concrete, permittivity_concrete),
((1, 13), ( 18 ,13), thickness_concrete, permittivity_concrete),
]


basic6_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((1, 7), (10 ,7), thickness_wood, permittivity_wood),
((1, 13), ( 18 ,13), thickness_wood, permittivity_wood),
]

basic6_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),
((1, 7), (10 ,7), thickness_glass, permittivity_glass),
((1, 13), ( 18 ,13), thickness_glass, permittivity_glass),
]

basic6_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
((1, 7), (10 ,7), thickness_brick, permittivity_brick),
((1, 13), ( 18 ,13),thickness_brick, permittivity_brick),
]

basic6_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((1, 7), (10 ,7), thickness_plaster, permittivity_plaster),
((1, 13), ( 18 ,13),thickness_plaster, permittivity_plaster),
]

basic5_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),
((10, 0), (10, 5), thickness_concrete, permittivity_concrete),
((3, 20), (3, 17), thickness_concrete, permittivity_concrete),
((3, 17), (10, 17), thickness_concrete, permittivity_concrete),
]


basic5_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),   thickness_wood, permittivity_wood),
((0,0 ), (0, 20),  thickness_wood, permittivity_wood),
((0, 20), (20, 20),  thickness_wood, permittivity_wood),
((20, 20), (20, 0),  thickness_wood, permittivity_wood),
((10, 0), (10, 5),  thickness_wood, permittivity_wood),
((3, 20), (3, 17),  thickness_wood, permittivity_wood),
((3, 17), (10, 17),  thickness_wood, permittivity_wood),
]

basic5_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),   thickness_glass, permittivity_glass),
((0,0 ), (0, 20),  thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0),  thickness_glass, permittivity_glass),
((10, 0), (10, 5), thickness_glass, permittivity_glass),
((3, 20), (3, 17),  thickness_glass, permittivity_glass),
((3, 17), (10, 17),  thickness_glass, permittivity_glass),
]


basic5_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_brick, permittivity_brick),
((0,0 ), (0, 20),  thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0),  thickness_brick, permittivity_brick),
((10, 0), (10, 5), thickness_brick, permittivity_brick),
((3, 20), (3, 17),  thickness_brick, permittivity_brick),
((3, 17), (10, 17),  thickness_brick, permittivity_brick),
]

basic5_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20),  thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0),  thickness_plaster, permittivity_plaster),
((10, 0), (10, 5), thickness_plaster, permittivity_plaster),
((3, 20), (3, 17), thickness_plaster, permittivity_plaster),
((3, 17), (10, 17),  thickness_plaster, permittivity_plaster),
]


basic4_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),
((0, 2), (8, 2), thickness_concrete, permittivity_concrete),
((8, 2), (8, 3), thickness_concrete, permittivity_concrete),
((15, 20), (15, 15), thickness_concrete, permittivity_concrete),
((15, 15), (20, 15), thickness_concrete, permittivity_concrete),
]



basic4_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((0, 2), (8, 2), thickness_wood, permittivity_wood),
((8, 2), (8, 3), thickness_wood, permittivity_wood),
((15, 20), (15, 15),thickness_wood, permittivity_wood),
((15, 15), (20, 15), thickness_wood, permittivity_wood),
]


basic4_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_glass, permittivity_glass),
((0,0 ), (0, 20),thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),
((0, 2), (8, 2), thickness_glass, permittivity_glass),
((8, 2), (8, 3), thickness_glass, permittivity_glass),
((15, 20), (15, 15), thickness_glass, permittivity_glass),
((15, 15), (20, 15),thickness_glass, permittivity_glass),
]


basic4_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
((0, 2), (8, 2), thickness_brick, permittivity_brick),
((8, 2), (8, 3), thickness_brick, permittivity_brick),
((15, 20), (15, 15), thickness_brick, permittivity_brick),
((15, 15), (20, 15), thickness_brick, permittivity_brick),
]


basic4_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((0, 2), (8, 2), thickness_plaster, permittivity_plaster),
((8, 2), (8, 3), thickness_plaster, permittivity_plaster),
((15, 20), (15, 15), thickness_plaster, permittivity_plaster),
((15, 15), (20, 15), thickness_plaster, permittivity_plaster),
]



basic3_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),

((8, 12), (12, 12), thickness_concrete, permittivity_concrete),
((8, 8), (12, 8), thickness_concrete, permittivity_concrete),
((8, 8), (8, 12), thickness_concrete, permittivity_concrete),
((12, 8), (12, 12), thickness_concrete, permittivity_concrete),
]


basic3_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),
((8, 8), (12, 8), thickness_wood, permittivity_wood),
((8, 8), (8, 12), thickness_wood, permittivity_wood),
((12, 8), (12, 12), thickness_wood, permittivity_wood),
((8, 12), (12, 12), thickness_wood, permittivity_wood),
]


basic3_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0),thickness_glass, permittivity_glass),
((8, 8), (12, 8), thickness_glass, permittivity_glass),
((8, 8), (8, 12), thickness_glass, permittivity_glass),
((12, 8), (12, 12), thickness_glass, permittivity_glass),
((8, 12), (12, 12), thickness_glass, permittivity_glass),
]


basic3_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0),thickness_brick, permittivity_brick),
((8, 8), (12, 8), thickness_brick, permittivity_brick),
((8, 8), (8, 12), thickness_brick, permittivity_brick),
((12, 8), (12, 12), thickness_brick, permittivity_brick),
((8, 12), (12, 12), thickness_brick, permittivity_brick),
]

basic3_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((8, 8), (12, 8), thickness_plaster, permittivity_plaster),
((8, 8), (8, 12), thickness_plaster, permittivity_plaster),
((12, 8), (12, 12), thickness_plaster, permittivity_plaster),
((8, 12), (12, 12), thickness_plaster, permittivity_plaster),
]


basic2_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20), thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),
((7, 0), (7, 20), thickness_concrete, permittivity_concrete),
((13, 0), (13, 20), thickness_concrete, permittivity_concrete),
]


basic2_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((7, 0), (7, 20), thickness_concrete, permittivity_concrete),
((13, 0), (13, 20), thickness_concrete, permittivity_concrete),
]


basic2_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((7, 0), (7, 20), thickness_wood, permittivity_wood),
((13, 0), (13, 20), thickness_wood, permittivity_wood),
]



basic2_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((7, 0), (7, 20), thickness_glass, permittivity_glass),
((13, 0), (13, 20), thickness_glass, permittivity_glass),
]


basic2_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((7, 0), (7, 20), thickness_brick, permittivity_brick),
((13, 0), (13, 20), thickness_brick, permittivity_brick),
]


basic2_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((7, 0), (7, 20), thickness_plaster, permittivity_plaster),
((13, 0), (13, 20), thickness_plaster, permittivity_plaster),
]





basic1_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((9, 0), (9, 20), thickness_concrete, permittivity_concrete),
((11, 0), (11, 20), thickness_concrete, permittivity_concrete),
]


basic1_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((9, 0), (9, 20), thickness_wood, permittivity_wood),
((11, 0), (11, 20), thickness_wood, permittivity_wood),
]



basic1_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((9, 0), (9, 20), thickness_glass, permittivity_glass),
((11, 0), (11, 20), thickness_glass, permittivity_glass),
]


basic1_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((9, 0), (9, 20), thickness_brick, permittivity_brick),
((11, 0), (11, 20), thickness_brick, permittivity_brick),
]


basic1_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),
((9, 0), (9, 20), thickness_plaster, permittivity_plaster),
((11, 0), (11, 20), thickness_plaster, permittivity_plaster),
]






basic0_sc0_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_concrete, permittivity_concrete),
((0,0 ), (0, 20),thickness_concrete, permittivity_concrete),
((0, 20), (20, 20), thickness_concrete, permittivity_concrete),
((20, 20), (20, 0), thickness_concrete, permittivity_concrete),

]


basic0_sc1_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_wood, permittivity_wood),
((0,0 ), (0, 20), thickness_wood, permittivity_wood),
((0, 20), (20, 20), thickness_wood, permittivity_wood),
((20, 20), (20, 0), thickness_wood, permittivity_wood),

]



basic0_sc2_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ), thickness_glass, permittivity_glass),
((0,0 ), (0, 20), thickness_glass, permittivity_glass),
((0, 20), (20, 20), thickness_glass, permittivity_glass),
((20, 20), (20, 0), thickness_glass, permittivity_glass),

]


basic0_sc3_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_brick, permittivity_brick),
((0,0 ), (0, 20), thickness_brick, permittivity_brick),
((0, 20), (20, 20), thickness_brick, permittivity_brick),
((20, 20), (20, 0), thickness_brick, permittivity_brick),
]



basic0_sc4_obstacles_list=[
#((, ), (, ), thickness_plaster, permittivity_plaster),
#((, ), (, ), thickness_glass, permittivity_glass),
#((, ), (, ), thickness_brick, permittivity_brick),
#((, ), (, ), thickness_wood, permittivity_wood),
#((, ), (, ), thickness_concrete, permittivity_concrete),
((0, 0), (20,0 ),  thickness_plaster, permittivity_plaster),
((0,0 ), (0, 20), thickness_plaster, permittivity_plaster),
((0, 20), (20, 20), thickness_plaster, permittivity_plaster),
((20, 20), (20, 0), thickness_plaster, permittivity_plaster),

]

# Generate the obstacle matrix
#obstacle_matrix = generate_obstacle_matrix(basic12_sc4_obstacles_list)
#unique_values = np.unique(obstacle_matrix)
#print(unique_values)
#visualize_matrix(obstacle_matrix)
#folder_path = 'C:\/Users\/ZAIMEN\/PycharmProjects\/GenerateCNNinput\/computedTensors\/basic_t12\/sc4'
#file_path = os.path.join(folder_path, 'conductivity.csv')
#np.savetxt(file_path, obstacle_matrix, delimiter=',', fmt='%f' )


#print(unique_values)
# Display the final matrix
from scipy.ndimage import rotate
data_list = []
titles = []


def plot_multiple_imshow(*args,titles, rows, cols, figsize=(12, 8), cmap='viridis'):
    data_list.append(obstacle_matrix)
    titles.append("original")
    flipped_left_matrix = np.rot90(obstacle_matrix, k=1)
    data_list.append(flipped_left_matrix)
    titles.append("rot k1")
    flipped_left_matrix = np.rot90(obstacle_matrix, k=2)
    data_list.append(flipped_left_matrix)
    titles.append("rot k2")
    flipped_left_matrix = np.rot90(obstacle_matrix, k=3)
    data_list.append(flipped_left_matrix)
    titles.append("rot k3")
    flipped_left_matrix = np.flipud(obstacle_matrix)
    data_list.append(flipped_left_matrix)
    titles.append("flipdown")
    flipped_left_matrix = np.rot90(flipped_left_matrix, k=1)
    data_list.append(flipped_left_matrix)
    titles.append("flip down + rot k1")
    flipped_left_matrix = np.fliplr(obstacle_matrix)
    data_list.append(flipped_left_matrix)
    titles.append("flip left")
    flipped_left_matrix = np.rot90(flipped_left_matrix, k=1)
    data_list.append(flipped_left_matrix)
    titles.append("flip left+ rot k1")

    # Create a grid of subplots
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Plot each imshow in a subplot
    for i, data in enumerate(args):
        axs[i].imshow(data, cmap=cmap)
        axs[i].set_title(titles[i])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

# Example usage:
# Create some example data and titles


# Call the function with the data matrices, titles, and grid dimensions
#plot_multiple_imshow(*data_list, titles=titles, rows=2, cols=4)





#import generateLosMatrix
#loss= generateLosMatrix.generate_line_of_sight(obstacle_matrix)
#visualize_matrix(loss)
#file_path = os.path.join(loss, 'Los.csv')
#np.savetxt(file_path, obstacle_matrix, delimiter=',', fmt='%f' )
# Assuming 'indoor_environment' is the generated 400x400 matrix
