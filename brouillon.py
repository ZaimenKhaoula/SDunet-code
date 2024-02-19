import numpy as np
import keras.backend as K

def root_mean_squared_error(y_true, y_pred, axis=None):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    return np.sqrt(np.mean(np.square(y_pred_np - y_true_np), axis=axis))

# Example usage
y_true_matrix = np.array([[1, 20], [3, 4]])
y_pred_matrix = np.array([[2, 2], [13, 3]])

rmse_with_axis_none = root_mean_squared_error(y_true_matrix, y_pred_matrix, axis=None)
rmse_without_axis = root_mean_squared_error(y_true_matrix, y_pred_matrix)  # Defaults to axis=None

print("RMSE with axis=None:", rmse_with_axis_none)
print("RMSE without specifying axis:", rmse_without_axis)
