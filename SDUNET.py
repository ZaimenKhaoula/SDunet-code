import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from scipy.ndimage import rotate
import numpy as np
import os


def get_subfolders( directory):
    subfolders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
            subfolders.extend(get_subfolders(item_path))
    return subfolders

def generateDataset():
    X_data=[]
    Y_data=[]
    root_dirs=["computedTensors\\basic_t0", "computedTensors\\basic_t1", "computedTensors\\basic_t2", "computedTensors\\basic_t3", "computedTensors\\basic_t4", "computedTensors\\basic_t5", "computedTensors\\basic_t6", "computedTensors\\basic_t7",
               "computedTensors\\basic_t8", "computedTensors\\basic_t9", "computedTensors\\basic_t10", "computedTensors\\basic_t11", "computedTensors\\basic_t12"]
    for root_dir in root_dirs:
        subfolders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
        for subfolder in subfolders:
            input_files = ['conductivity.csv', 'distance.csv', 'Los.csv', 'permittivity.csv']
            input_tensor = []
            for file in input_files:
                file_path = os.path.join(root_dir, subfolder, file)
                matrix = np.loadtxt(file_path, delimiter=',')
                input_tensor.append(matrix)

            target_file = os.path.join(root_dir, subfolder, 'rss_map.csv')
            target_matrix = np.loadtxt(target_file, delimiter=',')
            X_data.append(input_tensor)
            Y_data.append(target_matrix)
    root_dirs = ["computedTensors\\complexe_t0", "computedTensors\\complexe_t1", "computedTensors\\complexe_t2",
                 "computedTensors\\complexe_t3", "computedTensors\\complexe_t4", "computedTensors\\complexe_t5",
                 "computedTensors\\complexe_t6", "computedTensors\\complexe_t7",
                 "computedTensors\\complexe_t8", "computedTensors\\complexe_t9", "computedTensors\\complexe_t10",
                 "computedTensors\\complexe_t11", "computedTensors\\complexe_t12", "computedTensors\\complexe_t13", "computedTensors\\complexe_t14"
                 , "computedTensors\\complexe_t15", "computedTensors\\complexe_t16", "computedTensors\\complexe_t17", "computedTensors\\complexe_t18",
                 "computedTensors\\complexe_t19", "computedTensors\\complexe_t20", "computedTensors\\complexe_t22", "computedTensors\\complexe_t23"
                 , "computedTensors\\complexe_t24"]
    for root_dir in root_dirs:
        input_files = ['conductivity.csv', 'distance.csv', 'Los.csv', 'permittivity.csv']
        input_tensor = []
        for file in input_files:
            file_path = os.path.join(root_dir, file)
            matrix = np.loadtxt(file_path, delimiter=',')
            input_tensor.append(matrix)

        target_file = os.path.join(root_dir, 'rss_map.csv')
        target_matrix = np.loadtxt(target_file, delimiter=',')
        X_data.append(input_tensor)
        Y_data.append(target_matrix)

    return X_data, Y_data


def flip_left(original_matrix):
    flipped_left_matrix = np.fliplr(original_matrix)
    return flipped_left_matrix



def flip_downward(original_matrix):
    flipped_down_matrix = np.flipud(original_matrix)
    return flipped_down_matrix



def rotate_180(original_matrix):
    rotated_180 = np.rot90(original_matrix, k=2)
    return rotated_180

def rotate_90(original_matrix):
    rotated_90 = np.rot90(original_matrix, k=1)
    return rotated_90

def rotate_270(original_matrix):
    rotated_270 = np.rot90(original_matrix, k=3)
    return rotated_270


def augment_and_split_data():
    pass



def sdunet_block(input, n_filters):
    input = Conv2D(n_filters//2, 3, dilation_rate=(1, 1), padding="same", activation="relu",
                   kernel_initializer = "he_normal")(input)

    input = Conv2D(n_filters//4, 3, dilation_rate=(3, 3), padding="same", activation="relu",
                   kernel_initializer="he_normal")(input)
    input = Conv2D(n_filters//8, 3, dilation_rate=(6, 6), padding="same", activation="relu",
                   kernel_initializer="he_normal")(input)
    input = Conv2D(n_filters//16, 3, dilation_rate=(9, 9), padding="same", activation="relu",
                   kernel_initializer="he_normal")(input)
    input = Conv2D(n_filters//16, 3, dilation_rate=(12, 12), padding="same", activation="relu",
                   kernel_initializer="he_normal")(input)

    return input



def dilated_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   f = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   batch_norm1 = BatchNormalization()(f)
   # Conv2D then ReLU activation
   f = sdunet_block(batch_norm1, n_filters)

   return f




def downsample_block(x, n_filters):
   f = dilated_conv_block(x, n_filters)
   p = MaxPool2D(2)(f)
   #p = Dropout(0.3)(p)
   return f, p


def upsample_block(x, conv_features, n_filters):
   # upsample
   x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = concatenate([x, conv_features])
   # dropout
   #x = Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = dilated_conv_block(x, n_filters)
   return x


def unet():
    inputs = Input(shape=(400, 400, 4))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = dilated_conv_block(p4, 512)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = Conv2D(1, 1, padding="same", activation="linear")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model



unet_model=unet()

def root_mean_squared_error(y_true, y_pred):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    return np.sqrt(np.mean(np.square(y_pred_np - y_true_np)))


unet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss=root_mean_squared_error,
    metrics=[
        keras.metrics.mean_absolute_error,
        keras.metrics.mean_absolute_percentage_error,
        keras.metrics.RootMeanSquaredError,
    ],
)



x_train, y_train, x_val, y_val, x_test, y_test = augment_and_split_data()

nb_batch_size = 4
nb_epoch= 250

history = unet_model.fit(x_train, y_train, epochs=nb_epoch, batch_size=nb_batch_size, validation_data=(x_val, y_val))

#save the history of evaluation to plot them after
import pickle

#write
with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

#read
#with open('/trainHistoryDict', "rb") as file_pi:
#    history = pickle.load(file_pi)
evaluation = unet_model.evaluate(x_test, y_test, batch_size=nb_batch_size)

# Extracting loss and metrics values
loss_value = evaluation[0]
metric_values = evaluation[1:]

# Create a string with the loss and metrics information
evaluation_string = f'Loss: {loss_value}\n'
for i, metric_value in enumerate(metric_values):
    metric_name = unet_model.metrics_names[i]  # Include the loss, no need to skip
    evaluation_string += f'{metric_name}: {metric_value}\n'

# Save the evaluation string to a file
with open('evaluation_results.txt', 'w') as file:
    file.write(evaluation_string)
unet_model.save("saved-model")
#unet_model = tf.keras.models.load_model("saved-model")


predictions = unet_model.predict(x_test)
with open('predictions.pkl', 'wb') as file:
    pickle.dump(predictions, file)