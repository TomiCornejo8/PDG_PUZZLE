import os
import pandas as pd
import numpy as np
import tensorflow as tf

def load_data_from_folder(channels):
    resultPath = 'Results'
    resultsFolders = os.listdir(resultPath)
    data_list = []
    for experi in resultsFolders:
        path = os.path.join(experi, 'SolutionsCsv')
        folder_path = os.path.join(resultPath, path)

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, header=None)

                data_channels = []
                
                matrix_with_border = np.pad(df, ((1, 1), (1, 1)), mode='constant')
                for i in range(channels):
                    channel = np.where(matrix_with_border == i, 1, 0)
                    data_channels.append(channel)
                data_channels =np.moveaxis(data_channels, [0, 1, 2], [1, 2, 0])
                data_list.append(np.array(data_channels))
    data_array = np.array(data_list)
    data_tensor = tf.convert_to_tensor(data_array, dtype=tf.float32)
    return np.array(data_tensor)
