import os
import pandas as pd
import numpy as np
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_data_from_folders():
    resultPath= 'Results'
    resultsFolders = os.listdir(resultPath)
    data_list = []
    for experi in resultsFolders:
        path = os.path.join(experi,'SolutionsCsv')
        folder_path = os.path.join(resultPath, path)

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, header=None)
                data = df.values.astype(np.float32)  # Convertir los datos a float32
                data_6_channels = np.zeros((2, 10, 10))  # Inicializar una matriz vacía de 10x10x6
                # Asignar valores a cada canal según las entidades
                for i in range(2):
                    data_6_channels[i, :, :] = (data == i).astype(np.float32)
                
                data_list.append(data_6_channels)
    return np.array(data_list)



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
                
                # Redimensionar la matriz a 14x14
                data_14x14 = np.zeros((14, 14))
                data_14x14[2:12, 2:12] = df.values  # Colocar los valores del CSV en el centro de la matriz 14x14
            

                data_channels = []
                for i in range(channels):
                    channel = np.where(data_14x14 == i, 1, 0)
                    data_channels.append(channel)
                
                data_list.append(np.array(data_channels))
    data_array = np.array(data_list)
    data_tensor = tf.convert_to_tensor(data_array, dtype=tf.float32)
    return np.array(data_tensor)