import os
import pandas as pd
import numpy as np

def load_data_from_folder():
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
                data_6_channels = np.zeros((6, 10, 10))  # Inicializar una matriz vacía de 10x10x6
                
                # Asignar valores a cada canal según las entidades
                for i in range(6):
                    data_6_channels[i, :, :] = (data == i).astype(np.float32)
                
                data_list.append(data_6_channels)
    return np.array(data_list)

