import pandas as pd
import numpy as np
import os

def load_data_from_folder():
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
                df=np.array(df)
                data_list.append(df)
                for _ in range(3):
                  rotateMatrix=np.rot90(df)
                  data_list.append(rotateMatrix)
                  df=rotateMatrix
    return data_list

