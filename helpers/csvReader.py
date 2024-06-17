import os
import pandas as pd
import numpy as np
import torch 


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


def load_data_from_folder_RL_test():
    resultPath = 'Results'
    resultsFolders = os.listdir(resultPath)
    data_list = []
    maxW,maxH=0,0
    for experi in resultsFolders:
        path = os.path.join(experi, 'SolutionsCsv')
        folder_path = os.path.join(resultPath, path)

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, header=None)
                df=np.array(df)
                mWith,mHeight = df.shape
                if maxW < mWith:
                    maxW = mWith
                if maxH < mHeight:
                    maxH = mHeight
                data_list.append(df)

    # data_array = np.array(data_list)
    # data_tensor = tf.convert_to_tensor(data_array, dtype=tf.float32)
    maxDim=0
    if maxW >= maxH:
        maxDim = maxW
    else:
        maxDim = maxH
    data_listPadded=[]

    if maxDim % 2 != 0:
        maxDim+=1
    file=''
    for maxtrix in data_list:
        mWith,mHeight = maxtrix.shape
        if mWith >= mHeight:
            maxLDim =mWith 
        else:
            maxLDim =mHeight 
        squareMatrix = np.ones((maxLDim, maxLDim), dtype=maxtrix.dtype)
        dimM = maxDim - maxLDim 
        padedMatrix = np.pad(squareMatrix, ((dimM//2,dimM//2),(dimM//2,dimM//2)), 'constant', constant_values=(1))
        data_listPadded.append(padedMatrix)
    return data_listPadded,maxDim

def load_data_from_folder_RL():
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
    return data_list

def load_data_from_folderTor(channels):
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
                for i in range(channels):
                    channel = np.where(df.values == i, 1, 0)
                    data_channels.append(channel)
                
                data_list.append(np.array(data_channels))
    data_array = np.array(data_list)
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    return np.array(data_tensor)