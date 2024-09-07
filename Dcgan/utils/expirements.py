import numpy as np
from collections import namedtuple
import os
from . import solver

def solveMaps(maps,fixedbool=True):
    results = []
    Result = namedtuple("Result", "nSol, nMoves")
    sMaps=[]
    for i,map in enumerate(maps):
        nSol,nMoves=0,0
        if fixedbool:
            map = np.argmax(map.cpu().detach().numpy(), axis=0)
        nplayers=np.column_stack(np.where(map == 5))
        ndoors=np.column_stack(np.where(map == 4))
        if len(nplayers) == 1 and len(ndoors) == 1:
            print(f"Start solving map {i}")
            nSol,nMoves=solver.nSolutions(map)
        if nSol>0:
            results.append(Result(nSol=nSol, nMoves=nMoves))
            sMaps.append(map)
    return results,sMaps

def getDoors( matrix):
    return np.where(matrix == 4)

def getPlayers( matrix):
    return np.where(matrix == 5)

def getMapsWoDoP(maps): #getMapsWithOutDoorsOrPlayer
    mDoor= 0
    mPlayer= 0
    for map in maps:
        map = np.argmax(map.cpu().detach().numpy(), axis=0)
        nplayers=np.column_stack(np.where(map == 5))
        ndoors=np.column_stack(np.where(map == 4))
        if len(nplayers) > 0:
            mPlayer+=1
        if len(ndoors) > 0:
            mDoor+=1
    return mDoor, mPlayer

def fixMaps(maps):
    fixedMaps = []
    for map in maps:
        map=map.cpu().detach().numpy()
        auxMap = np.argmax(map, axis=0)
        npla=np.column_stack(np.where(auxMap == 5))
        ndor=np.column_stack(np.where(auxMap == 4))
        if len(npla) == 0 :
            playerCoor = np.unravel_index(np.argmax(map[5]), (10,10))
            auxMap[playerCoor[0],playerCoor[1]]= 5
        if len(ndor) == 0:
            doorCoor = np.unravel_index(np.argmax(map[4]), (10,10))
            auxMap[doorCoor[0],doorCoor[1]]=4
        if len(npla) > 1:
             nplayers=np.column_stack(np.where(auxMap == 5))
             idx = np.random.randint(0,len(nplayers))
             aux=0
             for i in nplayers:
                if aux!=idx:
                    auxMap[i[0],i[1]]=0
                aux+=1
        elif len(ndor) > 1:
             nDoors=np.column_stack(np.where(auxMap == 4))
             idx = np.random.randint(0,len(nDoors))
             aux=0
             for i in nDoors:
                if aux!=idx:
                    auxMap[i[0],i[1]]=0
                aux+=1
        fixedMaps.append(auxMap)
    return  fixedMaps

def save_matrix_to_file(matrix, filename):
    with open(filename, 'w') as file:
        for row in matrix:
            line = ' '.join(map(str, row))
            file.write(line + '\n')

def save_matrices_to_files(matrices, base_filename):
    for index, matrix in enumerate(matrices):
        filename = f"{base_filename}dungeon_{index + 1}.txt"
        save_matrix_to_file(matrix, filename)

def save_data_and_matrices(data_list, matrices, base_filename):
    save_matrices_to_files(matrices, base_filename)
    with open(f"{base_filename}dungeonResults.txt", 'w') as file:
        for index, data in enumerate(data_list):
            matrix_file = f"dungeon_{index + 1}"
            file.write(f"Number of solutions:{data.nSol}, Minimum moves: {data.nMoves} , Matrix file: {matrix_file} \n")

def createFolder(nombre_carpeta):
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)

def saveDoorAndplayerData(datos, filename):
    with open(filename, 'a') as file:
        for line in datos:
            file.write(line + '\n')

def experiment(maps,i):
    fixedMaps = fixMaps(maps)
    resultsPath= f"Dcgan/results/resultsGan.txt"
    dungeonsPath = f"Dcgan/results/dungeons{i}"
    createFolder(dungeonsPath)
    createFolder(f"{dungeonsPath}/dungeons")
    createFolder(f"{dungeonsPath}/fixedDungeons")
    mapNdoors, mapNplayers = getMapsWoDoP(maps)
    rsolvedmaps,sMaps = solveMaps(maps)
    rsolvedFixedMaps,sFixedMaps = solveMaps(fixedMaps,False)
    results=[f"Experiment n:{i}",f"Maps with doors: {mapNdoors} Maps with players: {mapNplayers}", 
             f"Maps solved: {len(sMaps)} Maps solved with fixed doors and players: {len(sFixedMaps)}"]
    saveDoorAndplayerData(results, resultsPath)
    save_data_and_matrices(rsolvedmaps, sMaps, f"{dungeonsPath}/dungeons/")
    save_data_and_matrices(rsolvedFixedMaps, sFixedMaps, f"{dungeonsPath}/fixedDungeons/")
    print("Experiment finished")



