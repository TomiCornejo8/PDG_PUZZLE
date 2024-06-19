import numpy as np
import os
import glob
import time as T
from modules import solver

def read_solutions(folder_path):
    # Obtén la lista de todos los archivos CSV en la carpeta
    csv_files = glob.glob(os.path.join(folder_path, 'Solution_*.csv'))

    # Inicializa una lista para guardar las matrices
    matrices = []

    # Lee cada archivo CSV y convierte su contenido a una matriz de NumPy
    for file in csv_files:
        matrix = np.loadtxt(file, delimiter=',')
        matrices.append(matrix)
    
    return matrices

folder_path = 'results/Experiment 18-06/SolutionsCsv'
matrices = read_solutions(folder_path)

start = T.time()
print(f"Comienzo del calculo de movimientos y jugabilidad entre {len(matrices)} mapas")

meanMoves = 0
meanPlayable = 0
for mapa in matrices:
    n, moves = solver.nSolutions(mapa)
    if n != 0: 
        meanPlayable += 1
        meanMoves += moves

meanMoves = meanMoves / meanPlayable
meanPlayable = meanPlayable / len(matrices)

meanPlayable *= 100

print(f"Movimientos {meanMoves} Jugabilidad {meanPlayable}")
print(f"Se termino de ejecutar en {T.time() - start} segundos")