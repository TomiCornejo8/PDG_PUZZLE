import numpy as np
import time as T
import csv
from anytree import Node,RenderTree
from collections import deque
from utils import colorRender as color
from modules import solver as S

def csv_to_numpy_matrix(file):
    with open(f'results/{file}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
    
    # Convertimos los datos leídos a un array de NumPy
    numpy_matrix = np.array(data, dtype=int)
    return numpy_matrix


inicio = T.time()
print(f"inicia el solver")


valid1 = np.array([
    [1,1,1,1,1,1],
    [1,2,0,3,2,1],
    [1,3,0,0,4,1],
    [1,2,0,3,1,1],
    [1,5,0,2,1,1],
    [1,1,1,1,1,1]
])
valid2 = np.array([
    [1,1,1,1,1],
    [1,1,1,2,1],
    [1,4,0,0,1],
    [1,0,3,5,1],
    [1,1,1,1,1]
])
invalid = np.array([
    [1,1,1,1,1,1],
    [1,2,0,0,0,1],
    [1,0,0,2,4,1],
    [1,0,2,3,1,1],
    [1,5,2,0,1,1],
    [1,1,1,1,1,1]
])

dungeon = csv_to_numpy_matrix()

solutions = S.solve(dungeon)

if len(solutions) > 0:
    best = solutions[0]
    for sol in solutions:
        if len(sol) < len(best):
            best = sol
    color.renderMatrixQueue(best)

print(f"Dungeon resuelto en {T.time() - inicio} segundos")