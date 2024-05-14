import numpy as np
import time as T
from anytree import Node,RenderTree
from collections import deque

from modules import solver as S

def getRoutes(nodo):
    ruta = deque()
    while nodo is not None:
        ruta.append(nodo.name)
        nodo = nodo.parent
    return ruta

def display_tree(tree_root):
    for pre, _, node in RenderTree(tree_root):
        # Convertir la matriz a un string y dividir en líneas para añadir el prefijo
        matrix_string = np.array2string(node.name, separator=', ')
        matrix_lines = matrix_string.split('\n')
        # Añadir el prefijo a cada línea de la matriz
        formatted_matrix = '\n'.join(f"{pre}{line}" for line in matrix_lines)
        print(formatted_matrix)

def crossover(parent1, parent2):
    # Obtener las dimensiones de la matriz
    m, n = parent1.shape
    
    # Generar un número aleatorio entre 0 y 1 para determinar si se utiliza la fila o la columna
    if np.random.rand() < 0.5:  # 50% de probabilidad para filas
        # Seleccionar un índice de fila aleatorio
        row_idx = np.random.randint(m)
        # Cruzamiento por fila
        child1 = np.vstack((parent1[:row_idx], parent2[row_idx:]))
        child2 = np.vstack((parent2[:row_idx], parent1[row_idx:]))
    else:  # 50% de probabilidad para columnas
        # Seleccionar un índice de columna aleatorio
        col_idx = np.random.randint(n)
        # Cruzamiento por columna
        child1 = np.hstack((parent1[:, :col_idx], parent2[:, col_idx:]))
        child2 = np.hstack((parent2[:, :col_idx], parent1[:, col_idx:]))
    
    return child1, child2

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

#child1, child2 = crossover(valid1,invalid)

# print(f"valid1\n{valid1}\ninvalid\n{invalid}\nchild1\n{child1}\nchild2\n{child2}")
print(f"{invalid.size} {invalid.shape}")

print(f"Dungeon resuelto en {T.time() - inicio} segundos")