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

solutions,nodo = S.solve(valid2)
print(len(solutions))

# display_tree(nodo)

it = 0
for solution in solutions:
    it += 1
    print(it,end="\n------------------------------------\n")
    while solution:
        print(solution.pop(),end="\n\n")

print(f"Dungeon resuelto en {T.time() - inicio} segundos")



