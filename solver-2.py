import heapq
import numpy as np
from collections import deque
from searchBase import solver as sol
import time
import pygame

import glob
import os
import pandas as pd

# Definición del movimiento en 4 direcciones (arriba, abajo, izquierda, derecha)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Función que verifica si una posición está dentro de los límites de la matriz y no es una pared
def is_valid_move(grid, x, y):
    rows, cols = grid.shape
    return 0 <= x < rows and 0 <= y < cols and grid[x, y] == 0

# Función que realiza el movimiento en línea recta hasta chocar con una pared
def move_until_hit(grid, start_x, start_y, move_x, move_y):
    x, y = start_x, start_y
    while is_valid_move(grid, x + move_x, y + move_y):
        x += move_x
        y += move_y
    return x, y

def is_goal(current,end):
    for move in MOVES:
        x_move = current[0] + move[0]
        y_move = current[1] + move[1]
        if x_move == end[0] and y_move == end[1]:
            return True
    return False

# Heurística (distancia de Manhattan)
def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

# Implementación del algoritmo A*
def a_star(grid, start, end):
    # El open set (frontera de búsqueda) usa una cola de prioridad basada en la función f(n) = g(n) + h(n)
    open_set = []
    heapq.heappush(open_set, (0, start))

    # Diccionarios para rastrear el costo hasta llegar a un nodo y el nodo desde donde se llegó
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        # Si llegamos al objetivo, devolvemos el camino
        if is_goal(current,end):
            return current

        current_x, current_y = current

        # Para cada movimiento (arriba, abajo, izquierda, derecha), se mueve hasta chocar con una pared
        for move_x, move_y in MOVES:
            new_x, new_y = move_until_hit(grid, current_x, current_y, move_x, move_y)
            neighbor = (new_x, new_y)
            tentative_g_score = g_score[current] + 1  # Cada movimiento cuesta 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + manhattan_dist(new_x, new_y, end[0], end[1])
                heapq.heappush(open_set, (f_score, neighbor))

    return None  # No se encontró un camino

# Implementación del algoritmo BFS
def bfs(grid, start, end):
    # Cola para los nodos a explorar, inicializada con la posición de inicio
    queue = deque([start])
    
    # Diccionario para rastrear las posiciones visitadas
    visited = set([start])

    # Mientras haya nodos por explorar
    while queue:
        current = queue.popleft()

        # Si el nodo actual está adyacente a la meta, devolvemos la posición actual
        if is_goal(current, end):
            return current

        current_x, current_y = current

        # Para cada dirección, nos movemos hasta chocar con una pared
        for move_x, move_y in MOVES:
            new_x, new_y = move_until_hit(grid, current_x, current_y, move_x, move_y)
            neighbor = (new_x, new_y)

            # Si el vecino no ha sido visitado, lo añadimos a la cola y marcamos como visitado
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return None  # Si no se encuentra un camino, se devuelve None

# Implementación del algoritmo DFS
def dfs(grid, start, end):
    # Pila para manejar los nodos a explorar (LIFO), inicializada con la posición de inicio
    stack = [(start)]
    
    # Conjunto para rastrear las posiciones visitadas
    visited = set([start])

    # Mientras haya nodos por explorar
    while stack:
        current = stack.pop()

        # Si el nodo actual está adyacente a la meta, devolvemos la posición actual
        if is_goal(current, end):
            return current

        current_x, current_y = current

        # Para cada dirección, nos movemos hasta chocar con una pared
        for move_x, move_y in MOVES:
            new_x, new_y = move_until_hit(grid, current_x, current_y, move_x, move_y)
            neighbor = (new_x, new_y)

            # Si el vecino no ha sido visitado, lo añadimos a la pila y marcamos como visitado
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return None  # Si no se encuentra un camino, se devuelve None

def dijkstra(grid, start, end):
    # Cola de prioridad para los nodos a explorar
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))  # La cola almacena tuplas (costo, nodo)

    # Diccionario para rastrear las posiciones visitadas y el costo mínimo hasta ellas
    visited = {start: 0}

    while priority_queue:
        current_cost, current = heapq.heappop(priority_queue)

        # Si llegamos a una casilla adyacente al objetivo, devolvemos la posición actual
        if is_goal(current, end):
            return current

        current_x, current_y = current

        # Para cada dirección, nos movemos hasta chocar con una pared
        for move_x, move_y in MOVES:
            new_x, new_y = move_until_hit(grid, current_x, current_y, move_x, move_y)
            neighbor = (new_x, new_y)
            new_cost = current_cost + 1  # Cada movimiento tiene un costo uniforme de 1

            # Si el vecino no ha sido visitado o encontramos un costo menor, lo añadimos a la cola de prioridad
            if neighbor not in visited or new_cost < visited[neighbor]:
                visited[neighbor] = new_cost
                heapq.heappush(priority_queue, (new_cost, neighbor))

    return None  # Si no se encuentra un camino, se devuelve None

def straight_line_with_heuristics(grid, start, end):
    # Cola de prioridad para los nodos a explorar
    open_set = []
    heapq.heappush(open_set, (0, start))  # Almacena tuplas (heurística, nodo)

    # Diccionario para rastrear el costo mínimo hasta cada posición
    g_score = {start: 0}

    while open_set:
        # Extrae el nodo con el valor más bajo de f_score (más cercano al objetivo)
        _, current = heapq.heappop(open_set)

        # Si llegamos a una casilla adyacente a la meta, devolvemos la posición actual
        if is_goal(current, end):
            return current

        current_x, current_y = current

        # Para cada movimiento en línea recta (arriba, abajo, izquierda, derecha)
        for move_x, move_y in MOVES:
            # Realizamos el movimiento en línea recta hasta chocar con una pared o borde
            new_x, new_y = move_until_hit(grid, current_x, current_y, move_x, move_y)
            neighbor = (new_x, new_y)

            # Calculamos el nuevo costo g(n) de llegar al vecino
            tentative_g_score = g_score[current] + 1  # Cada movimiento en línea recta tiene un costo de 1

            # Si el vecino no ha sido visitado o encontramos un costo menor, lo actualizamos
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                # Calculamos f(n) = g(n) + h(n), donde h(n) es la heurística (distancia de Manhattan)
                f_score = tentative_g_score + manhattan_dist(new_x, new_y, end[0], end[1])
                heapq.heappush(open_set, (f_score, neighbor))

    return None  # Si no se encuentra un camino, se devuelve None

def getRoute(state,player,goal):
    # return straight_line_with_heuristics(state, to_tuple(player), to_tuple(goal))
    # return dijkstra(state, to_tuple(player), to_tuple(goal))
    # return dfs(state, to_tuple(player), to_tuple(goal))
    # return bfs(state, to_tuple(player), to_tuple(goal))
    return a_star(state, to_tuple(player), to_tuple(goal))

def to_tuple(tile):
    return (tile[0],tile[1])

def step(grid,enemy,newPlayer):
    player = np.argwhere(grid == 5)[0]
    grid[player[0],player[1]] = 0
    grid[enemy[0],enemy[1]] = 0
    grid[newPlayer[0],newPlayer[1]] = 5
    return grid

def solver(grid):
    states = deque()
    states.append(grid.copy())
    while states:
        currentState = states.popleft()
        player = np.argwhere(currentState == 5)[0]
        enemys = np.argwhere(currentState == 3)

        if len(enemys) == 0:
            door = np.argwhere(currentState == 4)[0]
            newPlayer = getRoute(currentState,player,door)
            if newPlayer is not None: return True
        else:
            for enemy in enemys:
                newPlayer = a_star(currentState, to_tuple(player), to_tuple(enemy))
                if newPlayer is not None:
                    newState = step(currentState.copy(),enemy,newPlayer)
                    states.append(newState.copy())
    return False

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

matrices = read_solutions('results/Experiment 30-05/SolutionsCsv')
datos = []

# Matriz problema => 37

for iter,matriz in enumerate(matrices):
    if (iter+1) > 50: break
    if (iter+1) == 37: continue

    print(f"Dungeon {iter+1}",end=" ")

    start = time.time()
    solB,_ = sol.nSolutions(matriz)
    T_viejo = time.time()-start

    print(f"Tiempo viejo:{T_viejo:.2f}",end=" ")

    start = time.time()
    solA = solver(matriz)
    T_nuevo = time.time()-start

    print(f"Tiempo nuevo:{T_nuevo:.2f}")

    if solA == True and solB == 0: err = 1
    elif solA == False and solB > 0: err = 1
    else: err = 0
    
    datos.append({
        'T_Nuevo':T_nuevo,
        'T_Viejo':T_viejo,
        'Err': err,
        'Sol_Nuevo':solA,
        'Sol_Viejo':solB
    })

df = pd.DataFrame(datos)

df.to_csv('result-bulk.csv', index=False)

# with open('result-mean.txt', 'w') as file:
#     file.write(f"MEDIA\n{df[['T_Nuevo','T_Viejo']].mean()}")
#     file.write(f"STD\n{df[['T_Nuevo','T_Viejo']].std()}")
#     file.write(f"ERROR\n{df['Err'].sum()}")

pygame.mixer.init()
pygame.mixer.music.load('helpers/a.mp3')
print("TERMINO LA EJECUCIÓN")
pygame.mixer.music.play()
time.sleep(1.5)