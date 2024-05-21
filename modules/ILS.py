import numpy as np
import random

from modules import mechanics as M
from modules import solver

"""
Función de Objetivo de Equilibrio de Obstáculos:
    Esta función mide el balance entre los diferentes tipos de obstáculos (bloques, enemigos, paredes)
    en el mapa.
"""
def fitness_obstacle_balance(dungeon):
    total_cells = dungeon.size
    num_blocks = np.sum(dungeon == M.BLOCK)
    num_enemies = np.sum(dungeon == M.ENEMY)
    return 1 - abs(num_blocks / total_cells - num_enemies / total_cells)

"""
Función de Objetivo de Distancia Promedio al Enemigo:
    Esta función mide la distancia promedio desde cada celda del jugador 
    a los enemigos. Un mapa donde los enemigos están demasiado lejos puede ser menos desafiante.
    Ahora modificada para aumentar la distancia.
"""
def fitness_enemy_distance(dungeon):
    player_pos = np.argwhere(dungeon == M.PLAYER)[0]
    enemy_positions = np.argwhere(dungeon == M.ENEMY)
    if enemy_positions.size == 0:
        return 0
    distances = [np.linalg.norm(player_pos - enemy) for enemy in enemy_positions]
    avg_distance = np.mean(distances)
    return avg_distance

"""
Función de Objetivo de Distancia a la Puerta:
    Esta función mide la distancia desde el jugador a la puerta. 
    Una mayor distancia puede hacer el juego más desafiante.
"""
def fitness_door_distance(dungeon):
    player_pos = np.argwhere(dungeon == M.PLAYER)[0]
    door_pos = np.argwhere(dungeon == M.DOOR)[0]
    distance = np.linalg.norm(player_pos - door_pos)
    return distance

"""
Función de Objetivo de Cercanía de los Bloques:
    Esta función mide la cercanía de los bloques al jugador.
    Una menor distancia puede hacer el juego más desafiante.
"""
def fitness_block_proximity(dungeon):
    player_pos = np.argwhere(dungeon == M.PLAYER)[0]
    block_positions = np.argwhere(dungeon == M.BLOCK)
    if block_positions.size == 0:
        return 0
    distances = [np.linalg.norm(player_pos - block) for block in block_positions]
    avg_distance = np.mean(distances)
    return 1 / avg_distance if avg_distance > 0 else 0

def fitness(dungeon, w1=1.0, w2=0.1, w3=0.8, w4=0.1):
    fitness_ob = fitness_obstacle_balance(dungeon)
    fitness_ed = fitness_enemy_distance(dungeon)
    fitness_dd = fitness_door_distance(dungeon)
    fitness_bp = fitness_block_proximity(dungeon)
    
    # Combining the fitness scores
    # combined_score = w1 * fitness_ob + w2 * fitness_ed + w3 * fitness_dd + w4 * fitness_bp
    combined_score = w1 * fitness_ob  + w3 * fitness_dd + w4 * fitness_bp
    return combined_score

# def isFeasible(dungeon):
#     solutions = solver.solve(dungeon.copy())
#     if len(solutions) == 0: return False
#     return True

def isFeasible(dungeon):
    return solver.playable(dungeon)

def getAdjacentWalls(dungeon):
    # Obtener las posiciones de los 1s en la dungeon
    walls = np.argwhere(dungeon == M.WALL)
    adjacentWalls = []

    for i, j in walls:
        # Verificar si el 1 está adyacente a un 0 en forma de cruz
        if (i > 0 and dungeon[i-1][j] == M.EMPTY) or \
           (i < dungeon.shape[0]-1 and dungeon[i+1][j] == M.EMPTY) or \
           (j > 0 and dungeon[i][j-1] == 0) or \
           (j < dungeon.shape[1]-1 and dungeon[i][j+1] == M.EMPTY):
            adjacentWalls.append((i, j))

    return adjacentWalls

def perturbation(dungeon):
    tiles = np.argwhere((dungeon != M.WALL) & (dungeon != M.EMPTY))
    tile = random.choice(tiles)

    mutateTile = dungeon[tile[0],tile[1]]

    if mutateTile == M.DOOR:
        walls = getAdjacentWalls(dungeon)
        newDoor = random.choice(walls)
        dungeon[tile[0],tile[1]] = M.WALL
        dungeon[newDoor[0],newDoor[1]] = M.DOOR
    elif np.random.rand() <= 0.5 and mutateTile in [M.ENEMY,M.BLOCK]:
        if mutateTile == M.ENEMY:
            dungeon[tile[0],tile[1]] = M.BLOCK
        else:
            dungeon[tile[0],tile[1]] = M.ENEMY
    else:
        emptys = np.argwhere(dungeon == M.EMPTY)
        newEntity = random.choice(emptys)
        dungeon[tile[0],tile[1]] = M.EMPTY
        if mutateTile == M.PLAYER:
            dungeon[newEntity[0],newEntity[1]] = M.PLAYER
        else:
            dungeon[newEntity[0],newEntity[1]] = random.choice([M.ENEMY,M.BLOCK])
    return dungeon

def busqueda_local(solucion):
    mejor_solucion = solucion.copy()
    mejor_fitness = fitness(solucion)
    
    for _ in range(100):  # Número de intentos de mejora
        nueva_solucion = perturbation(solucion.copy())
        if isFeasible(nueva_solucion):
            nuevo_fitness = fitness(nueva_solucion)
            if nuevo_fitness > mejor_fitness:  # Suponiendo que mayor fitness es mejor
                mejor_solucion = nueva_solucion
                mejor_fitness = nuevo_fitness
                
    return mejor_solucion

def ils(solucion_inicial, iteraciones):
    mejor_solucion = busqueda_local(solucion_inicial)
    mejor_fitness = fitness(mejor_solucion)
    
    for it in range(iteraciones):
        print(f"iter {it} fitness: {mejor_fitness}")
        solucion_perturbada = perturbation(mejor_solucion.copy())
        if isFeasible(solucion_perturbada):
            solucion_perturbada = busqueda_local(solucion_perturbada)
            fitness_perturbada = fitness(solucion_perturbada)
            
            if fitness_perturbada > mejor_fitness:
                mejor_solucion = solucion_perturbada
                mejor_fitness = fitness_perturbada
    
    return mejor_solucion, mejor_fitness
