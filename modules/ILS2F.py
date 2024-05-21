import numpy as np
import random

from modules import mechanics as M
from modules import solver

def fitness(nSol, minMoves, w1= 1 , w2 = 1):
    combined_score = w2 * minMoves - w1 * nSol
    return combined_score

def isFeasible(nSol):
    if nSol > 0: return True
    return False

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
    nSol,minMoves = solver.nSolutions(solucion)
    mejor_fitness = fitness(nSol,minMoves)
    
    for _ in range(100):  # Número de intentos de mejora
        nueva_solucion = perturbation(solucion.copy())
        if isFeasible(nSol):
            nSol,minMoves = solver.nSolutions(solucion)
            nuevo_fitness = fitness(nSol,minMoves)
            if nuevo_fitness > mejor_fitness:  # Suponiendo que mayor fitness es mejor
                mejor_solucion = nueva_solucion
                mejor_fitness = nuevo_fitness
    nSol,minMoves = solver.nSolutions(mejor_solucion)            
    return mejor_solucion,nSol,minMoves

def ils(solucion_inicial, iteraciones):
    mejor_solucion,nSol,minMoves = busqueda_local(solucion_inicial)
    mejor_fitness = fitness(nSol,minMoves)
    
    for it in range(1,iteraciones):
        solucion_perturbada = perturbation(mejor_solucion.copy())
        nSol,minMoves = solver.nSolutions(solucion_perturbada)

        if isFeasible(nSol):
            mejor_solucion,nSol,minMoves = busqueda_local(solucion_perturbada)
            fitness_perturbada = fitness(nSol,minMoves)
            
            if fitness_perturbada > mejor_fitness:
                mejor_solucion = solucion_perturbada
                mejor_fitness = fitness_perturbada
        print(f"iter {it} fitness: {mejor_fitness}")
    
    return mejor_solucion, mejor_fitness