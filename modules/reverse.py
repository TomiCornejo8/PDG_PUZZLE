from modules import mechanics as M
import numpy as np
import random

# Functions
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

def setDoor(dungeon):
    allowDoorPosition = getAdjacentWalls(dungeon)
    door = random.choice(allowDoorPosition)
    dungeon[door[0],door[1]] = M.DOOR
    return dungeon,door

def setHansRoute(dungeon,route):
    stop = random.choice(route)
    


def getDungeon(dungeon,maxMoves):
    dungeon,door= setDoor(dungeon)
    
    return(getAdjacentWalls(dungeon))